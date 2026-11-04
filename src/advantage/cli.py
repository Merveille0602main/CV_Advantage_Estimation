import argparse
import json
from pathlib import Path


def read_split(dataset,split):
    """Verify IDs, family separation, paths and hashes before training/scoring."""
    from .schema import Episode,file_hash
    root=Path(dataset).resolve()
    manifest=json.loads((root/"manifest.json").read_text())
    membership={}
    for name,ids in manifest["split"].items():
        for eid in ids:
            if eid in membership: raise ValueError("episode appears in multiple splits")
            membership[eid]=name
    records=manifest["episodes"]
    if len({r["episode_id"] for r in records})!=len(records): raise ValueError("duplicate episode ID")
    if set(membership)!={r["episode_id"] for r in records}: raise ValueError("split/record mismatch")
    families={}; result=[]
    for row in records:
        eid=row["episode_id"];family=row["parent_id"];name=membership[eid]
        if family in families and families[family]!=name: raise ValueError("family leakage between splits")
        families[family]=name
        if name!=split: continue
        path=(root/row["path"]).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=row["sha256"]: raise ValueError("dataset path/hash mismatch")
        ep=Episode.load(path)
        if ep.episode_id!=eid or ep.parent_id!=family: raise ValueError("episode identity mismatch")
        result.append(ep)
    if not result: raise ValueError("selected split is empty")
    return result


def main():
    parser = argparse.ArgumentParser(description="TD/N-step offline tools")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("synthetic")
    exp.add_argument("--output", required=True)
    exp.add_argument("--episodes", type=int, default=450)
    exp.add_argument("--epochs", type=int, default=120)
    exp.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    exp.add_argument("--gamma", type=float, default=0.99)
    inspect = sub.add_parser("inspect-rlinf")
    inspect.add_argument("root")
    convert = sub.add_parser("convert-rlinf")
    convert.add_argument("root"); convert.add_argument("--config", required=True)
    convert.add_argument("--output", required=True); convert.add_argument("--seed", type=int, default=0)
    samp = sub.add_parser("convert-samp830")
    samp.add_argument("source_root"); samp.add_argument("--task-index", type=int, required=True)
    samp.add_argument("--output", required=True); samp.add_argument("--work-dir", required=True)
    samp.add_argument("--seed", type=int, default=0)
    samp.add_argument("--task-horizon", type=int, default=500)
    samp.add_argument("--overwrite-work", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("dataset"); train.add_argument("--output", required=True)
    train.add_argument("--n", type=int, default=1); train.add_argument("--gamma", type=float, default=.99)
    train.add_argument("--updates", type=int, default=1000); train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cpu")
    train.add_argument("--method", choices=["mc", "nstep"], default="mc")
    score = sub.add_parser("score")
    score.add_argument("dataset"); score.add_argument("--checkpoint", required=True)
    score.add_argument("--output", required=True); score.add_argument("--ns", type=int, nargs="+", default=[1,5,10,20])
    score.add_argument("--split", choices=["train","validation","test"], default="test")
    qtrain = sub.add_parser("train-q")
    qtrain.add_argument("dataset");qtrain.add_argument("--output",required=True)
    qtrain.add_argument("--gamma",type=float,default=.99);qtrain.add_argument("--updates",type=int,default=1000)
    qtrain.add_argument("--seed",type=int,default=0);qtrain.add_argument("--device",default="cpu")
    qscore = sub.add_parser("score-qv")
    qscore.add_argument("dataset");qscore.add_argument("--q-checkpoint",required=True)
    qscore.add_argument("--v-checkpoint",required=True);qscore.add_argument("--output",required=True)
    qscore.add_argument("--split",choices=["train","validation","test"],default="test")
    qexp = sub.add_parser("synthetic-q")
    qexp.add_argument("--output",required=True);qexp.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    qexp.add_argument("--episodes",type=int,default=450);qexp.add_argument("--updates",type=int,default=1000)
    qexp.add_argument("--anchors",type=int,default=40);qexp.add_argument("--repeats",type=int,default=16)
    smoke = sub.add_parser("smoke-samples")
    smoke.add_argument("samples");smoke.add_argument("--output",required=True)
    smoke.add_argument("--updates",type=int,default=200);smoke.add_argument("--device",default="cpu")
    args = parser.parse_args()
    if args.command == "synthetic":
        from .synthetic import experiment
        report = experiment(args.output, args.seeds, args.episodes, args.epochs, args.gamma)
        print(json.dumps(report["summary"], indent=2))
    elif args.command == "inspect-rlinf":
        from .adapters import inspect_parquet
        print(json.dumps(inspect_parquet(args.root), indent=2))
    elif args.command == "convert-rlinf":
        from .adapters import load_rlinf, write_dataset
        episodes = load_rlinf(args.root, json.loads(Path(args.config).read_text()))
        manifest = write_dataset(episodes, args.output, args.seed)
        print(json.dumps({"episodes": len(episodes), "split_sizes": {k:len(v) for k,v in manifest["split"].items()}}, indent=2))
    elif args.command == "convert-samp830":
        from .samp830 import convert_task
        summary = convert_task(args.source_root, args.task_index, args.output, args.work_dir,
                               seed=args.seed, task_horizon=args.task_horizon,
                               overwrite_work=args.overwrite_work)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.command == "train":
        import torch
        from .schema import Episode, file_hash
        from .value import fit_value
        episodes = read_split(args.dataset,"train")
        if Path(args.output).exists():
            raise ValueError("checkpoint exists; choose a new output path")
        result = fit_value(episodes, n=args.n, gamma=args.gamma, updates=args.updates, seed=args.seed, device=args.device, method=args.method)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, args.output)
        print(json.dumps({"training_episodes": len(episodes), "last_loss": result["losses"][-1]}))

    elif args.command == "score":
        import csv
        from .schema import Episode, file_hash
        from .value import ValuePredictor
        from .targets import n_step_scores
        if Path(args.output).exists() or any(n < 1 for n in args.ns):
            raise ValueError("use a new output file and positive lookaheads")
        root = Path(args.dataset)
        manifest = json.loads((root/"manifest.json").read_text())
        selected = set(manifest["split"][args.split])
        predictor = ValuePredictor(args.checkpoint)
        checkpoint_hash = file_hash(args.checkpoint)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with Path(args.output).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["episode_id","t","method","score","checkpoint_sha256"])
            writer.writeheader()
            for record in manifest["episodes"]:
                if record["episode_id"] not in selected:
                    continue
                path = (root/record["path"]).resolve()
                if not path.is_relative_to(root.resolve()) or file_hash(path) != record["sha256"]:
                    raise ValueError("dataset path or hash mismatch")
                ep = Episode.load(path)
                v, nv = predictor.predict(ep), predictor.predict(ep, True)
                for n in args.ns:
                    scores = n_step_scores(ep,v,nv,n,predictor.checkpoint["gamma"])
                    for t, value in enumerate(scores):
                        writer.writerow(dict(episode_id=ep.episode_id,t=t,method=f"n{n}",
                            score=float(value),checkpoint_sha256=checkpoint_hash))
                        total += 1
        print(json.dumps({"scores":total,"checkpoint_sha256":checkpoint_hash}))

    elif args.command == "train-q":
        import torch
        from .qvalue import fit_q
        episodes = read_split(args.dataset,"train")
        if Path(args.output).exists(): raise ValueError("checkpoint exists")
        checkpoint=fit_q(episodes,gamma=args.gamma,updates=args.updates,seed=args.seed,device=args.device)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        torch.save(checkpoint,args.output)
        print(json.dumps({"train_episodes":len(episodes),"last_loss":checkpoint["losses"][-1]}))
    elif args.command == "score-qv":
        import csv
        from .qvalue import QPredictor,qv_scores
        from .value import ValuePredictor
        from .schema import file_hash
        if Path(args.output).exists(): raise ValueError("output exists")
        episodes=read_split(args.dataset,args.split)
        qp,vp=QPredictor(args.q_checkpoint),ValuePredictor(args.v_checkpoint)
        hashes={"q_sha256":file_hash(args.q_checkpoint),"v_sha256":file_hash(args.v_checkpoint)}
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        with Path(args.output).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=["episode_id","t","method","score",*hashes])
            writer.writeheader()
            for ep in episodes:
                for t,score in enumerate(qv_scores(qp,vp,ep)):
                    writer.writerow(dict(episode_id=ep.episode_id,t=t,method="mc_q_minus_shared_v",score=float(score),**hashes))
        print(json.dumps({"episodes":len(episodes),**hashes}))
    elif args.command == "synthetic-q":
        from .qexperiment import experiment
        print(json.dumps(experiment(args.output,args.seeds,args.episodes,args.updates,args.anchors,args.repeats)["summary"],indent=2))

    elif args.command == "smoke-samples":
        from .sample_smoke import run
        report=run(args.samples,args.output,args.updates,args.device)
        print(json.dumps({k:report[k] for k in ("status","scope","raw_samples","transitions","score_rows","device")},indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path


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
    elif args.command == "train":
        import torch
        from .schema import Episode, file_hash
        from .value import fit_value
        root = Path(args.dataset)
        manifest = json.loads((root/"manifest.json").read_text())
        selected = set(manifest["split"]["train"])
        episodes = []
        for row in manifest["episodes"]:
            if row["episode_id"] in selected:
                path = (root/row["path"]).resolve()
                if not path.is_relative_to(root.resolve()) or file_hash(path) != row["sha256"]:
                    raise ValueError("dataset path or hash mismatch")
                episodes.append(Episode.load(path))
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


if __name__ == "__main__":
    main()

"""Real fixture wiring check, explicitly NOT train/test generalization evaluation."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .schema import Episode,file_hash
from .samp830 import convert_task
from .targets import discounted_returns,n_step_scores,td_scores,n_step_targets
from .value import fit_value,ValuePredictor
from .qvalue import fit_q,QPredictor,qv_scores
from .cli import read_split


def run(samples,output,updates=200,device="cpu"):
    import torch
    if updates<1:raise ValueError("updates must be positive")
    samples=Path(samples).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("choose a new empty output directory")
    records=[json.loads(line) for line in (samples/"manifest.jsonl").read_text().splitlines() if line.strip()]
    if not records:raise ValueError("no sample records")
    if len({r["episode_index"] for r in records})!=len(records):raise ValueError("duplicate sample episode")
    output.mkdir(parents=True,exist_ok=True)
    source=output/"staged_source";data=source/"data/chunk-000";data.mkdir(parents=True)
    audits=[]
    for record in records:
        # local_path is repository-relative; only basenames within samples are accepted.
        path=(samples/Path(record["local_path"]).name).resolve()
        if not path.is_relative_to(samples):raise ValueError("sample path escapes samples directory")
        if file_hash(path)!=record["sha256"]:raise ValueError("sample SHA256 mismatch")
        table=pq.read_table(path,columns=["state","actions","timestamp","frame_index","task_index"])
        rows=table.to_pydict();state=np.asarray(rows["state"]);action=np.asarray(rows["actions"])
        if state.shape!=(record["length"],8) or action.shape!=(record["length"],7):raise ValueError("unexpected real sample dimensions")
        if not np.isfinite(state).all() or not np.isfinite(action).all():raise ValueError("nonfinite sample")
        if np.max(np.abs(action))>1.00001:raise ValueError("sample actions outside declared native range")
        dt=np.diff(np.asarray(rows["timestamp"],float))
        if not np.allclose(dt,.1,atol=1e-5):raise ValueError("unexpected sample row timing")
        audits.append(dict(file=path.name,sha256=record["sha256"],transitions=len(state),
            state_dim=8,action_dim=7,success=record["success"],task_index=record["task_index"],
            action_min=float(action.min()),action_max=float(action.max()),row_interval_seconds=float(np.median(dt))))
        (data/f"episode_{record['episode_index']:06d}.parquet").symlink_to(path)
    (source/"episode_manifest.jsonl").write_text("\n".join(json.dumps(r) for r in records)+"\n")
    torch.set_num_threads(1)
    if device.startswith("cuda") and not torch.cuda.is_available():raise ValueError("requested CUDA is unavailable")
    tasks=[];scores=[]
    for task_index in sorted({int(r["task_index"]) for r in records}):
        task_dir=output/f"task_{task_index}";task_dir.mkdir()
        canonical=task_dir/"canonical"
        conversion=convert_task(source,task_index,canonical,task_dir/"conversion_work",seed=0)
        manifest=json.loads((canonical/"manifest.json").read_text())
        episodes=[Episode.load(canonical/r["path"]) for r in manifest["episodes"]]
        # Deliberate in-sample optimizer check. Never relabel test data as a formal training split.
        normal_training_available=True
        try:read_split(canonical,"train")
        except ValueError as exc:
            if str(exc)!="selected split is empty":raise
            normal_training_available=False
        vc=fit_value(episodes,method="mc",updates=updates,seed=0,device=device)
        qc=fit_q(episodes,updates=updates,seed=0,device=device)
        vpath,qpath=task_dir/"shared_v.pt",task_dir/"mc_q.pt"
        torch.save(vc,vpath);torch.save(qc,qpath)
        vp,qp=ValuePredictor(vpath,device),QPredictor(qpath,device)
        vh,qh=file_hash(vpath),file_hash(qpath)
        verror=[];qerror=[]
        for ep in episodes:
            actual=discounted_returns(ep)
            v,nv=vp.predict(ep),vp.predict(ep,True)
            q=qp.predict(ep)
            verror.extend((v-actual)**2);qerror.extend((q-actual)**2)
            np.testing.assert_allclose(n_step_scores(ep,v,nv,1),td_scores(ep,v,nv),atol=1e-12)
            np.testing.assert_allclose(n_step_targets(ep,nv,len(ep)+1),actual,atol=1e-12)
            methods={f"td_n{n}":n_step_scores(ep,v,nv,n) for n in (1,5,10,20)}
            methods["mc_q_minus_shared_v"]=qv_scores(qp,vp,ep)
            for method,values in methods.items():
                if not np.isfinite(values).all():raise ValueError("nonfinite scores")
                for frame,value in enumerate(values):
                    scores.append(dict(episode_id=ep.episode_id,task_id=ep.task_id,t=frame,
                        method=method,score=float(value),v_checkpoint_sha256=vh,q_checkpoint_sha256=qh,
                        scope="same_episode_smoke_only"))
        if file_hash(vpath)!=vh or file_hash(qpath)!=qh:raise ValueError("scoring mutated checkpoint")
        tasks.append(dict(task_index=task_index,conversion=conversion,
            normal_train_split_available=normal_training_available,
            scope="same_episode_smoke_only",v_loss_first20=float(np.mean(vc["losses"][:20])),
            v_loss_last20=float(np.mean(vc["losses"][-20:])),q_loss_first20=float(np.mean(qc["losses"][:20])),
            q_loss_last20=float(np.mean(qc["losses"][-20:])),
            in_sample_v_mc_rmse=float(np.sqrt(np.mean(verror))),in_sample_q_mc_rmse=float(np.sqrt(np.mean(qerror))),
            v_checkpoint_sha256=vh,q_checkpoint_sha256=qh))
    with (output/"scores.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
    report=dict(status="passed",scope="same_episode_smoke_only",device=device,
        torch_version=torch.__version__,numpy_version=np.__version__,python_version=sys.version.split()[0],
        updates_per_model=updates,raw_samples=len(records),transitions=sum(r["transitions"] for r in audits),
        score_rows=len(scores),audits=audits,tasks=tasks,
        limitations=["One episode per task: no valid held-out learning/generalization evaluation.",
            "Rewards reconstructed from manifest: final-success reward only; terminal next_obs is a masked placeholder.",
            "Source samp830/libero_40_suboptimal_v1, not the RLinf RECAP release.",
            "Policy identity and primitive-step/action alignment follow source metadata; no simulator replay validation.",
            "State8 omits images/object history; this smoke test makes no full-Markov or true-value accuracy claim."])
    (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

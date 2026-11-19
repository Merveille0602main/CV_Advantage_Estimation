"""Extract the pinned low-dimensional demonstration rows and freeze the protocol before evaluation."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from study import REV, ROOT, TASKS, WEIGHTS, read, task_dir, write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(data, weights):
    assert REV in read(weights/"train_config.json")["dataset"]["root"]
    lengths = pq.read_table(data/"meta/episodes/chunk-000/file-000.parquet", columns=["episode_index","length"]).to_pandas()
    sizes = dict(zip(lengths.episode_index, lengths.length))
    tasks = pq.read_table(data/"meta/tasks.parquet").to_pandas()
    records = {task:[] for task in TASKS}
    sources = {}
    for path in sorted((data/"data").rglob("*.parquet")):
        present = set(pq.read_table(path, columns=["task_index"])[0].to_pylist()) & TASKS.keys()
        if not present:
            continue
        rel = path.relative_to(data).as_posix()
        local_meta = data/".cache/huggingface/download"/(rel+".metadata")
        if local_meta.exists():
            assert local_meta.read_text().splitlines()[0] == REV
        sources[rel] = sha(path)
        table = pq.read_table(path).to_pandas()
        for task in present:
            for idx, group in table[table.task_index==task].groupby("episode_index"):
                records[task].append((int(idx),group,rel))
    write(ROOT/"demo_sources.json", dict(dataset="lerobot/libero", revision=REV, sha256=sources))
    for task, parts in records.items():
        language = TASKS[task][2]
        assert int(tasks.loc[language,"task_index"]) == task
        # Episodes can cross a parquet shard boundary. Preserve original episode/frame IDs.
        import pandas as pd
        rows = pd.concat([g for _,g,_ in parts]).sort_values(["episode_index","frame_index"])
        ids = sorted(set(int(i) for i in rows.episode_index))
        assert len(ids) == {13:40,19:43,25:45}[task]
        for idx,g in rows.groupby("episode_index"):
            np.testing.assert_array_equal(g.frame_index,np.arange(sizes[idx]))
        rng = np.random.default_rng(1200+task)
        donors = rng.permutation(ids).tolist()
        ntrain = int(.8*len(ids))
        train, test = sorted(donors[:ntrain]), sorted(donors[ntrain:])
        state = np.stack(rows["observation.state"]).astype(np.float32)
        action = np.stack(rows.action).astype(np.float32)
        assert np.isfinite(state).all() and np.isfinite(action).all() and np.abs(action).max()<=1.00001
        train_mask = np.isin(rows.episode_index,train)
        adjacent = (np.diff(rows.episode_index)==0)&train_mask[:-1]
        velocity = np.diff(state[:,:3],axis=0)[adjacent]
        delta = np.diff(np.abs(state[:,6]),axis=0)[adjacent]
        corrs = [float(np.corrcoef(action[:-1,k][adjacent],velocity[:,k])[0,1]) for k in range(3)]
        grip = action[:-1,6][adjacent]
        grip_corr = float(np.corrcoef(grip,delta)[0,1]) if np.ptp(grip)>0 and np.ptp(delta)>0 else None
        assert min(corrs)>0 and (grip_corr is None or grip_corr<0)
        median = float(np.median([sizes[i] for i in train]))
        frames = [int(np.floor(median*f)) for f in (.25,.5,.75)]
        out = task_dir(task)
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/"demo_library.npz",state=state,action=action,episode=rows.episode_index.to_numpy(),
                            frame=rows.frame_index.to_numpy(),train=train_mask,
                            scale=np.maximum(state[train_mask].std(axis=0),1e-6))
        write(out/"demo.json",dict(task=task,language=language,train_ids=train,test_ids=test,frames=frames,
              median_training_length=median,episodes=len(ids),steps=len(rows),library_sha256=sha(out/"demo_library.npz"),
              end_effector_delta_correlation=corrs,gripper_opening_correlation=grip_corr,
              files={str(idx):sorted({s for i,_,s in parts if i==idx}) for idx in ids}))
        print(task,len(ids),len(rows),frames,corrs,grip_corr)
    write(ROOT/"protocol.json",dict(tasks=list(TASKS),model_version="38e7414",eval_code="64c3cc8",
          gamma=.99,seeds=[0,1,2],split_seed=12,training_families=35,validation_families=5,test_families=10,
          anchors="rep0, floor of 25%, 50%, 75% of median duration of train-donor demos; omit if episode ended",
          candidates=["recorded","resampled","nearest demonstration","half translation","reverse translation and flip gripper"],
          distance="Euclidean distance in robot state, each coordinate divided by training-demo standard deviation, ties by episode/frame",
          augmentation="6000 suffix control steps per arm/task in four shards of 1500; same scheduled train anchors and continuation seed per slot; exclude censored last suffixes from MC training. Prefix reconstruction is overhead.",
          sampling="base episodes plus every transition of complete added suffixes, uniform transition sampling; unchanged upstream MC fitter",
          reference="8 paired full continuations; two other paired seeds for score trajectories capped at 20",
          distinguishable="95% paired bootstrap beyond +/-0.005 and >=6 of 8 paired differences in winning direction; ties in prediction count half",
          aggregation="mean over pairs within anchor, anchors within initial state, then initial states; seed RMSE averaged after square root",
          primary="RMSE of each of four alternative-minus-recorded action gaps, against paired full-simulation mean; macro across tasks",
          diagnostics="signed gap error and absolute Q bias by action and phase; three training seeds reported; family-bootstrap paired difference intervals",
          restore="verify full original suffix, all 7D actions, 8D states, rewards and length, before branching"))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("data",type=Path)
    p.add_argument("--weights",type=Path,default=WEIGHTS)
    a=p.parse_args()
    main(a.data,a.weights)

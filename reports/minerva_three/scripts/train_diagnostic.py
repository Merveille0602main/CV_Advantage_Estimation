"""Measure frozen baseline action-gap errors on matched fresh training-family continuations."""
import numpy as np
from advantage.qvalue import QPredictor
from advantage.schema import Episode
from analysis import model_dir, save_csv
from study import NAMES, TASKS, base_dir, read, task_dir, write


def run(task):
    root=task_dir(task)
    q=[QPredictor(model_dir(task,"base")/f"Q_seed{seed}.pt") for seed in range(3)]
    rows=[]
    for path in sorted((root/"augmentation"/"branch").glob("shard*/slot*.npz")):
        control_path=root/"augmentation"/"plain"/path.parent.name/path.name
        if not control_path.exists():
            continue
        do,plain=Episode.load(path),Episode.load(control_path)
        for k in ("init_id","frame","slot","shard","continuation_seed"):
            assert do.metadata[k]==plain.metadata[k]
        if not (do.terminated[-1] and plain.terminated[-1]):
            continue
        np.testing.assert_array_equal(do.obs[0],plain.obs[0])
        measured=float(do.reward[-1])*.99**(len(do)-1)-float(plain.reward[-1])*.99**(len(plain)-1)
        for seed,predictor in enumerate(q):
            predicted=float(predictor.predict(do)[0])-float(predictor.predict(plain)[0])
            rows.append(dict(init=do.metadata["init_id"],frame=do.metadata["frame"],seed=seed,
                      continuation_seed=do.metadata["continuation_seed"],shard=do.metadata["shard"],slot=do.metadata["slot"],
                      candidate=do.metadata["candidate"],reference_gap=measured,predicted_gap=predicted,error=predicted-measured))
    summary={}
    for name in NAMES:
        selected=[r for r in rows if r["candidate"]==name]
        assert selected
        summary[name]=dict(paired_rollouts=len(selected)//3,families=len({r["init"] for r in selected}),
                  mean_reference_gap=float(np.mean([r["reference_gap"] for r in selected])),
                  mean_predicted_gap=float(np.mean([r["predicted_gap"] for r in selected])),
                  pooled_error_rmse=float(np.sqrt(np.mean([r["error"]**2 for r in selected]))))
    save_csv(root/"train_diagnostic.csv",rows)
    write(root/"train_diagnostic.json",dict(task=task,paired_slots=len(rows)//3,by_candidate=summary,
          note="Exploratory training-family errors. Match both suffixes by start and policy noise. One pair of realized returns per slot. Does not enter test labels or training targets."))
    print(task,len(rows)//3,flush=True)


def drawer_input_probe():
    """Separate what drives the network prediction from what changes simulator return."""
    task=19
    root=task_dir(task)
    train=read(root/"dataset_manifest.json")["episodes"]
    gripper=np.concatenate([Episode.load(base_dir(task)/"episodes"/(r["episode"]+".npz")).action[:,6]
                            for r in train if r["split"]=="train"])
    models={(arm,seed):QPredictor(model_dir(task,arm)/f"Q_seed{seed}.pt")
            for arm in ("base","plain","branch") for seed in range(3)}
    rows=[]
    for path in sorted((root/"anchors").glob("*.json")):
        a=read(path)
        if a["split"]!="train":
            continue
        ep=Episode.load(base_dir(task)/"episodes"/(a["episode"]+".npz"))
        f=a["frame"]
        original=ep.action[f]
        both=np.asarray(a["candidates"]["strong"],dtype=original.dtype)
        reverse=original.copy()
        reverse[:3]=both[:3]
        flip=original.copy()
        flip[6]=both[6]
        one=Episode(ep.episode_id,ep.parent_id,ep.task_id,ep.policy_id,ep.obs[f:f+1],ep.next_obs[f:f+1],
                    ep.action[f:f+1],ep.reward[f:f+1],ep.terminated[f:f+1],~ep.terminated[f:f+1],ep.remaining_steps[f:f+1])
        for arm in ("base","plain","branch"):
            for seed in range(3):
                model=models[arm,seed]
                initial=float(model.predict(one)[0])
                for name,action in (("translation_only",reverse),("gripper_only",flip),("both",both)):
                    gap=float(model.predict(one,actions=action[None])[0])-initial
                    rows.append(dict(init=a["init"],frame=f,arm=arm,seed=seed,component=name,predicted_gap=gap))
    save_csv(root/"drawer_model_probe.csv",rows)
    means={}
    for arm in ("base","plain","branch"):
        means[arm]={k:float(np.mean([r["predicted_gap"] for r in rows if r["arm"]==arm and r["component"]==k]))
                    for k in ("translation_only","gripper_only","both")}
    write(root/"drawer_model_probe.json",dict(training_steps=len(gripper),gripper_min=float(gripper.min()),
          gripper_max=float(gripper.max()),gripper_sd=float(gripper.std()),positive_commands=int((gripper>0).sum()),
          training_anchors=len(rows)//27,mean_predicted_gap=means,
          interpretation="Model-input sensitivity at training anchors. No simulator reference is assigned to isolated components."))


if __name__=="__main__":
    for t in TASKS:
        run(t)
    drawer_input_probe()

"""Train fixed-budget predictors and evaluate paired returns on held-out families."""
import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from advantage.qvalue import QPredictor, fit_q
from advantage.schema import Episode, file_hash
from advantage.targets import discounted_returns, n_step_scores
from advantage.value import ValuePredictor, fit_value
from study import NAMES, OLD, POLICY, ROOT, TASKS, WEIGHTS, base_dir, eid, read, split, task_dir, write

ARMS=("base","plain","branch")


def save_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".partial")
    with temp.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)


def model_dir(task,arm):
    return base_dir(task)/"models" if arm=="base" else task_dir(task)/"models"/arm


def validate_base(task):
    horizon=TASKS[task][3]
    root=base_dir(task)
    data,records,stats={},[],{}
    hashes=set()
    for part,ids in split().items():
        episodes=[]
        for init in ids:
            siblings=[]
            for rep in range(2):
                path=root/"episodes"/(eid(init,rep)+".npz")
                ep=Episode.load(path)
                assert ep.task_id=="%s_%d"%TASKS[task][:2] and ep.policy_id==POLICY
                assert ep.metadata["split"]==part and ep.metadata["init_id"]==init and ep.episode_id==eid(init,rep)
                assert ep.parent_id==f"init{init:02}" and ep.remaining_steps[0]==horizon
                assert ep.terminated[-1] and (ep.reward[-1]==1 or (ep.reward[-1]==0 and len(ep)==horizon))
                assert np.all(ep.reward[:-1]==0) and ep.obs.shape==(len(ep),8) and ep.action.shape==(len(ep),7)
                records.append(dict(episode=ep.episode_id,split=part,init=init,steps=len(ep),success=int(ep.reward[-1]),
                                    initial_sha256=ep.metadata["init_sha256"],sha256=file_hash(path)))
                siblings.append(ep)
            np.testing.assert_allclose(siblings[0].obs[0],siblings[1].obs[0],atol=1e-6)
            h=siblings[0].metadata["init_sha256"]
            assert h==siblings[1].metadata["init_sha256"] and h not in hashes
            hashes.add(h)
            episodes.extend(siblings)
        data[part]=episodes
        stats[part]=dict(episodes=len(episodes),families=len(ids),steps=sum(map(len,episodes)),success=int(sum(ep.reward[-1] for ep in episodes)))
    assert len(hashes)==50 and len(list((root/"episodes").glob("*.npz")))==100
    if task==13:
        old=read(OLD/"manifest.json")
        assert old["families"]==split()
        assert {r["episode_id"]:r["sha256"] for r in old["episodes"]}=={r["episode"]:r["sha256"] for r in records}
    train=data["train"]
    commands=np.concatenate([e.action for e in train])
    delta=np.concatenate([e.next_obs-e.obs for e in train])
    jaw=np.concatenate([np.abs(e.next_obs[:,6])-np.abs(e.obs[:,6]) for e in train])
    corrs=[float(np.corrcoef(commands[:,k],delta[:,k])[0,1]) for k in range(3)]
    jaw_corr=float(np.corrcoef(commands[:,6],jaw)[0,1]) if commands[:,6].std()>1e-8 and jaw.std()>1e-8 else None
    write(task_dir(task)/"dataset_manifest.json",dict(task=task,horizon=horizon,policy=POLICY,split=stats,
          episodes=records,families=split(),eef_delta_corr=corrs,gripper_opening_corr=jaw_corr))
    return data


def load_augmentation(task,arm):
    complete=[]
    spent=discarded=slots=0
    choices={}
    used=set()
    for shard in range(4):
        paths=sorted((task_dir(task)/"augmentation"/arm/f"shard{shard}").glob("slot*.npz"))
        shard_steps=0
        for i,path in enumerate(paths):
            ep=Episode.load(path)
            m=ep.metadata
            assert m["slot"]==i and m["arm"]==arm and m["split"]=="train" and m["shard"]==shard
            assert m["init_id"] in split()["train"]
            assert ep.remaining_steps[0]==TASKS[task][3]-m["frame"]
            used.add(m["init_id"])
            shard_steps+=len(ep)
            if ep.terminated[-1]:
                assert ep.reward[-1]==1 or ep.remaining_steps[-1]==1
                complete.append(ep)
                choice=m["candidate"] or "natural"
                choices[choice]=choices.get(choice,0)+1
            else:
                assert i==len(paths)-1 and ep.truncated[-1]
                discarded+=len(ep)
        assert shard_steps==1500
        slots+=len(paths)
        spent+=shard_steps
    assert spent==6000 and sum(map(len,complete))==spent-discarded
    return complete,dict(spent=spent,usable=spent-discarded,censored_steps=discarded,attempts=slots,
                         complete=len(complete),families=sorted(used),first_actions=choices)


def train(task):
    data=validate_base(task)
    device="cuda" if torch.cuda.is_available() else "cpu"
    val=data["validation"]
    targets=np.concatenate([discounted_returns(e,.99) for e in val])
    train_mean=float(np.concatenate([discounted_returns(e,.99) for e in data["train"]]).mean())
    report=dict(device=device,seed=[0,1,2],checkpoints={},budgets={},validation_constant_rmse=float(np.sqrt(np.mean((train_mean-targets)**2))))
    old=read(OLD/"result.json")
    assert {p.name:file_hash(p) for p in WEIGHTS.iterdir() if p.is_file()}==read(OLD/"provenance.json")["weight_files"]
    for arm in ARMS:
        episodes=list(data["train"])
        if arm!="base":
            extra,budget=load_augmentation(task,arm)
            episodes+=extra
            report["budgets"][arm]=budget
        for kind in (("V","Q") if arm=="base" else ("Q",)):
            predictor=ValuePredictor if kind=="V" else QPredictor
            for seed in range(3):
                path=model_dir(task,arm)/f"{kind}_seed{seed}.pt"
                if task==13 and arm=="base":
                    assert file_hash(path)==old["checkpoints"][path.stem]["sha256"]
                elif not path.exists():
                    path.parent.mkdir(parents=True,exist_ok=True)
                    fit=fit_value if kind=="V" else fit_q
                    ck=fit(episodes,seed=seed,device=device,updates=1000,gamma=.99)
                    temp=path.with_suffix(".partial")
                    torch.save(ck,temp)
                    temp.replace(path)
                model=predictor(path)
                ck=model.checkpoint
                assert ck["seed"]==seed and ck["updates"]==1000 and ck["gamma"]==.99 and np.isfinite(ck["losses"]).all()
                pred=np.concatenate([model.predict(ep) for ep in val])
                rmse=float(np.sqrt(np.mean((pred-targets)**2)))
                report["checkpoints"][f"{arm}/{path.stem}"]=dict(sha256=file_hash(path),validation_rmse=rmse)
                print(task,arm,kind,seed,"validation RMSE",rmse,flush=True)
    assert report["budgets"]["plain"]["families"]==report["budgets"]["branch"]["families"]
    write(task_dir(task)/"training.json",report)


def score_episode(r,a,name,source):
    states=np.array(r["states"],np.float32)
    commands=np.array(r["actions"],np.float32)
    rewards=np.array(r["rewards"],np.float32)
    t=len(commands)
    assert t==r["steps"] and 1<=t<=20 and (r["terminal"] or t==20)
    np.testing.assert_array_equal(commands[0],a["candidates"][name])
    np.testing.assert_array_equal(states[0],a["state"])
    terminal=np.zeros(t,bool)
    terminal[-1]=r["terminal"]
    truncated=np.zeros(t,bool)
    truncated[-1]=not r["terminal"]
    return Episode(source.episode_id,source.parent_id,source.task_id,source.policy_id,states[:-1],states[1:],commands,
            rewards,terminal,truncated,source.remaining_steps[a["frame"]]-np.arange(t))


def predictions(task):
    out=task_dir(task)
    training=read(out/"training.json")
    estimators={}
    for arm in ARMS:
        for kind in (("V","Q") if arm=="base" else ("Q",)):
            for seed in range(3):
                p=model_dir(task,arm)/f"{kind}_seed{seed}.pt"
                assert file_hash(p)==training["checkpoints"][f"{arm}/{p.stem}"]["sha256"]
                estimators[arm,kind,seed]=(QPredictor if kind=="Q" else ValuePredictor)(p)
    cells, pairs=[],[]
    starts=read(out/"demo.json")["frames"]
    phases=dict(zip(starts,["early","middle","late"]))
    streams=set()
    replayed=[]
    for init in split()["test"]:
        source=Episode.load(base_dir(task)/"episodes"/(eid(init)+".npz"))
        for f in starts:
            if f>=len(source):
                continue
            key=f"{eid(init)}_t{f:03}"
            a=read(out/"anchors"/(key+".json"))
            assert a["split"]=="test" and a["frame"]==f and a["init"]==init and a["replayed"]==len(source)-f
            assert a["donor"]["episode"] in read(out/"demo.json")["test_ids"]
            replayed.append(a)
            ref=[]
            short={name:[] for name in NAMES}
            for rep in range(10):
                group="ref" if rep<8 else "score"
                row=read(out/"evaluation"/f"{key}_{group}{rep:02}.json")
                assert row["seed"]==(20000000 if rep<8 else 30000000)+task*100000+init*2000+f*10+rep
                assert row["group"]==group and (row["init"],row["frame"])==(init,f) and row["seed"] not in streams
                streams.add(row["seed"])
                refs=[]
                for name in NAMES:
                    r=row["rows"][name]
                    if rep<8:
                        assert r["terminal"] and (r["success"] or r["steps"]==TASKS[task][3]-f)
                        refs.append(.99**(r["steps"]-1) if r["success"] else 0.)
                    else:
                        short[name].append(score_episode(r,a,name,source))
                if rep<8:
                    ref.append(refs)
            ref=np.array(ref)
            means=ref.mean(axis=0)
            scores={}
            for idx,name in enumerate(NAMES):
                # Evaluate Q and the common current-state V with the same single-row shape.
                ep=short[name][0]
                one=Episode(ep.episode_id,ep.parent_id,ep.task_id,ep.policy_id,ep.obs[:1],ep.next_obs[:1],ep.action[:1],
                            ep.reward[:1],ep.terminated[:1],~ep.terminated[:1],ep.remaining_steps[:1])
                for seed in range(3):
                    v=estimators["base","V",seed]
                    value=float(v.predict(one)[0])
                    pred={arm:float(estimators[arm,"Q",seed].predict(one)[0]) for arm in ARMS}
                    methods={arm+"_QV":q-value for arm,q in pred.items()}
                    rolls=[]
                    for branch in short[name]:
                        vs=v.predict(branch)
                        vs[0]=value
                        nv=v.predict(branch,next_obs=True)
                        rolls.append([n_step_scores(branch,vs,nv,n,.99)[0] for n in (1,5,10,20)])
                    for n,s in zip((1,5,10,20),np.mean(rolls,axis=0)):
                        methods[f"N{n}"]=float(s)
                    scores[idx,seed]=methods
                    cells.append(dict(task=task,init=init,frame=f,phase=phases[f],candidate=name,seed=seed,
                             reference=float(means[idx]),reference_se=float(ref[:,idx].std(ddof=1)/np.sqrt(8)),
                             V=value,**{arm+"_Q":q for arm,q in pred.items()},**methods))
            rng=np.random.default_rng(450000+task*10000+init*200+f)
            draws=rng.integers(8,size=(10000,8))
            for j,k in itertools.combinations(range(5),2):
                gaps=ref[:,k]-ref[:,j]
                lo,hi=np.quantile(gaps[draws].mean(axis=1),[.025,.975])
                win=1 if lo>.005 and sum(gaps>0)>=6 else -1 if hi<-.005 and sum(gaps<0)>=6 else 0
                for seed in range(3):
                    diffs={m:s-scores[j,seed][m] for m,s in scores[k,seed].items()}
                    pairs.append(dict(task=task,init=init,frame=f,phase=phases[f],left=NAMES[j],right=NAMES[k],seed=seed,
                                      reference_gap=float(gaps.mean()),lo=float(lo),hi=float(hi),winner=win,**diffs))
    assert len(streams)==len(replayed)*10
    save_csv(out/"predictions.csv",cells)
    save_csv(out/"pairs.csv",pairs)
    return cells,pairs,dict(anchors=len(replayed),families=len({a["init"] for a in replayed}),planned_anchors=30,
            full_branches=len(replayed)*40,score_branches=len(replayed)*10,replayed_suffix_steps=sum(a["replayed"] for a in replayed),
            max_state_diff=max(a["max_state_error"] for a in replayed),max_action_diff=max(a["max_action_error"] for a in replayed))


def aggregate(task,cells,pairs,counts):
    def family_mean(rows,measure):
        frames=defaultdict(list)
        for row in rows:
            frames[row["seed"],row["init"],row["frame"]].append(measure(row))
        families=defaultdict(list)
        for (seed,init,frame),values in frames.items():
            families[seed,init].append(np.mean(values))
        ids=sorted({init for _,init in families})
        return np.array([[np.mean(families[seed,init]) for init in ids] for seed in range(3)])
    def group(rows,key):
        buckets=defaultdict(list)
        for row in rows:
            buckets[row[key]].append(row)
        return buckets
    def rmse(rows,error):
        return float(np.sqrt(family_mean(rows,lambda r:error(r)**2).mean(axis=1)).mean())
    g=[r for r in pairs if r["left"]=="recorded"]
    eligible=[r for r in pairs if r["winner"]]
    counts.update(all_pairs=len(pairs)//3,eligible_pairs=len(eligible)//3,
                  eligible_families=len({r["init"] for r in eligible}),
                  eligible_snapshots=len({(r["init"],r["frame"]) for r in eligible}))
    methods=[arm+"_QV" for arm in ARMS]+["N1","N5","N10","N20"]
    ranking={}
    for m in methods:
        def correct(r):
            return .5 if r[m]==0 else float(np.sign(r[m])==r["winner"])
        seed_values=family_mean(eligible,correct).mean(axis=1) if eligible else []
        ranking[m]=dict(accuracy=float(np.mean(seed_values)) if len(seed_values) else None,by_seed=list(seed_values))
    errors={}
    family_sq={}
    for arm in ARMS:
        def gap_error(r):
            return r[arm+"_QV"]-r["reference_gap"]
        def q_error(r):
            return r[arm+"_Q"]-r["reference"]
        fm=family_mean(g,lambda r:gap_error(r)**2)
        family_sq[arm]=fm
        byseed=np.sqrt(fm.mean(axis=1))
        errors[arm]=dict(gap_rmse=float(byseed.mean()),gap_rmse_seeds=byseed.tolist(),
                q_rmse=rmse(cells,q_error),
                q_bias={name:float(family_mean(part,q_error).mean()) for name,part in group(cells,"candidate").items()},
                gap_bias={name:float(family_mean(part,gap_error).mean()) for name,part in group(g,"right").items()},
                gap_rmse_by_action={name:rmse(part,gap_error) for name,part in group(g,"right").items()},
                gap_rmse_by_phase={name:rmse(part,gap_error) for name,part in group(g,"phase").items()})
    errors["zero_gap"]=dict(gap_rmse=rmse(g,lambda r:r["reference_gap"]))
    rng=np.random.default_rng(281120+task)
    size=family_sq["base"].shape[1]
    samples=rng.integers(size,size=(10000,size))
    boot={arm:np.sqrt(x[:,samples].mean(axis=2)).mean(axis=0) for arm,x in family_sq.items()}
    contrasts={}
    for left,right in (("branch","plain"),("branch","base"),("plain","base")):
        contrasts[left+"-"+right]=dict(difference=errors[left]["gap_rmse"]-errors[right]["gap_rmse"],
                      family_bootstrap_ci95=np.quantile(boot[left]-boot[right],[.025,.975]).tolist())
    coverage={}
    for left,right in itertools.combinations(NAMES,2):
        subset=[r for r in pairs if r["seed"]==0 and r["left"]==left and r["right"]==right]
        sel=[r for r in subset if r["winner"]]
        coverage[left+"-"+right]=dict(total=len(subset),eligible=len(sel),families=len({r["init"] for r in sel}))
    return dict(task=task,counts=counts,errors=errors,contrasts=contrasts,ranking=ranking,coverage=coverage)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("train","metrics"))
    mode=p.parse_args().mode
    summaries=[]
    for task in TASKS:
        if mode=="train":
            train(task)
        else:
            result=aggregate(task,*predictions(task))
            write(task_dir(task)/"metrics.json",result)
            summaries.append(result)
            print(task,result["counts"],result["contrasts"],flush=True)
    if summaries:
        write(ROOT/"summary.json",dict(tasks=summaries,macro_gap_rmse={arm:float(np.mean([s["errors"][arm]["gap_rmse"] for s in summaries])) for arm in (*ARMS,"zero_gap")}))


if __name__=="__main__":
    main()

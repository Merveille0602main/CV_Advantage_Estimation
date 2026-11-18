"""Frozen MC V, learned MC Q, and independent same-state oracle evaluation."""
import csv
import json
from pathlib import Path
import numpy as np
from .schema import Episode,split_families
from .synthetic import generate,exact_values,outcomes,HORIZON,POLICY_ADVANCE
from .value import fit_value,ValuePredictor
from .qvalue import fit_q,QPredictor,qv_scores
from .targets import n_step_scores
from .evaluation import ranking_metrics


def branch(state,remaining,action,seed):
    rng=np.random.default_rng(seed)
    obs=[];nxt=[];actions=[];rewards=[]
    for i in range(remaining):
        a=action if i==0 else int(rng.random()<POLICY_ADVANCE)
        options=outcomes(state,a)
        _,new,r,done=options[rng.choice(3,p=[o[0] for o in options])]
        obs.append([state]);nxt.append([new]);actions.append([a]);rewards.append(r)
        state=new
        if done:break
    term=np.zeros(len(rewards),bool);term[-1]=True
    return Episode(f"branch-{seed}-{action}",f"anchor-{seed}","progress-v1","fixed-random-policy-v1",
        np.array(obs),np.array(nxt),np.array(actions),np.array(rewards),term,np.zeros(len(rewards),bool),
        remaining-np.arange(len(rewards)),{"synthetic":True,"forced_first_action":True})


def experiment(output,seeds=(0,1,2),episodes=450,updates=1000,anchors=40,repeats=16):
    import torch
    if episodes<30 or updates<1 or anchors<1 or repeats<1 or not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("positive budgets, at least 30 episodes, unique seeds required")
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError("use a new empty output directory")
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    gamma=.99
    # Oracle arrays are used only after fitting; neither learner accepts oracle input.
    oracle_v,oracle_q=exact_values(gamma)
    rows=[];candidate_rows=[];splits=[]
    for seed in seeds:
        data=generate(episodes,seed)
        split=split_families(data,seed)
        train=[e for e in data if e.episode_id in split["train"]]
        test=[e for e in data if e.episode_id in split["test"]]
        vc=fit_value(train,gamma=gamma,updates=updates,seed=seed,method="mc")
        qc=fit_q(train,gamma=gamma,updates=updates,seed=seed)
        vp_path,qp_path=output/f"v_seed{seed}.pt",output/f"q_seed{seed}.pt"
        torch.save(vc,vp_path);torch.save(qc,qp_path)
        vp,qp=ValuePredictor(vp_path),QPredictor(qp_path)
        verrors=[];qerrors=[]
        for ep in test:
            t=HORIZON-ep.remaining_steps;s=ep.obs[:,0].astype(int);a=ep.action[:,0].astype(int)
            verrors.extend((vp.predict(ep)-oracle_v[t,s])**2)
            qerrors.extend((qp.predict(ep)-oracle_q[t,s,a])**2)
        rng=np.random.default_rng(100000+seed)
        chosen=rng.permutation(len(test))[:anchors]
        methods={"mc_q_minus_v":[] ,**{f"td_n{n}":[] for n in (1,5,10,20)}}
        labels=[];reference_a=[]
        for anchor,episode_index in enumerate(chosen):
            ep=test[int(episode_index)]
            frame=int(rng.integers(len(ep)))
            s=int(ep.obs[frame,0]);remaining=int(ep.remaining_steps[frame]);t=HORIZON-remaining
            true_q=oracle_q[t,s,:].copy();labels.append(true_q)
            reference_a.append(true_q-oracle_v[t,s])
            predictions={key:[] for key in methods}
            for action in (0,1):
                values={n:[] for n in (1,5,10,20)}
                learned=None
                for repetition in range(repeats):
                    # Same random seed across candidate actions; separate from training RNG.
                    branch_seed=int(np.random.SeedSequence([seed,anchor,repetition,9876]).generate_state(1)[0])
                    b=branch(s,remaining,action,branch_seed)
                    v,nv=vp.predict(b),vp.predict(b,True)
                    for n in values:
                        values[n].append(float(n_step_scores(b,v,nv,n,gamma)[0]))
                    if learned is None:learned=float(qv_scores(qp,vp,b)[0])
                predictions["mc_q_minus_v"].append(learned)
                for n in values:predictions[f"td_n{n}"].append(float(np.mean(values[n])))
            for method,scores in predictions.items():
                methods[method].append(scores)
                for a,score in enumerate(scores):
                    candidate_rows.append(dict(seed=seed,anchor_id=f"{ep.episode_id}:{frame}",
                        episode_id=ep.episode_id,t=frame,state=s,remaining_steps=remaining,action=a,
                        method=method,score=score,oracle_q=float(true_q[a]),oracle_advantage=float(true_q[a]-oracle_v[t,s])))
        for method,pred in methods.items():
            result=ranking_metrics(pred,labels,min_gap=.005)
            rows.append(dict(seed=seed,method=method,**result,
                advantage_rmse=float(np.sqrt(np.mean((np.asarray(pred)-reference_a)**2))),
                heldout_v_rmse=float(np.sqrt(np.mean(verrors))),heldout_q_rmse=float(np.sqrt(np.mean(qerrors)))))
        splits.append(dict(seed=seed,split=split))
    summary=[]
    for method in methods:
        subset=[r for r in rows if r["method"]==method]
        valid=[r["ranking_accuracy"] for r in subset if r["ranking_accuracy"] is not None]
        summary.append(dict(method=method,ranking_accuracy_mean=float(np.mean(valid)) if valid else None,
            advantage_rmse_mean=float(np.mean([r["advantage_rmse"] for r in subset])),
            pair_coverage_mean=float(np.mean([r["pair_coverage"] for r in subset]))))
    report=dict(config=dict(seeds=list(seeds),episodes_per_seed=episodes,updates=updates,
        max_test_anchors_per_seed=anchors,score_branches_per_candidate=repeats,gamma=gamma,branch_seed_scheme="SeedSequence(seed,anchor,repeat,9876)"),
        summary=summary,runs=rows,limitations=[
            "Synthetic MDP only; exact DP labels never supervise V or Q.",
            "Learned Q-V is a comparison method, not a ground-truth judge.",
            "TD scores average repeated future branches; Q-V reads no future observations.",
            "One anchor per held-out episode; these are preliminary results without confidence intervals.",
            "Single forced action then fixed reference policy; candidates stay in its action support."])
    (output/"results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (output/"splits.json").write_text(json.dumps(splits,indent=2),encoding="utf-8")
    for name,records in (("metrics.csv",rows),("candidate_scores.csv",candidate_rows)):
        with (output/name).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return report

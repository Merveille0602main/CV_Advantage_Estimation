"""Frozen MC V, learned MC Q, and independent same-state oracle evaluation."""
import json
from pathlib import Path

import numpy as np

from .evaluation import ranking_metrics
from .experiment_io import write_csv
from .schema import split_families
from .synthetic import HORIZON, branch, exact_values, generate
from .synthetic_comparison import (
    choose_anchors, comparison_branch_seed, estimator_specs, fit_comparison_models, score_branches,
)


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
    specs = estimator_specs(value_ns=())
    rows=[];candidate_rows=[];splits=[]
    for seed in seeds:
        data=generate(episodes,seed)
        split=split_families(data,seed)
        train=[e for e in data if e.episode_id in split["train"]]
        test=[e for e in data if e.episode_id in split["test"]]
        vp, qp = fit_comparison_models(train, output, seed, updates, gamma)
        verrors=[];qerrors=[]
        for ep in test:
            t=HORIZON-ep.remaining_steps;s=ep.obs[:,0].astype(int);a=ep.action[:,0].astype(int)
            verrors.extend((vp.predict(ep)-oracle_v[t,s])**2)
            qerrors.extend((qp.predict(ep)-oracle_q[t,s,a])**2)
        chosen = choose_anchors(test, anchors, seed)
        methods = {spec["name"]: [] for spec in specs}
        labels=[];reference_a=[]
        for anchor, (ep, frame) in enumerate(chosen):
            s=int(ep.obs[frame,0]);remaining=int(ep.remaining_steps[frame]);t=HORIZON-remaining
            true_q=oracle_q[t,s,:].copy();labels.append(true_q)
            reference_a.append(true_q-oracle_v[t,s])
            predictions={key:[] for key in methods}
            for action in range(oracle_q.shape[2]):
                branches = (
                    branch(s, remaining, action, comparison_branch_seed(seed, anchor, repetition))
                    for repetition in range(repeats)
                )
                repeated = score_branches(branches, specs, vp, qp, gamma, reuse_qv=True)
                for spec in specs:
                    name = spec["name"]
                    samples = repeated[name]
                    score = samples[0] if spec["family"] == "q_minus_v" else float(np.mean(samples))
                    predictions[name].append(score)
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
        write_csv(output / name, records)
    return report

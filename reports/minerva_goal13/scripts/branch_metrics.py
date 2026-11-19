"""Score frozen checkpoints on independent short branches and compare to full-return labels."""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
from advantage.discrimination import aggregate_metrics, pairwise_records
from advantage.qvalue import QPredictor
from advantage.schema import Episode, file_hash
from advantage.targets import n_step_scores
from advantage.value import ValuePredictor

SOURCE = Path("/mnt/d/CV/training_runs/minerva_goal13/20261119_run01")
METHODS = ["Q-V", "TD1", "N5", "N10", "N20"]
NAMES = ["recorded", "resampled", "perturbed"]
PAIRS = list(itertools.combinations(range(3), 2))
BOOT = 10000


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(temp, path)


def episode(r, anchor, action, source):
    end = np.array(r["next_states"], np.float32)
    states = np.vstack((r["state"],end[:-1])).astype(np.float32)
    steps = len(end)
    assert 1 <= steps <= 20 and len(r["actions"]) == r["steps"] == steps
    assert np.array_equal(r["state"],anchor["state"]) and np.array_equal(r["actions"][0],action)
    terminated = np.zeros(steps, bool)
    truncated = np.zeros(steps, bool)
    terminated[-1] = r["terminal"]
    truncated[-1] = not r["terminal"]
    assert r["terminal"] or steps == 20
    assert sum(r["rewards"]) == r["success"] and (not r["success"] or r["rewards"][-1]==1)
    return Episode(source.episode_id, source.parent_id, source.task_id, source.policy_id, states,
                   end, np.array(r["actions"]), np.array(r["rewards"]), terminated, truncated,
                   300-anchor["frame"]-np.arange(steps))


def interval(a):
    return [float(x) for x in np.quantile(a,[.025,.975])]


def rank_metrics(scores, truth, pairs, ids, family_draws):
    # Input shape [training seed, anchor, candidate]; reference eligibility shared by all methods.
    accuracies, family_scores = [], []
    grouped = {f"{NAMES[j]}-{NAMES[k]}":[] for j,k in PAIRS}
    for seed_scores in scores:
        records = pairwise_records(seed_scores,truth,min_gap=0.0,anchor_ids=ids)
        for rec,label in zip(records,pairs,strict=True):
            assert (rec["anchor_id"],rec["candidate_i"],rec["candidate_j"]) == (
                label["id"],label["i"],label["j"])
            rec["distinguishable"] = label["eligible"]
            if not label["eligible"]:
                rec["correct"] = None
            else:
                grouped[f"{NAMES[label['i']]}-{NAMES[label['j']]}"].append(rec["correct"])
        accuracies.append(aggregate_metrics(records)["anchor_accuracy_mean"])
        for_anchor = []
        for idx in ids:
            correct = [r["correct"] for r in records if r["anchor_id"]==idx and r["distinguishable"]]
            for_anchor.append(np.mean(correct) if correct else np.nan)
        family_scores.append(for_anchor)
    per_family = np.mean(family_scores,axis=0)
    selected = per_family[family_draws]
    counts = np.isfinite(selected).sum(axis=1)
    valid = counts > 0
    ci = interval(np.nansum(selected[valid],axis=1)/counts[valid]) if valid.any() else None
    return dict(accuracy=float(np.nanmean(per_family)) if np.isfinite(per_family).any() else None,
                by_seed=accuracies, family_ci95=ci,
                by_pair_type={k:float(np.mean(v)) if v else None for k,v in grouped.items()})


def q_metrics(pred, truth, reference_draws, family_draws):
    # pred [seed,family,action]; draw noise for all actions with the same indices within family.
    diff = pred - truth
    per_seed = np.sqrt(np.mean(diff**2,axis=(1,2)))
    boot_pred = pred[:,family_draws,:]
    sampled_truth = np.take_along_axis(reference_draws,family_draws[:,:,None],axis=1)
    errors = boot_pred - sampled_truth[None]
    boot_rmse = np.sqrt(np.mean(errors**2,axis=(2,3))).mean(axis=0)
    return dict(rmse=float(per_seed.mean()),by_seed_rmse=per_seed.tolist(),
                mae=float(np.abs(diff).mean()),bias=float(diff.mean()),rmse_ci95=interval(boot_rmse),
                by_candidate_rmse={name:float(np.sqrt(np.mean(diff[:,:,j]**2,axis=1)).mean())
                                   for j,name in enumerate(NAMES)})


def main(out):
    anchors = [read(p) for p in sorted(out.glob("*_anchor.json"))]
    assert len(anchors) == len({a["family"] for a in anchors}) == 10
    ids = [a["episode"] for a in anchors]
    prior = read(SOURCE/"result.json")
    predictors = []
    models = [SOURCE/"models"/f"{kind}_seed{seed}.pt" for seed in range(3) for kind in ("Q","V")]
    for path in models:
        assert file_hash(path)==prior["checkpoints"][path.stem]["sha256"]
    for seed in range(3):
        predictors.append((QPredictor(models[2*seed]),ValuePredictor(models[2*seed+1])))
    refs = np.empty((10,8,3))
    scores = np.empty((5,3,10,3))
    qvals, vvals = np.empty((3,10,3)), np.empty((3,10,3))
    reference_seeds, score_seeds = set(), set()
    for i,a in enumerate(anchors):
        eid=a["episode"]
        source=Episode.load(SOURCE/"episodes"/(eid+".npz"))
        for rep in range(8):
            row=read(out/f"{eid}_ref_{rep:02d}.json")
            assert row["seed"]==100000+100*i+rep and row["group"]=="ref" and row["frame"]==a["frame"]
            reference_seeds.add(row["seed"])
            for k,name in enumerate(NAMES):
                r=row["rows"][name]
                assert r["terminal"] and (r["success"] or r["steps"]==300-a["frame"])
                assert np.isclose(r["ret"],0.99**(r["steps"]-1) if r["success"] else 0)
                refs[i,rep,k]=r["ret"]
        repeat_scores=np.empty((2,5,3,3))
        for j,rep in enumerate((8,9)):
            row=read(out/f"{eid}_score_{rep:02d}.json")
            assert row["seed"]==200000+100*i+rep and row["group"]=="score" and row["frame"]==a["frame"]
            score_seeds.add(row["seed"])
            for k,name in enumerate(NAMES):
                ep=episode(row["rows"][name],a,a["candidates"][name],source)
                first=Episode(ep.episode_id,ep.parent_id,ep.task_id,ep.policy_id,
                              ep.obs[:1],ep.next_obs[:1],ep.action[:1],ep.reward[:1],ep.terminated[:1],
                              ~ep.terminated[:1],ep.remaining_steps[:1])
                for seed,(qp,vp) in enumerate(predictors):
                    v=vp.predict(ep)
                    # An identical current state must have exactly the same V across candidate branches.
                    v[0]=vp.predict(first)[0]
                    q=qp.predict(first)
                    nv=vp.predict(ep,next_obs=True)
                    qvals[seed,i,k]=q[0]
                    vvals[seed,i,k]=v[0]
                    repeat_scores[j,0,seed,k]=q[0]-v[0]
                    for h,n in enumerate((1,5,10,20),start=1):
                        repeat_scores[j,h,seed,k]=n_step_scores(ep,v,nv,n,.99)[0]
        scores[:,:,i,:]=repeat_scores.mean(axis=0)
    assert len(reference_seeds)==80 and len(score_seeds)==20 and not (reference_seeds & score_seeds)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(vvals,vvals[:,:,:1].repeat(3,axis=2),atol=1e-6,rtol=0)
    rng=np.random.default_rng(1901)
    family_draws=rng.integers(10,size=(BOOT,10))
    truth=refs.mean(axis=1)
    ref_se=refs.std(axis=1,ddof=1)/np.sqrt(8)
    reference_draws=np.empty((BOOT,10,3))
    pairs=[]
    cells=[]
    for i,a in enumerate(anchors):
        draws=rng.integers(8,size=(BOOT,8))
        reference_draws[:,i]=refs[i,draws].mean(axis=1)
        for j,k in PAIRS:
            diff=refs[i,:,k]-refs[i,:,j]
            lo,hi=interval(diff[draws].mean(axis=1))
            eligible=bool((lo>.005 and np.sum(diff>0)>=6) or (hi<-.005 and np.sum(diff<0)>=6))
            pairs.append(dict(id=ids[i],i=j,j=k,mean_gap=float(diff.mean()),ci95=[lo,hi],
                              eligible=eligible,paired_gaps=diff.tolist()))
        for k,name in enumerate(NAMES):
            cells.append(dict(id=ids[i],frame=a["frame"],candidate=name,reference=float(truth[i,k]),
                              reference_se=float(ref_se[i,k]),successes=int(np.sum(refs[i,:,k]>0)),
                              returns=refs[i,:,k].tolist(),q_predictions=qvals[:,i,k].tolist(),
                              v_predictions=vvals[:,i,k].tolist(),
                              scores={m:scores[h,:,i,k].tolist() for h,m in enumerate(METHODS)}))
    eligible=[p for p in pairs if p["eligible"]]
    rank={m:rank_metrics(scores[h],truth,pairs,ids,family_draws) for h,m in enumerate(METHODS)}
    q_summary={m:q_metrics(x,truth,reference_draws,family_draws)
               for m,x in (("Q",qvals),("V_only",vvals),("train_mean",np.full((1,10,3),prior["train_return_mean"])))}
    result=dict(anchors=len(anchors),pairs=len(pairs),eligible_pairs=len(eligible),
                eligible_anchors=len({p["id"] for p in eligible}),coverage=len(eligible)/len(pairs),
                pair_type_counts={f"{NAMES[j]}-{NAMES[k]}":sum(p["i"]==j and p["j"]==k for p in eligible)
                                  for j,k in PAIRS},ranking=rank,q_error=q_summary,
                ref_se_median=float(np.median(ref_se)),ref_se_max=float(ref_se.max()),
                replay_steps=sum(a["replay"]["steps"] for a in anchors),
                replay_max_action_diff=max(a["replay"]["max_action_err"] for a in anchors),
                replay_max_state_diff=max(a["replay"]["max_state_err"] for a in anchors),
                protocol_sha256=file_hash(out/"protocol.json"),
                source_manifest_sha256=file_hash(SOURCE/"manifest.json"),
                script_hashes={name:file_hash(Path(__file__).with_name(name)) for name in
                               ("branch_eval.py","branch_metrics.py","branches.sh","collect.py")},
                simulation_files={p.name:file_hash(p) for pat in ("*_anchor.json","*_ref_??.json","*_score_??.json")
                                  for p in sorted(out.glob(pat))},
                model_hashes={p.name:file_hash(p) for p in models})
    save(out/"reference_pairs.json", pairs)
    save(out/"action_values_and_scores.json",cells)
    save(out/"results.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    main(parser.parse_args().output)

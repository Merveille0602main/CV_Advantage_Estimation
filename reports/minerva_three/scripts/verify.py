"""Independent CPU audit using exported predictions and raw termination records."""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def nested_mean(values):
    per_family=defaultdict(list)
    for (init,frame),samples in values.items():
        per_family[init].append(float(np.mean(samples)))
    return float(np.mean([np.mean(v) for v in per_family.values()]))


def check(root):
    summary=read(root/"summary.json")
    proof=[]
    names=("recorded","resampled","demo","mild","strong")
    methods=("base_QV","plain_QV","branch_QV","N1","N5","N10","N20")
    for result in summary["tasks"]:
        task=result["task"]
        out=root/str(task)
        demo=read(out/"demo.json")
        train_donors=set(demo["train_ids"])
        test_donors=set(demo["test_ids"])
        assert not train_donors & test_donors
        manifest=read(out/"dataset_manifest.json")
        assert sum(s["episodes"] for s in manifest["split"].values())==100
        assert sum(len(v) for v in manifest["families"].values())==len(set(sum(manifest["families"].values(),[])))==50
        library=np.load(out/"demo_library.npz",allow_pickle=False)
        with (out/"predictions.csv").open(encoding="utf8",newline="") as f:
            rows=list(csv.DictReader(f))
        lookup={(int(r["init"]),int(r["frame"]),r["candidate"],int(r["seed"])):r for r in rows}
        assert len(rows)==len(lookup)
        all_sq=defaultdict(lambda:defaultdict(list))
        correct=defaultdict(lambda:defaultdict(list))
        evaluated=0
        eligible=0
        families=set()
        for path in sorted((out/"anchors").glob("*.json")):
            a=read(path)
            mask=library["train"] if a["split"]=="train" else ~library["train"]
            subset=np.flatnonzero(mask)
            distances=np.linalg.norm((library["state"][subset]-np.asarray(a["state"],np.float32))/library["scale"],axis=1)
            nearest=subset[np.argmin(distances)]
            d=a["donor"]
            assert (int(library["episode"][nearest]),int(library["frame"][nearest]))==(d["episode"],d["frame"])
            np.testing.assert_allclose(d["distance"],np.min(distances),rtol=1e-5,atol=1e-6)
            np.testing.assert_array_equal(a["candidates"]["demo"],library["action"][nearest])
            if a["split"]!="test":
                continue
            init,frame=a["init"],a["frame"]
            assert init in manifest["families"]["test"]
            evaluated+=1
            returns=np.empty((8,5))
            for repeat in range(8):
                raw=read(out/"evaluation"/f"{path.stem}_ref{repeat:02}.json")
                assert raw["repeat"]==repeat and raw["frame"]==frame and raw["init"]==init
                for j,name in enumerate(names):
                    item=raw["rows"][name]
                    returns[repeat,j]=float(item["success"])*(.99**(item["steps"]-1))
            draw=np.random.default_rng(450000+task*10000+init*200+frame).integers(8,size=(10000,8))
            for j,k in itertools.combinations(range(5),2):
                samples=returns[:,k]-returns[:,j]
                lo,hi=np.percentile(samples[draw].mean(axis=1),[2.5,97.5])
                winner=(int(lo>.005 and np.count_nonzero(samples>0)>=6)-int(hi<-.005 and np.count_nonzero(samples<0)>=6))
                if winner:
                    eligible+=1
                    families.add(init)
                for seed in range(3):
                    left,right=(lookup[init,frame,names[i],seed] for i in (j,k))
                    np.testing.assert_allclose(float(right["reference"]),returns[:,k].mean(),atol=1e-10)
                    if j==0:
                        for arm in ("base","plain","branch"):
                            estimate=float(right[arm+"_Q"])-float(left[arm+"_Q"])
                            all_sq[arm,seed][init,frame].append((estimate-samples.mean())**2)
                    if winner:
                        for method in methods:
                            estimate=float(right[method])-float(left[method])
                            correct[method,seed][init,frame].append(.5 if estimate==0 else float(np.sign(estimate)==winner))
        assert evaluated==result["counts"]["anchors"] and eligible==result["counts"]["eligible_pairs"]
        assert len(families)==result["counts"]["eligible_families"] and len(rows)==evaluated*15
        for arm in ("base","plain","branch"):
            rmse=[np.sqrt(nested_mean(all_sq[arm,s])) for s in range(3)]
            np.testing.assert_allclose(rmse,result["errors"][arm]["gap_rmse_seeds"],rtol=1e-8,atol=1e-10)
        for method in methods:
            if eligible:
                accuracy=[nested_mean(correct[method,s]) for s in range(3)]
                np.testing.assert_allclose(accuracy,result["ranking"][method]["by_seed"],atol=1e-12)
            else:
                assert result["ranking"][method]["accuracy"] is None
        for arm in ("plain","branch"):
            actual=complete=0
            for ep_path in (out/"augmentation"/arm).glob("shard*/slot*.npz"):
                with np.load(ep_path,allow_pickle=False) as z:
                    t=len(z["reward"])
                    actual+=t
                    complete+=t*bool(z["terminated"][-1])
            budget=read(out/"training.json")["budgets"][arm]
            assert actual==6000==budget["spent"] and complete==budget["usable"]
        proof.append(dict(task=task,evaluated=evaluated,eligible=eligible,families=len(families),models=12))
    for arm,macro in summary["macro_gap_rmse"].items():
        np.testing.assert_allclose(macro,np.mean([r["errors"][arm]["gap_rmse"] for r in summary["tasks"]]))
    return proof


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    args=p.parse_args()
    print(json.dumps(check(args.run),indent=2))

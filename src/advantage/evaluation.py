"""Same-state candidate ranking against EXTERNALLY supplied reference labels."""
import numpy as np


def ranking_metrics(predictions,reference_q,min_gap=0.005):
    pred,truth=np.asarray(predictions,float),np.asarray(reference_q,float)
    if pred.ndim != 2 or pred.shape != truth.shape or pred.shape[1]<2 or pred.shape[0]<1:
        raise ValueError("expected matching nonempty anchor-by-candidate matrices")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all() or not np.isfinite(min_gap) or min_gap<0:
        raise ValueError("finite scores and nonnegative reference gap required")
    per_anchor=[];valid_pairs=0;all_pairs=0
    for p,q in zip(pred,truth):
        scores=[]
        for i in range(len(p)):
            for j in range(i+1,len(p)):
                all_pairs+=1
                if abs(q[i]-q[j])<=min_gap:
                    continue
                valid_pairs+=1
                diff=p[i]-p[j]
                scores.append(0.5 if abs(diff)<=1e-10 else float(np.sign(diff)==np.sign(q[i]-q[j])))
        if scores: per_anchor.append(float(np.mean(scores)))
    return {"ranking_accuracy":float(np.mean(per_anchor)) if per_anchor else None,
        "anchors":len(pred),"evaluable_anchors":len(per_anchor),"valid_pairs":valid_pairs,
        "pair_coverage":valid_pairs/all_pairs,"min_reference_gap":min_gap}

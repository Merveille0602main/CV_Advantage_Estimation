"""Discrimination and stability metrics for advantage estimators.

Every function here consumes an EXTERNAL reference label (oracle Q/V or
independently measured returns). Nothing in this module trains an estimator,
chooses a checkpoint, or treats a learned score as truth.

Terminology
-----------
anchor    : one state where several candidate actions are compared
candidate : one action considered at that anchor
pair      : two candidates at the same anchor; its reference gap decides
            whether the pair is distinguishable at all

A pair whose reference gap is at or below ``min_gap`` is kept in the records
but excluded from accuracy: it is not a label the estimator can be wrong
about. Coverage reports how many pairs survive that filter.
"""
import numpy as np

TIE_TOLERANCE = 1e-10


def _as_matrix(pred, truth, min_gap):
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if pred.ndim != 2 or pred.shape != truth.shape or pred.shape[1] < 2:
        raise ValueError("expected matching anchor-by-candidate matrices with at least two candidates")
    if pred.shape[0] < 1:
        raise ValueError("at least one anchor required")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("scores and reference labels must be finite")
    if not np.isfinite(min_gap) or min_gap < 0:
        raise ValueError("min_gap must be a finite nonnegative number")
    return pred, truth


def pairwise_records(pred, truth, min_gap=0.005, anchor_ids=None):
    """One record per (anchor, candidate pair), in a fixed candidate order."""
    pred, truth = _as_matrix(pred, truth, min_gap)
    if anchor_ids is None:
        anchor_ids = [str(i) for i in range(len(pred))]
    if len(anchor_ids) != len(pred):
        raise ValueError("one anchor id per row required")
    records = []
    for a, anchor_id in enumerate(anchor_ids):
        for i in range(pred.shape[1] - 1):
            for j in range(i + 1, pred.shape[1]):
                true_gap = float(truth[a, j] - truth[a, i])
                pred_gap = float(pred[a, j] - pred[a, i])
                distinguishable = bool(abs(true_gap) > min_gap)
                correct = None
                if distinguishable:
                    correct = 0.5 if abs(pred_gap) <= TIE_TOLERANCE else float(np.sign(pred_gap) == np.sign(true_gap))
                records.append(dict(anchor_index=a, anchor_id=anchor_id, candidate_i=i, candidate_j=j,
                                    pred_gap=pred_gap, true_gap=true_gap,
                                    distinguishable=distinguishable, correct=correct))
    return records


def per_anchor_counts(records):
    """Correct-pair count and evaluable-pair count per anchor."""
    anchors = sorted({r["anchor_index"] for r in records})
    correct = np.zeros(len(anchors), dtype=float)
    valid = np.zeros(len(anchors), dtype=int)
    index = {a: k for k, a in enumerate(anchors)}
    for r in records:
        if not r["distinguishable"]:
            continue
        correct[index[r["anchor_index"]]] += r["correct"]
        valid[index[r["anchor_index"]]] += 1
    return anchors, correct, valid


def aggregate_metrics(records):
    """Overall accuracy, coverage and per-anchor spread for one estimator."""
    anchors, correct, valid = per_anchor_counts(records)
    total_valid = int(valid.sum())
    total_pairs = len(records)
    per_anchor = [float(c / v) for c, v in zip(correct, valid) if v > 0]
    pairs_per_anchor = np.zeros(len(anchors), dtype=int)
    position = {anchor: k for k, anchor in enumerate(anchors)}
    for record in records:
        pairs_per_anchor[position[record["anchor_index"]]] += 1
    return dict(anchors=len(anchors), evaluable_anchors=len(per_anchor), pairs=total_pairs,
                valid_pairs=total_valid, pair_coverage=total_valid / total_pairs if total_pairs else 0.0,
                accuracy=float(correct.sum() / total_valid) if total_valid else None,
                anchor_accuracy_mean=float(np.mean(per_anchor)) if per_anchor else None,
                anchor_accuracy_std=float(np.std(per_anchor)) if per_anchor else None,
                evaluable_anchor_rate=float(np.mean(valid == pairs_per_anchor)) if len(anchors) else 0.0)


def anchor_records(records):
    """Per-anchor summary rows (one anchor can carry several candidate pairs)."""
    anchors, correct, valid = per_anchor_counts(records)
    by_anchor = {}
    for r in records:
        by_anchor.setdefault(r["anchor_index"], []).append(r)
    rows = []
    for a, correct_sum, valid_count in zip(anchors, correct, valid):
        group = by_anchor[a]
        rows.append(dict(anchor_index=a, anchor_id=group[0]["anchor_id"],
                         pairs=len(group), valid_pairs=int(valid_count),
                         correct=float(correct_sum),
                         accuracy=float(correct_sum / valid_count) if valid_count else None,
                         evaluable=bool(valid_count),
                         true_gap=group[0]["true_gap"], pred_gap=group[0]["pred_gap"]))
    return rows


def _bootstrap_indices(anchor_count, n_boot, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.integers(0, anchor_count, size=anchor_count)


def bootstrap_accuracy(pred, truth, min_gap=0.005, n_boot=2000, seed=0):
    """Cluster bootstrap over anchors for one estimator's pairwise accuracy.

    Resampling anchors (not pairs or frames) keeps within-state correlation
    intact: the same anchor cannot enter a resample more independently than
    it occurs in the data.
    """
    pred, truth = _as_matrix(pred, truth, min_gap)
    _, correct, valid = per_anchor_counts(pairwise_records(pred, truth, min_gap))
    point = float(correct.sum() / valid.sum()) if valid.sum() else None
    draws = []
    for idx in _bootstrap_indices(len(valid), n_boot, seed):
        total = int(valid[idx].sum())
        if total == 0:
            continue
        draws.append(float(correct[idx].sum() / total))
    if not draws:
        return dict(accuracy=point, ci_low=None, ci_high=None, bootstrap_samples=0)
    return dict(accuracy=point, ci_low=float(np.percentile(draws, 2.5)),
                ci_high=float(np.percentile(draws, 97.5)), bootstrap_samples=len(draws))


def bootstrap_paired_difference(pred_a, pred_b, truth, min_gap=0.005, n_boot=2000, seed=0):
    """Paired cluster bootstrap of accuracy(A) - accuracy(B).

    Both estimators are scored on the same resampled anchors, so the interval
    reflects the paired comparison rather than two independent samples.
    """
    pred_a, truth = _as_matrix(pred_a, truth, min_gap)
    pred_b, _ = _as_matrix(pred_b, truth, min_gap)
    _, correct_a, valid_a = per_anchor_counts(pairwise_records(pred_a, truth, min_gap))
    _, correct_b, valid_b = per_anchor_counts(pairwise_records(pred_b, truth, min_gap))
    draws = []
    for idx in _bootstrap_indices(len(valid_a), n_boot, seed):
        total_a, total_b = int(valid_a[idx].sum()), int(valid_b[idx].sum())
        if total_a == 0 or total_b == 0:
            continue
        draws.append(float(correct_a[idx].sum() / total_a - correct_b[idx].sum() / total_b))
    point = None
    if valid_a.sum() and valid_b.sum():
        point = float(correct_a.sum() / valid_a.sum() - correct_b.sum() / valid_b.sum())
    if not draws:
        return dict(delta=point, ci_low=None, ci_high=None, bootstrap_samples=0,
                    excludes_zero=None)
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return dict(delta=point, ci_low=low, ci_high=high, bootstrap_samples=len(draws),
                excludes_zero=bool(low > 0 or high < 0))


def gap_bin_accuracy(records, edges=(0.005, 0.02, 0.05, 0.1, float("inf"))):
    """Accuracy as a function of the reference gap between candidates.

    Small gaps are the hard cases for a local action-quality estimator; a
    single overall accuracy hides that.
    """
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        subset = [r for r in records if r["distinguishable"] and low <= abs(r["true_gap"]) < high]
        if not subset:
            rows.append(dict(gap_low=low, gap_high=high, pairs=0, accuracy=None))
            continue
        rows.append(dict(gap_low=low, gap_high=high, pairs=len(subset),
                         accuracy=float(np.mean([r["correct"] for r in subset]))))
    return rows


def gap_correlation(records):
    """Pearson correlation between predicted and reference gaps over all pairs.

    Kept for completeness, but on the toy MDP the reference gaps take only a
    few discrete values, so this is unstable and should not drive a
    conclusion. Prefer ``margin_calibration`` for scale and ``gap_bin_accuracy``
    for difficulty.
    """
    subset = [r for r in records if abs(r["true_gap"]) > 1e-12]
    if len(subset) < 3:
        return None
    pred = np.array([r["pred_gap"] for r in subset])
    truth = np.array([r["true_gap"] for r in subset])
    if pred.std() == 0 or truth.std() == 0:
        return None
    return float(np.corrcoef(pred, truth)[0, 1])


def margin_calibration(records):
    """Scale calibration on pairs the estimator can actually be judged on.

    ``margin_ratio_median`` is the median predicted-to-reference gap ratio
    (1.0 would mean no scale distortion); ``abs_margin_error_mean`` is the
    mean absolute error of the predicted margin itself.
    """
    subset = [r for r in records if r["distinguishable"] and abs(r["true_gap"]) > 1e-9]
    if not subset:
        return dict(margin_ratio_median=None, abs_margin_error_mean=None)
    ratios = [r["pred_gap"] / r["true_gap"] for r in subset]
    return dict(margin_ratio_median=float(np.median(ratios)),
                abs_margin_error_mean=float(np.mean([abs(r["pred_gap"] - r["true_gap"])
                                                     for r in subset])))


def stability_metrics(score_repeats, truth, min_gap=0.005):
    """Score dispersion and ordering stability across independent scoring runs.

    ``score_repeats`` has shape (anchor, candidate, repeat). Repeats model the
    randomness a score inherits from sampled future branches; for a
    deterministic estimator (Q-V) the repeat axis is constant and all
    stability numbers are trivially perfect.
    """
    scores = np.asarray(score_repeats, dtype=float)
    if scores.ndim != 3 or scores.shape[0] < 1 or scores.shape[2] < 1:
        raise ValueError("expected anchor-by-candidate-by-repeat scores")
    truth = np.asarray(truth, dtype=float)
    if truth.shape != scores.shape[:2]:
        raise ValueError("reference labels must match anchor-by-candidate")
    if not np.isfinite(scores).all() or not np.isfinite(truth).all():
        raise ValueError("scores and labels must be finite")
    repeats = scores.shape[2]
    dispersion = scores.std(axis=2)

    flips, comparisons, agreeing_anchors = 0, 0, []
    for a in range(scores.shape[0]):
        pairs = [(i, j) for i in range(scores.shape[1] - 1) for j in range(i + 1, scores.shape[1])]
        anchor_flip = 0
        anchor_pairs = 0
        for i, j in pairs:
            if abs(truth[a, j] - truth[a, i]) <= min_gap:
                continue
            gaps = scores[a, j, :] - scores[a, i, :]
            signs = np.where(np.abs(gaps) <= TIE_TOLERANCE, 0, np.sign(gaps))
            reference = np.sign(truth[a, j] - truth[a, i])
            for r in range(repeats - 1):
                for s in range(r + 1, repeats):
                    comparisons += 1
                    anchor_pairs += 1
                    if signs[r] != signs[s]:
                        flips += 1
                        anchor_flip += 1
        if anchor_pairs:
            agreeing_anchors.append(1.0 - anchor_flip / anchor_pairs)
    return dict(repeats=repeats,
                mean_score_std=float(dispersion.mean()),
                rank_flip_rate=flips / comparisons if comparisons else 0.0,
                anchor_ordering_agreement_mean=float(np.mean(agreeing_anchors)) if agreeing_anchors else None,
                comparisons=comparisons)


def majority_ordering_accuracy(score_repeats, truth, min_gap=0.005):
    """Accuracy of the sign of the mean score, the decision actually used."""
    scores = np.asarray(score_repeats, dtype=float)
    if scores.ndim != 3:
        raise ValueError("expected anchor-by-candidate-by-repeat scores")
    mean_gap = scores.mean(axis=2)
    records = pairwise_records(mean_gap, truth, min_gap)
    return aggregate_metrics(records)


def calibration_slope(pred, truth):
    """OLS slope of predicted advantage on reference advantage.

    1.0 means no scale distortion; 0.5 means predictions are compressed by
    half; a negative value means the ordering is inverted on average.
    """
    pred, truth = _as_matrix(pred, truth, 0.0)
    p, t = pred.ravel(), truth.ravel()
    if p.size < 2 or t.std() == 0 or p.std() == 0:
        return None
    return float(np.cov(p, t, bias=True)[0, 1] / t.var())


def advantage_error_metrics(pred, truth):
    """Absolute and scale-free error of an advantage estimate.

    ``normalized_rmse`` divides by the spread of the reference advantage so
    that settings with different return scales stay comparable; a raw RMSE
    across different gammas or horizons would not be.
    """
    pred, truth = _as_matrix(pred, truth, 0.0)
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    spread = float(np.std(truth))
    return dict(advantage_rmse=rmse, reference_spread=spread,
                normalized_rmse=rmse / spread if spread > 0 else None,
                calibration_slope=calibration_slope(pred, truth))


def accuracy_by_group(records, groups):
    """Pairwise accuracy per anchor group (e.g. normalized time-to-go)."""
    labels = []
    for record in records:
        label = groups.get(record["anchor_index"])
        if label is not None and label not in labels:
            labels.append(label)
    rows = []
    for label in labels:
        subset = [r for r in records if groups.get(r["anchor_index"]) == label]
        stats = aggregate_metrics(subset)
        rows.append(dict(group=label, pairs=stats["pairs"],
                         accuracy=stats["accuracy"]))
    return rows


def correctness_map(records):
    """anchor_index -> 1/0.5/0, or omitted when the pair is not evaluable."""
    return {r["anchor_index"]: r["correct"] for r in records if r["distinguishable"]}

"""Classification, calibration and action-selection metrics.

Accuracy alone is not comparable across the three representations: a
classifier's accuracy depends on its threshold and class prevalence, and a
continuous regressor has units. The common operational metrics used across
all representations are top-1 selection and regret at the same state.
"""
import numpy as np


def _masked(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must align")
    keep = labels >= 0
    return labels[keep], scores[keep]


def auroc(labels, scores):
    """Rank-based AUROC with average ranks for ties; None if one class."""
    from .labels import IGNORE
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    keep = labels != IGNORE
    y, s = labels[keep], scores[keep]
    positive = y == 1
    negative = y == 0
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ordered = s[order]
    ranks = np.empty(len(s), dtype=float)
    index = 0
    while index < len(s):
        end = index
        while end + 1 < len(s) and ordered[end + 1] == ordered[index]:
            end += 1
        ranks[order[index:end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    statistic = ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0
    return float(statistic / (n_pos * n_neg))


def average_precision(labels, scores):
    """Area under the precision-recall curve, step-wise."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    keep = labels >= 0
    y, s = labels[keep], scores[keep]
    if not (y == 1).any():
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    true_positive = np.cumsum(y == 1)
    predicted = np.arange(1, len(y) + 1)
    precision = true_positive / predicted
    recall = true_positive / true_positive[-1]
    recall = np.concatenate([[0.0], recall])
    return float(np.sum(np.diff(recall) * precision))


def brier(labels, probabilities):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    keep = labels >= 0
    if keep.sum() == 0:
        return None
    return float(np.mean((probabilities[keep] - labels[keep]) ** 2))


def expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    keep = labels >= 0
    y, p = labels[keep], probabilities[keep]
    if len(y) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y))
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        members = (p >= low) & (p < high if high < 1.0 else p <= high)
        if not members.any():
            continue
        confidence = float(p[members].mean())
        accuracy = float(y[members].mean())
        error += members.sum() / total * abs(accuracy - confidence)
    return float(error)


def threshold_accuracy(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    keep = labels >= 0
    if keep.sum() == 0:
        return None
    predicted = (probabilities[keep] >= threshold).astype(float)
    return float(np.mean(predicted == labels[keep]))


def binary_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    keep = labels >= 0
    prevalence = float(np.mean(labels[keep] == 1)) if keep.any() else None
    return dict(auroc=auroc(labels, probabilities),
                average_precision=average_precision(labels, probabilities),
                brier=brier(labels, probabilities),
                ece=expected_calibration_error(labels, probabilities),
                threshold_accuracy=threshold_accuracy(labels, probabilities),
                positive_rate=prevalence)


def selection_metrics(scores, truth, min_gap=0.005):
    """Top-1 accuracy and regret for choosing the highest-scoring action.

    A choice within ``min_gap`` of the best is counted as correct, matching
    the tie rule used by the pairwise ranking metrics.
    """
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if scores.shape != truth.shape or scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("expected matching anchor-by-candidate matrices")
    if not np.isfinite(scores).all() or not np.isfinite(truth).all():
        raise ValueError("scores and labels must be finite")
    chosen = scores.argmax(axis=1)
    rows = np.arange(len(scores))
    best = truth.max(axis=1)
    regret = best - truth[rows, chosen]
    spread = truth.std(axis=1)
    normalised = np.where(spread > 1e-12, regret / np.maximum(spread, 1e-12),
                          np.where(regret <= min_gap, 0.0, 1.0))
    return dict(top1=float(np.mean(regret <= min_gap)),
                regret=float(np.mean(regret)),
                normalized_regret=float(np.mean(normalised)))


def bootstrap(metric, n, n_boot=2000, seed=0):
    """Cluster bootstrap over anchors for an anchor-indexed statistic."""
    if n < 1 or n_boot < 1:
        raise ValueError("positive anchor count and bootstrap budget required")
    rng = np.random.default_rng(seed)
    point = metric(np.arange(n))
    draws = []
    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        value = metric(index)
        if value is not None and np.isfinite(value):
            draws.append(float(value))
    if not draws:
        return dict(value=point, ci_low=None, ci_high=None, samples=0)
    return dict(value=point, ci_low=float(np.percentile(draws, 2.5)),
                ci_high=float(np.percentile(draws, 97.5)), samples=len(draws))


def bootstrap_difference(first, second, n_boot=2000, seed=0):
    """Paired cluster bootstrap of mean(first) - mean(second)."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError("paired values must align")
    n = len(first)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        draws.append(float(np.mean(first[index] - second[index])))
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return dict(delta=float(np.mean(first - second)), ci_low=low, ci_high=high,
                excludes_zero=bool(low > 0 or high < 0), samples=len(draws))

"""Same-state candidate ranking against externally supplied reference labels."""
from .discrimination import aggregate_metrics, pairwise_records


def ranking_metrics(predictions, reference_q, min_gap=0.005):
    """Summarize candidate rankings with equal weight per evaluable state."""
    metrics = aggregate_metrics(pairwise_records(predictions, reference_q, min_gap))
    return {
        "ranking_accuracy": metrics["anchor_accuracy_mean"],
        "anchors": metrics["anchors"],
        "evaluable_anchors": metrics["evaluable_anchors"],
        "valid_pairs": metrics["valid_pairs"],
        "pair_coverage": metrics["pair_coverage"],
        "min_reference_gap": min_gap,
    }

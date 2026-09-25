"""Evaluation metrics for entity resolution (F_0.5)."""
import numpy as np


def f05_score_single(predicted_ids, true_ids):
    """F_0.5 for a single S1 entity.

    - Both empty (singleton correctly identified) → 1.0
    - Predicted empty, truth non-empty (missed) → 0.0
    - Predicted non-empty, truth empty (false merge) → 0.0
    """
    pred = set(predicted_ids) if predicted_ids else set()
    true = set(true_ids) if true_ids else set()

    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0

    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def evaluate_predictions(predictions, ground_truth):
    """Macro-averaged F_0.5 across all S1 entities in ground_truth.

    Args:
        predictions:  {s1_id: set(matched_ids)}
        ground_truth: {s1_id: set(true_ids)}

    Returns:
        float — macro-averaged F_0.5
    """
    scores = []
    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        scores.append(f05_score_single(pred_ids, true_ids))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_blocking_recall(candidates, ground_truth):
    """Fraction of true match IDs that appear in the candidate set.

    This is the recall ceiling — any true match missing from candidates
    cannot be recovered by the classifier.
    """
    total_true = 0
    found = 0

    for s1_id, true_ids in ground_truth.items():
        if not true_ids:
            continue
        cands = set(candidates.get(s1_id, []))
        total_true += len(true_ids)
        found += len(true_ids & cands)

    return found / total_true if total_true > 0 else 0.0

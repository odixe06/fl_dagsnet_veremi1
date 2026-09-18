"""The 10 metric contract, computed from a summed confusion matrix.

Evaluation is sharded across two GPUs, so the metrics are derived from integer
confusion counts rather than from a concatenated prediction vector: integer sums
are exactly order-independent, float metric averages are not.
"""
import numpy as np

METRIC_KEYS = [
    "accuracy",
    "precision_macro", "precision_micro", "precision_weighted",
    "recall_macro",    "recall_micro",    "recall_weighted",
    "f1_macro",        "f1_micro",        "f1_weighted",
]


def _safe_div(num, den):
    """0/0 -> 0, matching sklearn's zero_division=0."""
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def metrics_from_confusion(cm):
    """All 10 metrics from `cm[true, pred]` integer counts.

    Returns the METRIC_KEYS dict; every key is always present even when several
    of them coincide (they do: micro precision/recall/F1 and weighted recall all
    equal accuracy in single-label multi-class classification).
    """
    cm = np.asarray(cm, dtype=np.int64)
    assert cm.ndim == 2 and cm.shape[0] == cm.shape[1], f"bad confusion shape {cm.shape}"
    n = cm.sum()
    assert n > 0, "empty confusion matrix"

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)          # n_c, true counts
    predicted = cm.sum(axis=0).astype(np.float64)        # TP_c + FP_c

    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    accuracy = float(tp.sum() / n)
    w = support / n

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision.mean()),
        "precision_micro": float(tp.sum() / predicted.sum()),
        "precision_weighted": float((w * precision).sum()),
        "recall_macro": float(recall.mean()),
        "recall_micro": float(tp.sum() / support.sum()),
        "recall_weighted": float((w * recall).sum()),
        "f1_macro": float(f1.mean()),
        "f1_micro": accuracy,          # 2PR/(P+R) with P == R == accuracy
        "f1_weighted": float((w * f1).sum()),
    }


def per_class_report(cm, class_names):
    """Per-class precision / recall / F1 / support, as a list of dicts."""
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return [
        {"class_index": i, "class_name": class_names[i],
         "precision": float(precision[i]), "recall": float(recall[i]),
         "f1": float(f1[i]), "support": int(support[i]),
         "predicted": int(predicted[i])}
        for i in range(len(class_names))
    ]

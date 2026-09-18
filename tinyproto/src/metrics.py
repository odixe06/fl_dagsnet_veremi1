"""The 10 metrics, computed from integer confusion counts.

Schema and order are fixed by the project's `references/metrics.md`. Every value here is derived
from a full confusion matrix, never averaged over batches, and never over a padded or truncated
test set.

In TinyProto every client keeps its OWN model, so there is no global model to score. Following
the paper ("the average test accuracy across all clients per round") the headline number is the
mean over clients of a per-client metric. Two aggregates are reported:

  * `mean_over_clients` — compute the 10 metrics for each client, then average. This is the
    paper's protocol and the one to quote.
  * `pooled` — sum the per-client confusion matrices, then compute the 10 metrics once. This
    weights a client by how confidently it is wrong and is NOT the paper's number; it is kept
    because it answers a different, also-interesting question.

Two prediction rules are scored from the same forward pass:

  * `proto` — Eq. (12), argmin L2 distance to the client's own local prototypes. The paper's
    rule for PBFL methods, and the primary result.
  * `clf`   — argmax of the DAGSNet classifier head. Free to compute, and the only rule that
    can predict a class the client has never seen.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

METRIC_KEYS = [
    "accuracy",
    "precision_macro", "precision_micro", "precision_weighted",
    "recall_macro", "recall_micro", "recall_weighted",
    "f1_macro", "f1_micro", "f1_weighted",
]

RULES = ("proto", "clf")


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    """All 10 metrics from a (C, C) confusion matrix with rows = true, columns = predicted.

    Mirrors sklearn with `labels=arange(C)` and `zero_division=0`: a class the model never
    predicts contributes precision 0 rather than NaN, and macro averaging still divides by C.
    """
    cm = np.asarray(cm, dtype=np.float64)
    C = cm.shape[0]
    tp = np.diag(cm)
    pred = cm.sum(axis=0)                 # TP + FP per class
    true = cm.sum(axis=1)                 # TP + FN per class  (== support)
    n = cm.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pred > 0, tp / np.where(pred > 0, pred, 1), 0.0)
        r = np.where(true > 0, tp / np.where(true > 0, true, 1), 0.0)
        f = np.where((p + r) > 0, 2 * p * r / np.where((p + r) > 0, p + r, 1), 0.0)

    acc = float(tp.sum() / n) if n > 0 else 0.0
    w = true / n if n > 0 else np.zeros(C)
    out = {
        "accuracy": acc,
        "precision_macro": float(p.mean()), "precision_micro": acc,
        "precision_weighted": float((w * p).sum()),
        "recall_macro": float(r.mean()), "recall_micro": acc,
        "recall_weighted": float((w * r).sum()),
        "f1_macro": float(f.mean()), "f1_micro": acc,
        "f1_weighted": float((w * f).sum()),
    }
    return {k: out[k] for k in METRIC_KEYS}


def per_class_from_confusion(cm: np.ndarray, class_names: list[str]) -> list[dict]:
    cm = np.asarray(cm, dtype=np.float64)
    tp, pred, true = np.diag(cm), cm.sum(axis=0), cm.sum(axis=1)
    rows = []
    for c, name in enumerate(class_names):
        p = tp[c] / pred[c] if pred[c] > 0 else 0.0
        r = tp[c] / true[c] if true[c] > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append({"class": name, "support": int(true[c]),
                     "precision": float(p), "recall": float(r), "f1": float(f)})
    return rows


def aggregate_clients(cms: np.ndarray) -> dict:
    """cms: (n_clients, C, C) integer confusion matrices for ONE prediction rule."""
    per_client = [metrics_from_confusion(cm) for cm in cms]
    arr = np.array([[m[k] for k in METRIC_KEYS] for m in per_client], dtype=np.float64)
    pooled = metrics_from_confusion(cms.sum(axis=0))
    return {
        "per_client": per_client,
        "mean_over_clients": {k: float(v) for k, v in zip(METRIC_KEYS, arr.mean(axis=0))},
        "std_over_clients": {k: float(v) for k, v in zip(METRIC_KEYS, arr.std(axis=0, ddof=0))},
        "min_over_clients": {k: float(v) for k, v in zip(METRIC_KEYS, arr.min(axis=0))},
        "max_over_clients": {k: float(v) for k, v in zip(METRIC_KEYS, arr.max(axis=0))},
        "pooled": pooled,
    }


def check_metrics(m: dict[str, float], n_rows: int, cm_total: int, *, tol: float = 1e-9) -> None:
    """Refuse to publish a round whose numbers cannot be right."""
    missing = set(METRIC_KEYS) - set(m)
    if missing:
        raise ValueError(f"incomplete metric schema, missing {sorted(missing)}")
    for k in METRIC_KEYS:
        v = m[k]
        if not np.isfinite(v) or not (0.0 - tol <= v <= 1.0 + tol):
            raise ValueError(f"metric {k} out of range: {v}")
    if cm_total != n_rows:
        raise ValueError(f"confusion matrix covers {cm_total} rows, expected {n_rows}")
    # single-label multi-class identity: four columns must collapse onto accuracy
    for k in ("precision_micro", "recall_micro", "f1_micro", "recall_weighted"):
        if abs(m[k] - m["accuracy"]) > 1e-9:
            raise ValueError(f"{k}={m[k]} != accuracy={m['accuracy']}; the collapse identity broke")


# ---------------------------------------------------------------------------
# atomic artifact writes
# ---------------------------------------------------------------------------

def atomic_write_json(path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_csv(path, rows: list[dict], columns: list[str]) -> None:
    """Whole-file rewrite through a temp file. `rows` must be the COMPLETE table: history is
    rebuilt from the per-round JSON on every write, so a crash mid-rewrite can never leave a
    permanently truncated history (verifying-artifacts.md §2)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def history_row(rnd: int, agg: dict, extra: dict) -> dict:
    """One flat CSV row: the paper's aggregate for both rules, plus run diagnostics."""
    row = {"round": int(rnd)}
    for rule in RULES:
        for k in METRIC_KEYS:
            row[f"{rule}_{k}"] = agg[rule]["mean_over_clients"][k]
        row[f"{rule}_f1_macro_std"] = agg[rule]["std_over_clients"]["f1_macro"]
        row[f"{rule}_f1_macro_min"] = agg[rule]["min_over_clients"]["f1_macro"]
        row[f"{rule}_f1_macro_max"] = agg[rule]["max_over_clients"]["f1_macro"]
        row[f"{rule}_pooled_f1_macro"] = agg[rule]["pooled"]["f1_macro"]
    row.update(extra)
    return row


def history_columns(extra_keys: list[str]) -> list[str]:
    cols = ["round"]
    for rule in RULES:
        cols += [f"{rule}_{k}" for k in METRIC_KEYS]
        cols += [f"{rule}_f1_macro_std", f"{rule}_f1_macro_min",
                 f"{rule}_f1_macro_max", f"{rule}_pooled_f1_macro"]
    return cols + list(extra_keys)

import os, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

METRIC_KEYS = ["accuracy",
               "precision_macro", "precision_micro", "precision_weighted",
               "recall_macro",    "recall_micro",    "recall_weighted",
               "f1_macro",        "f1_micro",        "f1_weighted"]


def compute_metrics(y_true, y_pred, num_classes):
    """All 10 metrics from one pass. labels=arange pins the class axis so a class the
    model never predicts still occupies its column in the macro average."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    labels = np.arange(num_classes)
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "micro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, labels=labels, zero_division=0)
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"]    = float(r)
        out[f"f1_{avg}"]        = float(f)
    return out


def atomic_write_json(path, obj):
    path = Path(path); tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2)); os.replace(tmp, path)


def append_csv(path, row, columns):
    path = Path(path); new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new: w.writeheader()
        w.writerow(row)


def save_round_artifacts(round_idx, y_true, y_pred, y_prob, num_classes,
                         class_names, out_dir, extra=None, save_prob=False, metrics=None,
                         is_best=False):
    """y_pred is written every round (10.7 MB as int8). y_prob is 10.7M x 16 floats —
    344 MB per round, so 50 rounds would blow Kaggle's 20 GB output cap. It is written
    only for the best-f1_macro round and the final round (deviation 24).

    `metrics` takes an already-computed dict: the caller needs f1_macro to decide
    `save_prob`, and recomputing it here meant three more sklearn passes over 10.7M rows
    every single round."""
    out_dir = Path(out_dir); extra = extra or {}
    m = dict(metrics) if metrics is not None else compute_metrics(y_true, y_pred, num_classes)
    m["round"] = int(round_idx); m.update(extra)

    atomic_write_json(out_dir / "metrics" / f"round_{round_idx:03d}.json", m)
    append_csv(out_dir / "metrics" / "history.csv", m,
               columns=["round", *METRIC_KEYS, *extra.keys()])

    np.savez_compressed(out_dir / "preds" / f"round_{round_idx:03d}.npz",
                        y_pred=y_pred.astype(np.int8))
    if save_prob:
        # NOT savez_compressed: zlib on 344 MB of fp16 softmax costs a minute or more of
        # wall clock while rank 1 sits in the barrier.
        path = out_dir / "preds" / f"prob_round_{round_idx:03d}.npz"
        np.savez(path, y_prob=y_prob.astype(np.float16))
        # "best and final" means two files, not one per improving round. f1_macro improves
        # on most rounds early, so without this the run keeps every superseded copy —
        # 344 MB each, up to 16 GB by round 50. A new best supersedes the old one; the
        # final round is written after the last possible best, so it survives.
        if is_best:
            for old in Path(out_dir / "preds").glob("prob_round_*.npz"):
                if old != path:
                    old.unlink()
    yt = out_dir / "preds" / "y_true.npy"
    if not yt.exists():
        np.save(yt, y_true.astype(np.int8))          # identical every round — stored once

    np.save(out_dir / "confusion" / f"round_{round_idx:03d}.npy",
            confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)))
    with open(out_dir / "reports" / f"round_{round_idx:03d}.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, labels=np.arange(num_classes),
                                      target_names=class_names, digits=4, zero_division=0))
    return m

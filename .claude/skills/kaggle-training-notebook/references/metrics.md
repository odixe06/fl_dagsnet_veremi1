# The 10 metrics

Computed on the **entire test set** after every round, from one logical pass. Rank 0 may
evaluate alone, or workers may evaluate disjoint shards and sum integer confusion counts.
Never pad, drop or duplicate test rows. Notation: `C` classes, `N` test samples,
`n_c` true samples of class `c`, and per-class counts `TP_c`, `FP_c`, `FN_c`.

Each notebook markdown cell states the formula first; the code cell below implements it.

`METRIC_KEYS` below is the canonical schema and display order for the notebook and reports.
Keep all 10 named columns in per-round JSON/CSV, the rendered notebook table, and every
per-build or aggregate `report.md` result table. Equal values do not justify dropping columns.
Charts and brief heartbeat lines may emphasize selected metrics, but cannot replace that table.
Log all 10 as `eval/<metric>` when W&B is enabled, using the same computed result dictionary.

## 1. Accuracy

Share of correct predictions over the whole set.

$$\text{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\left[\hat{y}_i=y_i\right]=\frac{\sum_{c}TP_c}{N}$$

## Precision — 2. Macro, 3. Micro, 4. Weighted

Per class: $P_c=\dfrac{TP_c}{TP_c+FP_c}$

$$P_{\text{macro}}=\frac{1}{C}\sum_{c=1}^{C}P_c \qquad P_{\text{micro}}=\frac{\sum_c TP_c}{\sum_c (TP_c+FP_c)} \qquad P_{\text{weighted}}=\sum_{c=1}^{C}\frac{n_c}{N}P_c$$

Macro treats every class as equally important regardless of size. Micro pools all TP/FP first, so the majority classes dominate. Weighted scales each class by its true share `n_c/N`.

## Recall — 5. Macro, 6. Micro, 7. Weighted

Per class: $R_c=\dfrac{TP_c}{TP_c+FN_c}$

$$R_{\text{macro}}=\frac{1}{C}\sum_{c=1}^{C}R_c \qquad R_{\text{micro}}=\frac{\sum_c TP_c}{\sum_c (TP_c+FN_c)} \qquad R_{\text{weighted}}=\sum_{c=1}^{C}\frac{n_c}{N}R_c$$

## F1 — 8. Macro, 9. Micro, 10. Weighted

Per class, the harmonic mean: $F1_c=\dfrac{2\,P_c R_c}{P_c+R_c}$

$$F1_{\text{macro}}=\frac{1}{C}\sum_{c=1}^{C}F1_c \qquad F1_{\text{micro}}=\frac{2\,P_{\text{micro}}R_{\text{micro}}}{P_{\text{micro}}+R_{\text{micro}}} \qquad F1_{\text{weighted}}=\sum_{c=1}^{C}\frac{n_c}{N}F1_c$$

**Identity to state in the notebook:** in single-label multi-class classification every prediction is exactly one class, so $\sum_c FP_c=\sum_c FN_c$ and therefore

$$P_{\text{micro}}=R_{\text{micro}}=F1_{\text{micro}}=\text{Accuracy}$$

Four of the ten columns collapse to one number. That is correct, not a bug — say so in the markdown cell so nobody reads it as an error. `Recall_weighted` also equals accuracy for the same reason. The metrics that actually separate models on imbalanced network-traffic data are **macro F1** (every attack class counts equally) and **weighted F1** (reliability under the real class mix).

## Implementation

```python
import json, numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

METRIC_KEYS = [
    "accuracy",
    "precision_macro", "precision_micro", "precision_weighted",
    "recall_macro",    "recall_micro",    "recall_weighted",
    "f1_macro",        "f1_micro",        "f1_weighted",
]

def compute_metrics(y_true, y_pred, num_classes):
    """All 10 metrics from one pass. labels= pins the class axis so classes
    absent from a batch of predictions still occupy their column."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    labels = np.arange(num_classes)
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "micro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, labels=labels, zero_division=0
        )
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"]    = float(r)
        out[f"f1_{avg}"]        = float(f)
    return {key: out[key] for key in METRIC_KEYS}
```

`zero_division=0` matters: a rare attack class the model never predicts gives `0/0` precision, and the default would spam warnings and return an inconsistent value. `labels=np.arange(num_classes)` keeps macro averaging over **all** classes rather than only the ones that showed up.

## What to persist per round

Named `save_round_artifacts` so it never collides with `ckpt.save_round`, which writes the weights.

```python
import os, csv, json
from pathlib import Path

def atomic_write_json(path, obj):
    path = Path(path); tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2)); os.replace(tmp, path)

def append_csv(path, row, columns):
    """Append-or-create. Rewrites the row in `columns` order so a resumed
    session cannot shift the header."""
    path = Path(path); new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new: w.writeheader()
        w.writerow(row)

def save_round_artifacts(round_idx, y_true, y_pred, y_prob, num_classes,
                         class_names, out_dir, extra=None):
    out_dir = Path(out_dir); extra = extra or {}
    m = compute_metrics(y_true, y_pred, num_classes)
    m["round"] = int(round_idx)
    m.update(extra)                            # loss, lr, seconds, peak_gb, ...

    atomic_write_json(out_dir / "metrics" / f"round_{round_idx:03d}.json", m)
    append_csv(out_dir / "metrics" / "history.csv", m,
               columns=["round", *METRIC_KEYS, *extra.keys()])

    # y_true + y_pred are enough to recompute every metric in this document.
    # Include y_prob only when the run's output budget allows it.
    pred_payload = {"y_true": y_true, "y_pred": y_pred}
    if y_prob is not None:
        pred_payload["y_prob"] = y_prob
    np.savez_compressed(out_dir / "preds" / f"round_{round_idx:03d}.npz",
                        **pred_payload)
    np.save(out_dir / "confusion" / f"round_{round_idx:03d}.npy",
            confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)))
    with open(out_dir / "reports" / f"round_{round_idx:03d}.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, labels=np.arange(num_classes),
                                      target_names=class_names, digits=4, zero_division=0))
    return m
```

`history.csv` is appended, so a resumed run that redoes a round it had already logged would write a duplicate row. Guard it by de-duplicating on `round` (keeping the last) when the results cell reads the file:
`hist = pd.read_csv(p).drop_duplicates(subset='round', keep='last').sort_values('round')`.

`y_prob` enables threshold sweeps, ROC/PR curves, and calibration, but it can exceed Kaggle's output limit on a large multi-class test set. Estimate `rounds × N × C × bytes_per_probability` before launch. Always keep `y_pred` for every round; keep `y_prob` every round only when it fits, otherwise keep it for the best and final rounds and record that storage policy in `rebuild.md`. This repository's VeReMi run follows the best-and-final policy.

## Reporting

Before publishing a round, check that all 10 keys are present and numeric, finite, and in
`[0, 1]`, and that the confusion-matrix count equals the full fixed test count (10,761,343
for this repository's recorded dataset). Aggregate counts over the full set; do not average
batch-level macro/weighted metrics. Persist full precision and round only for display.
Validate the collapse identity within floating-point tolerance while retaining every column:
`accuracy = precision_micro = recall_micro = recall_weighted = f1_micro`.

If an older artifact lacks a metric, reconstruct it only from the matching full-test
predictions or raw confusion counts with the frozen class axis and `zero_division=0`.
Record the source and reconstruction; do not overwrite original evidence. If evidence is
insufficient, retain the column as `N/A` with a reason and mark results incomplete. A failed
run's finite scores may still be scientifically invalid; disclose the affected rounds.

After each round print one line per round and keep the table cumulative, so the trend is visible while training is still going:

```
round  loss    acc     P_mac   R_mac   F1_mac  F1_wtd  time
  001  0.4123  0.9412  0.7821  0.7433  0.7601  0.9388  412s
```

At the end, render `history.csv` as a full 10-column table plus the final round's per-class report and confusion matrix. Put paper and rebuild numbers in one comparison table only when their dataset, task, class semantics, and metric definitions are commensurable and project documentation permits it. Otherwise report them in separate tables and explain why a numerical delta would be misleading.

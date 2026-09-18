# Dataset audit

Run before writing model code. The point is that the notebook's shapes, class count, normalization and loss are derived from **measured** facts, not from filenames or the paper's dataset.

Audit **train and test separately, then compare them.** Most silent failures in this domain are train/test disagreements, not model bugs.

## The checklist

Report every line back to the user.

**Inventory**
- Files, sizes, formats; row counts for train and test; the train:test ratio.
- Is the test split fixed and given, or does it need creating? If created: stratified, fixed seed, and — for traffic captures — **split by time or by flow/session, never by random row**, so packets from one flow cannot land on both sides.

**Labels**
- Exact label column name, dtype, and raw values before any mapping.
- `C` = number of classes, and the frozen `class_names` list ordered by index. Persist the mapping to JSON — it must be identical across rounds and sessions.
- Per-class counts for train **and** test. State the imbalance ratio (max class / min class).
- Classes present in test but absent from train, and vice versa. Either is a stop-and-ask.
- Whitespace, case, and label-name drift between the two files (`DoS Hulk` vs `DoS_Hulk`).

**Features**
- Column count and dtypes; which are numeric, categorical, identifier-like.
- NaN, `inf`/`-inf`, and constant (zero-variance) columns. Flow-feature exports produce `inf` in rate columns routinely — decide clip vs drop vs impute and say which.
- Duplicate rows, and duplicates that cross train/test.
- Value ranges per feature: the scale gap between e.g. byte counts and flag booleans decides whether standardization is mandatory.

**Leakage** — the audit's highest-value column for network traffic. Flag and drop by default, asking before keeping any:
`Flow ID`, source/destination IP, source/destination port, timestamp, and any per-row index or capture-file identifier. A model that reaches 0.999 macro-F1 on an IDS dataset is nearly always reading an address or a port, not behaviour.

**Schema agreement**
- Same columns, same order, same dtypes in train and test. Report any difference explicitly.
- Same label mapping applied to both, built from the **union** of label values but with indices frozen by the train ordering.

**Sequence data** (packet/flow sequences rather than one row per flow)
- Length distribution: min, median, p95, max. `MAX_LEN` comes from p95–p99, not from max.
- Grouping key (flow 5-tuple / session id) and whether groups are contiguous in the file.
- Padding value and whether a mask is needed.

**Normalization**
- Statistics are fit on **train only** and applied to test. Persist `mean`/`std` (or quantiles) to disk next to the class mapping so a resumed session reuses identical statistics rather than refitting.
- For long-tailed byte/duration columns, log1p before standardizing usually beats raw standardization; test it rather than assuming.

## Audit cell

Written to run either locally or as a Kaggle probe notebook. It prints — it does not modify data.

```python
import numpy as np, pandas as pd, json
from pathlib import Path

def audit(df: pd.DataFrame, name: str, label_col: str):
    print(f"\n{'='*70}\n{name}: {df.shape[0]:,} rows x {df.shape[1]} cols\n{'='*70}")

    vc = df[label_col].value_counts(dropna=False)
    print(f"\n-- labels ({vc.size} classes) --")
    print(pd.DataFrame({"count": vc, "pct": (100*vc/len(df)).round(3)}))
    print(f"imbalance ratio: {vc.max()/max(vc.min(),1):,.1f} : 1")

    num = df.select_dtypes(include=[np.number])
    print(f"\n-- dtypes -- numeric={num.shape[1]} other={df.shape[1]-num.shape[1]}")
    print(df.dtypes.value_counts())

    nan  = df.isna().sum();                     nan  = nan[nan > 0]
    inf  = np.isinf(num).sum();                 inf  = inf[inf > 0]
    cst  = num.columns[num.nunique() <= 1]
    print(f"\n-- health --")
    print(f"NaN cols     : {len(nan)}"); print(nan.head(20).to_string() if len(nan) else "  none")
    print(f"inf cols     : {len(inf)}"); print(inf.head(20).to_string() if len(inf) else "  none")
    print(f"constant cols: {list(cst)[:20]}")
    print(f"duplicate rows: {df.duplicated().sum():,}")

    LEAK = ("flow id","src ip","dst ip","source ip","destination ip","src port",
            "dst port","source port","destination port","timestamp","unnamed: 0","id")
    hits = [c for c in df.columns if c.strip().lower() in LEAK]
    print(f"\n-- leakage candidates -- {hits if hits else 'none'}")

    print(f"\n-- ranges (first 12 numeric) --")
    print(num.describe().T[["min","max","mean","std"]].head(12))
    return set(df.columns), set(map(str, vc.index))

tr_cols, tr_lab = audit(train_df, "TRAIN", LABEL_COL)
te_cols, te_lab = audit(test_df,  "TEST",  LABEL_COL)

print("\n== train/test agreement ==")
print("cols only in train:", sorted(tr_cols - te_cols) or "none")
print("cols only in test :", sorted(te_cols - tr_cols) or "none")
print("labels only in train:", sorted(tr_lab - te_lab) or "none")
print("labels only in test :", sorted(te_lab - tr_lab) or "none")   # <- must be none
```

## Stop-and-ask triggers

Report these to the user and wait rather than deciding alone:

- A class appears in test but never in train.
- The test set has no labels (it is a submission set) — then a labelled validation split has to be carved from train, and the "full test set" evaluation means that split.
- Imbalance beyond ~1000:1, where the choice between class weights, focal loss, and resampling changes the reproduced result.
- Leakage columns that the paper being reproduced explicitly used.
- Train and test schemas that disagree in ways no rename fixes.

## Freeze the artifacts

Write `meta.json` beside the checkpoints and load it on resume instead of recomputing:

```python
meta = {"class_names": class_names, "num_classes": len(class_names),
        "feature_cols": feature_cols, "dropped_cols": dropped,
        "norm": {"mean": mean.tolist(), "std": std.tolist()},
        "max_len": MAX_LEN, "label_map": label_map}
```

A resumed session that re-derives its own label ordering silently scores every round after the interruption against a different class axis. Freezing the mapping is what prevents that.

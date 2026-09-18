"""Real audit of the VeReMi FL partitions + centralized test. Streams; ~1.6 GiB RSS ceiling."""
import json, math, sys, time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

ROOT = Path("/home/odixe/nckh/dataset")
FL   = ROOT / "fl_client/alpha05"
TEST = ROOT / "centralized/test"
META = json.load(open("/home/odixe/nckh/tinyproto/knowledge/meta.json"))
SCAL = json.load(open("/home/odixe/nckh/tinyproto/knowledge/scaler.json"))
FEATS = META["feature_cols"]; CLASSES = META["class_names"]; C = len(CLASSES)
out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "scenarios": {}, "test": {}}

def footer_rows(files):
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)

# ---------- 1. per-scenario footprint from parquet FOOTERS only ----------
for n in (20, 50, 100):
    base = FL / f"{n}_client" / "train"
    cdirs = sorted(base.glob("client_id=*"))
    per = {}
    tot_rows = tot_bytes = tot_files = 0
    for d in cdirs:
        fs = sorted(d.glob("*.parquet"))
        r = footer_rows(fs); b = sum(f.stat().st_size for f in fs)
        per[int(d.name.split("=")[1])] = {"rows": r, "files": len(fs), "bytes": b}
        tot_rows += r; tot_bytes += b; tot_files += len(fs)
    stats = json.load(open(FL / f"{n}_client" / "client_stats.json"))
    sidecar = {c["client_id"]: c["rows"] for c in stats["clients"]}
    mismatch = {k: (v["rows"], sidecar.get(k)) for k, v in per.items() if sidecar.get(k) != v["rows"]}
    out["scenarios"][n] = {
        "n_client_dirs": len(cdirs), "footer_rows_total": tot_rows,
        "sidecar_rows_total": stats["totals"]["rows"],
        "rows_match_sidecar": not mismatch, "mismatch": mismatch,
        "files": tot_files, "bytes": tot_bytes,
        "rows_min": min(v["rows"] for v in per.values()),
        "rows_max": max(v["rows"] for v in per.values()),
        "per_client_rows": {str(k): per[k]["rows"] for k in sorted(per)},
        "per_client_files": {str(k): per[k]["files"] for k in sorted(per)},
    }
    print(f"[footer] {n} clients: dirs={len(cdirs)} rows={tot_rows:,} files={tot_files} "
          f"bytes={tot_bytes/2**30:.2f} GiB match={not mismatch}", flush=True)

# ---------- 2. schema of one client file ----------
sample_f = sorted((FL / "20_client/train/client_id=000").glob("*.parquet"))[0]
sch = pq.ParquetFile(sample_f).schema_arrow
cols = list(sch.names)
out["client_schema"] = {
    "file": str(sample_f), "n_columns": len(cols), "columns": cols,
    "f_columns": [c for c in cols if c.startswith("f_")],
    "f_order_matches_meta": [c for c in cols if c.startswith("f_")] == FEATS,
    "dtypes": {c: str(sch.field(c).type) for c in cols},
}
print(f"[schema] client file has {len(cols)} cols, "
      f"{len(out['client_schema']['f_columns'])} f_*, order_ok="
      f"{out['client_schema']['f_order_matches_meta']}", flush=True)

tsch = pq.ParquetFile(sorted(TEST.glob("*.parquet"))[0]).schema_arrow
tcols = list(tsch.names)
out["test_schema"] = {"n_columns": len(tcols), "columns": tcols,
                      "f_columns": [c for c in tcols if c.startswith("f_")],
                      "f_order_matches_meta": [c for c in tcols if c.startswith("f_")] == FEATS,
                      "dtypes": {c: str(tsch.field(c).type) for c in tcols}}
print(f"[schema] test file has {len(tcols)} cols, order_ok={out['test_schema']['f_order_matches_meta']}", flush=True)

# ---------- 3. streaming stats over ALL train rows (20-client == full train) ----------
def stream_stats(files, label_col, feat_cols, scaler=None, class_index=None):
    """Welford-free two-pass-free: sum, sumsq, min, max per column + class counts + nonfinite."""
    d = len(feat_cols)
    s   = np.zeros(d, np.float64); ss = np.zeros(d, np.float64)
    mn  = np.full(d, np.inf);      mx = np.full(d, -np.inf)
    nonfinite = np.zeros(d, np.int64)
    counts = np.zeros(C, np.int64); n = 0
    ds = pds.dataset(files, format="parquet")
    for batch in ds.to_batches(columns=feat_cols + [label_col], batch_size=262_144):
        X = np.column_stack([batch.column(c).to_numpy(zero_copy_only=False) for c in feat_cols]).astype(np.float64)
        if scaler is not None:
            X = (X - scaler[0]) / scaler[1]
        bad = ~np.isfinite(X); nonfinite += bad.sum(0)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        s += X.sum(0); ss += (X * X).sum(0)
        mn = np.minimum(mn, X.min(0)); mx = np.maximum(mx, X.max(0))
        lab = batch.column(label_col).to_pylist() if class_index else batch.column(label_col).to_numpy(zero_copy_only=False)
        if class_index:
            idx = np.fromiter((class_index[v] for v in lab), np.int64, len(lab))
        else:
            idx = np.asarray(lab, np.int64)
        counts += np.bincount(idx, minlength=C)
        n += len(batch)
    mean = s / n; var = ss / n - mean * mean
    return dict(n=int(n), mean=mean, std=np.sqrt(np.maximum(var, 0.0)),
                min=mn, max=mx, nonfinite=nonfinite, class_counts=counts)

label_col = "attack_type" if "attack_type" in cols else "label"
cls_idx = {c: i for i, c in enumerate(CLASSES)}
train_files = sorted((FL / "20_client/train").rglob("*.parquet"))
print(f"[scan] train: {len(train_files)} files ...", flush=True)
t0 = time.time()
tr = stream_stats(train_files, label_col, FEATS, None, cls_idx if label_col == "attack_type" else None)
print(f"[scan] train done in {time.time()-t0:.0f}s  n={tr['n']:,}", flush=True)

out["train_scan"] = {
    "rows": tr["n"], "label_column": label_col,
    "class_counts": {CLASSES[i]: int(tr["class_counts"][i]) for i in range(C)},
    "class_counts_match_meta": [int(v) for v in tr["class_counts"]] == META["train_counts"],
    "already_standardized": bool(np.abs(tr["mean"]).max() < 0.05 and np.abs(tr["std"] - 1).max() < 0.05),
    "abs_mean_max": float(np.abs(tr["mean"]).max()),
    "std_min": float(tr["std"].min()), "std_max": float(tr["std"].max()),
    "abs_value_max": float(np.abs(np.concatenate([tr["min"], tr["max"]])).max()),
    "fp16_safe": bool(np.abs(np.concatenate([tr["min"], tr["max"]])).max() < 65504),
    "nonfinite_total": int(tr["nonfinite"].sum()),
    "per_feature": {FEATS[i]: {"mean": float(tr["mean"][i]), "std": float(tr["std"][i]),
                               "min": float(tr["min"][i]), "max": float(tr["max"][i])}
                    for i in range(len(FEATS))},
}
print("[scan] train standardized:", out["train_scan"]["already_standardized"],
      "abs_max:", out["train_scan"]["abs_value_max"], flush=True)

# ---------- 4. test: raw AND after applying the train scaler ----------
mu  = np.array([SCAL["features"][f]["mean"] for f in FEATS], np.float64)
sd  = np.array([SCAL["features"][f]["std_used"] for f in FEATS], np.float64)
test_files = sorted(TEST.glob("*.parquet"))
tlabel = "attack_type" if "attack_type" in tcols else "label"
print(f"[scan] test raw: {len(test_files)} files ...", flush=True)
te_raw = stream_stats(test_files, tlabel, FEATS, None, cls_idx if tlabel == "attack_type" else None)
print(f"[scan] test scaled ...", flush=True)
te_sc  = stream_stats(test_files, tlabel, FEATS, (mu, sd), cls_idx if tlabel == "attack_type" else None)

out["test"] = {
    "files": len(test_files), "rows": te_raw["n"], "label_column": tlabel,
    "rows_match_meta": te_raw["n"] == META["n_test"],
    "class_counts": {CLASSES[i]: int(te_raw["class_counts"][i]) for i in range(C)},
    "class_counts_match_meta": [int(v) for v in te_raw["class_counts"]] == META["test_counts"],
    "raw_is_standardized": bool(np.abs(te_raw["mean"]).max() < 0.05 and np.abs(te_raw["std"] - 1).max() < 0.05),
    "raw_abs_mean_max": float(np.abs(te_raw["mean"]).max()),
    "raw_f_snd_spd_mean": float(te_raw["mean"][FEATS.index("f_snd_spd")]),
    "scaled_abs_mean_max": float(np.abs(te_sc["mean"]).max()),
    "scaled_std_min": float(te_sc["std"].min()), "scaled_std_max": float(te_sc["std"].max()),
    "scaled_abs_value_max": float(np.abs(np.concatenate([te_sc["min"], te_sc["max"]])).max()),
    "scaled_fp16_safe": bool(np.abs(np.concatenate([te_sc["min"], te_sc["max"]])).max() < 65504),
    "raw_nonfinite_total": int(te_raw["nonfinite"].sum()),
    "scaled_per_feature": {FEATS[i]: {"mean": float(te_sc["mean"][i]), "std": float(te_sc["std"][i]),
                                      "min": float(te_sc["min"][i]), "max": float(te_sc["max"][i])}
                           for i in range(len(FEATS))},
}
print("[scan] test rows:", te_raw["n"], "raw standardized:", out["test"]["raw_is_standardized"],
      "scaled abs_max:", out["test"]["scaled_abs_value_max"], flush=True)

# ---------- 5. per-client class presence (from sidecars, cross-checked) ----------
for n in (20, 50, 100):
    stats = json.load(open(FL / f"{n}_client" / "client_stats.json"))
    present = {c["client_id"]: sum(1 for v in c["class_counts"].values() if v > 0) for c in stats["clients"]}
    union = np.zeros(C, np.int64)
    for c in stats["clients"]:
        for k, v in c["class_counts"].items():
            union[cls_idx[k]] += v
    out["scenarios"][n].update({
        "classes_present_min": int(min(present.values())),
        "classes_present_max": int(max(present.values())),
        "clients_with_all_16": int(sum(1 for v in present.values() if v == C)),
        "union_covers_all_16": bool((union > 0).all()),
        "union_class_counts_match_meta": [int(v) for v in union] == META["train_counts"],
        "classes_present_per_client": {str(k): int(v) for k, v in sorted(present.items())},
    })

Path("/tmp/claude-1000/-home-odixe-nckh-tinyproto/96aae4cb-f1a2-4517-b930-e0e4545a4884/scratchpad/audit_result.json").write_text(json.dumps(out, indent=2))
print("AUDIT_COMPLETE", flush=True)

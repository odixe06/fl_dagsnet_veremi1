"""Decode parquet once, keep everything resident on the GPU, delete the input pipeline.

Facts this module relies on, all measured in `knowledge/DATASET.md`:

* the 66 `f_*` columns of `train/` are ALREADY z-scored (|mean| ≤ 3.0e-8, std within 3e-8 of 1);
  `test/` is NOT, and needs `scaler.json` applied exactly once;
* `label` is an int8 column already equal to the class index, so no string decode is needed;
* max |x| after scaling is 570.44 on both splits, two orders of magnitude below the fp16 ceiling
  of 65504, and there are zero non-finite values.

Each client's rows are stored as a contiguous span in one array so a client is `X[lo:hi]` and
the sampler is a single `randperm`. Clients are assigned statically to GPUs (see `plan_gpus`),
so a worker only materializes its own clients and holds roughly half the training set.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as pds
import pyarrow.parquet as pq

FEATURE_PREFIX = "f_"
LABEL_COL = "label"


# ---------------------------------------------------------------------------
# data identity — footer only, so `--require-resume` can fail in seconds
# ---------------------------------------------------------------------------

def footer_fingerprint(files: list[Path]) -> str:
    """Fast metadata fingerprint, NOT a full content hash: same-statistics rewrites can collide.
    Include relative client paths and schema; ignore the machine-specific mount prefix.
    """
    h = hashlib.sha256()
    import os
    base = Path(os.path.commonpath([str(f.parent) for f in files])) if files else Path(".")
    for f in sorted(files):
        md = pq.ParquetFile(f).metadata
        h.update(f"{f.relative_to(base)}|{f.stat().st_size}|{md.num_rows}|{md.num_row_groups}|{md.num_columns}|{md.schema.to_arrow_schema().to_string()}".encode())
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            for c in range(rg.num_columns):
                col = rg.column(c)
                st = col.statistics
                h.update(f"{col.total_compressed_size}|".encode())
                if st is not None:
                    h.update(f"{st.min}|{st.max}|{st.null_count}|".encode())
    return h.hexdigest()[:16]


def footer_rows(files: list[Path]) -> int:
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def find_client_root(candidates: list[Path], n_clients: int) -> Path:
    """Resolve the FL partition root by sentinel, never by a fixed mount prefix.

    Kaggle nests dataset mounts by kind and owner, and the nesting has changed before. Requiring
    `train/client_id=000/*.parquet` and the right client count is what stops an unrelated
    `train/` directory from winning.
    """
    found = []
    for base in candidates:
        if not base.exists():
            continue
        for p in [base, *sorted(base.rglob("*"))]:
            if not p.is_dir():
                continue
            train = p / "train"
            if not train.is_dir():
                continue
            dirs = sorted(train.glob("client_id=*"))
            if len(dirs) == n_clients and any(dirs[0].glob("*.parquet")):
                found.append(p)
    uniq = sorted({str(p) for p in found})
    if len(uniq) != 1:
        raise RuntimeError(
            f"expected exactly one {n_clients}-client partition root, found {len(uniq)}: {uniq}\n"
            f"searched: {[str(c) for c in candidates]}")
    return Path(uniq[0])


def find_test_root(candidates: list[Path]) -> Path:
    """Directory holding the fixed global test parquet shards."""
    found = []
    for base in candidates:
        if not base.exists():
            continue
        for p in [base, *sorted(base.rglob("*"))]:
            if p.is_dir() and p.name == "test" and any(p.glob("*.parquet")):
                found.append(p)
    uniq = sorted({str(p) for p in found})
    if len(uniq) != 1:
        raise RuntimeError(f"expected exactly one test/ directory, found {len(uniq)}: {uniq}")
    return Path(uniq[0])


def client_files(root: Path, cid: int) -> list[Path]:
    return sorted((root / "train" / f"client_id={cid:03d}").glob("*.parquet"))


# ---------------------------------------------------------------------------
# static GPU assignment
# ---------------------------------------------------------------------------

def plan_gpus(rows_per_client: dict[int, int], world: int) -> list[list[int]]:
    """Longest-processing-time partition of clients across GPUs.

    Client sizes are known before the run, so a static split is both balanced and much cheaper
    than dynamic scheduling: a worker then owns its clients' optimizer states and training rows
    permanently, which removes those from the inter-process queue entirely. Measured imbalance
    on this partition: 1.96% (20 clients), 0.03% (50), 0.11% (100).
    """
    bins: list[list[int]] = [[] for _ in range(world)]
    load = [0] * world
    for cid, n in sorted(rows_per_client.items(), key=lambda kv: -kv[1]):
        k = min(range(world), key=lambda i: load[i])
        bins[k].append(cid)
        load[k] += n
    return [sorted(b) for b in bins]


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def _feature_order(schema_names: list[str], expected: list[str]) -> list[str]:
    got = [c for c in schema_names if c.startswith(FEATURE_PREFIX)]
    if got != expected:
        raise RuntimeError(
            "feature column order does not match meta.json.\n"
            f"  first mismatch at index {next(i for i, (a, b) in enumerate(zip(got, expected)) if a != b)}"
            if len(got) == len(expected) else f"  got {len(got)} columns, expected {len(expected)}")
    return got


def decode_clients(root: Path, cids: list[int], features: list[str],
                   *, batch_rows: int = 1_048_576, progress=print):
    """Decode the given clients into one fp16 feature array plus int8 labels.

    Returns (X (N, 66) float16, y (N,) int8, spans {cid: (lo, hi)}, audit dict).
    Audit statistics are accumulated on the float32 values BEFORE the fp16 cast, so the reported
    maximum is the true one and not a saturated one.
    """
    spans, total = {}, 0
    per_client_files = {}
    for cid in cids:
        fs = client_files(root, cid)
        if not fs:
            raise RuntimeError(f"client {cid} has no parquet files under {root}")
        n = footer_rows(fs)
        per_client_files[cid] = fs
        spans[cid] = (total, total + n)
        total += n

    X = np.empty((total, len(features)), dtype=np.float16)
    y = np.empty(total, dtype=np.int8)
    amax = 0.0
    nonfinite = 0
    counts = np.zeros(16, dtype=np.int64)
    t0 = time.time()

    for k, cid in enumerate(cids):
        lo, hi = spans[cid]
        ds = pds.dataset(per_client_files[cid], format="parquet")
        _feature_order(list(ds.schema.names), features)
        off = lo
        for b in ds.to_batches(columns=features + [LABEL_COL], batch_size=batch_rows):
            block = np.column_stack([b.column(c).to_numpy(zero_copy_only=False)
                                     for c in features]).astype(np.float32, copy=False)
            nonfinite += int((~np.isfinite(block)).sum())
            amax = max(amax, float(np.abs(block).max()))
            n = block.shape[0]
            X[off:off + n] = block.astype(np.float16)
            lab = np.asarray(b.column(LABEL_COL).to_numpy(zero_copy_only=False), dtype=np.int8)
            y[off:off + n] = lab
            counts += np.bincount(lab.astype(np.int64), minlength=16)
            off += n
        if off != hi:
            raise RuntimeError(f"client {cid}: decoded {off - lo} rows, footer said {hi - lo}")
        if progress and (k + 1) % max(1, len(cids) // 5) == 0:
            progress(f"    decoded {k + 1}/{len(cids)} clients, {off:,} rows, {time.time() - t0:.0f}s")

    audit = {"rows": int(total), "abs_max": amax, "nonfinite": int(nonfinite),
             "fp16_safe": bool(amax < 65504), "class_counts": counts.tolist(),
             "seconds": round(time.time() - t0, 1)}
    if nonfinite:
        raise RuntimeError(f"{nonfinite} non-finite feature values in train; the audit says there are none")
    if not audit["fp16_safe"]:
        raise RuntimeError(f"max |x| = {amax} exceeds the fp16 range; do not store features as fp16")
    return X, y, spans, audit


def decode_test(test_dir: Path, features: list[str], scaler: dict,
                *, batch_rows: int = 1_048_576, progress=print):
    """Decode the fixed global test set and apply the TRAIN-fitted scaler exactly once.

    `test/` ships unstandardized. Skipping this step raises no error -- the data is finite and
    looks clean -- it just silently scores the model on the wrong scale. The audit returns the
    raw mean of `f_snd_spd`, which is ~7.45 before scaling and ~0 after; assert on it.
    """
    files = sorted(test_dir.glob("*.parquet"))
    total = footer_rows(files)
    mu = np.array([scaler["features"][f]["mean"] for f in features], dtype=np.float32)
    sd = np.array([scaler["features"][f]["std_used"] for f in features], dtype=np.float32)

    X = np.empty((total, len(features)), dtype=np.float16)
    y = np.empty(total, dtype=np.int8)
    off, amax = 0, 0.0
    raw_sum_snd_spd, counts = 0.0, np.zeros(16, dtype=np.int64)
    j_snd = features.index("f_snd_spd")
    t0 = time.time()

    ds = pds.dataset(files, format="parquet")
    _feature_order(list(ds.schema.names), features)
    for b in ds.to_batches(columns=features + [LABEL_COL], batch_size=batch_rows):
        raw = np.column_stack([b.column(c).to_numpy(zero_copy_only=False)
                               for c in features]).astype(np.float32, copy=False)
        if not np.isfinite(raw).all():
            raise RuntimeError("non-finite feature in test")
        raw_sum_snd_spd += float(raw[:, j_snd].sum(dtype=np.float64))
        block = (raw - mu) / sd
        if not np.isfinite(block).all():
            raise RuntimeError("non-finite scaled test feature; check scaler")
        amax = max(amax, float(np.abs(block).max()))
        n = block.shape[0]
        X[off:off + n] = block.astype(np.float16)
        lab = np.asarray(b.column(LABEL_COL).to_numpy(zero_copy_only=False), dtype=np.int8)
        y[off:off + n] = lab
        counts += np.bincount(lab.astype(np.int64), minlength=16)
        off += n
    if off != total:
        raise RuntimeError(f"test: decoded {off} rows, footer said {total}")

    audit = {"rows": int(total), "abs_max": amax, "fp16_safe": bool(amax < 65504),
             "raw_mean_f_snd_spd": raw_sum_snd_spd / total,
             "class_counts": counts.tolist(), "seconds": round(time.time() - t0, 1)}
    if not (6.5 < audit["raw_mean_f_snd_spd"] < 8.5):
        raise RuntimeError(
            f"raw mean of f_snd_spd is {audit['raw_mean_f_snd_spd']:.4f}; expected ~7.45. "
            "Either test/ was already standardized upstream or the column order is wrong.")
    if not audit["fp16_safe"]:
        raise RuntimeError(f"max |x| = {amax} exceeds the fp16 range after scaling")
    return X, y, audit


# ---------------------------------------------------------------------------
# validation holdout — used ONLY by the mu grid search
# ---------------------------------------------------------------------------

def stratified_holdout(y: np.ndarray, spans: dict[int, tuple[int, int]],
                       frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split each client's rows into (train_idx, val_idx), stratified by class.

    The mu grid search must not read the test set, so it selects on a validation slice carved
    out of TRAIN. The production runs use 100% of train and never call this. Returns global row
    indices into the arrays produced by `decode_clients`.
    """
    if not 0 < frac < 1:
        raise ValueError("holdout fraction must be strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    tr_parts, va_parts = [], []
    for cid in sorted(spans):
        lo, hi = spans[cid]
        yl = y[lo:hi]
        for c in np.unique(yl):
            idx = lo + np.flatnonzero(yl == c)
            rng.shuffle(idx)
            k = int(math.floor(len(idx) * frac))
            if len(idx) >= 2:
                k = max(1, k)                 # never leave a present class unrepresented
            va_parts.append(idx[:k])
            tr_parts.append(idx[k:])
    tr = np.sort(np.concatenate(tr_parts))
    va = np.sort(np.concatenate(va_parts))
    assert len(tr) + len(va) == len(y) and len(np.intersect1d(tr, va)) == 0
    return tr, va


def worker_train_indices(local_indices: dict, spans: dict) -> dict:
    """Translate client-relative holdout indices into this worker's packed row offsets."""
    result = {}
    for cid, (lo, hi) in spans.items():
        idx = np.asarray(local_indices[cid])
        if (idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or len(idx) == 0
                or np.any(idx < 0) or np.any(idx >= hi - lo)
                or len(np.unique(idx)) != len(idx)):
            raise ValueError(f"invalid client-relative train indices for client {cid}")
        result[cid] = idx.astype(np.int64) + lo
    return result


def load_meta(path: Path) -> dict:
    m = json.loads(Path(path).read_text())
    assert len(m["feature_cols"]) == 66 and len(m["class_names"]) == m["num_classes"] == 16
    return m

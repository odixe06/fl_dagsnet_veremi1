"""Data loading for AFPHA-DAGSNet.

Contract (see .agents/skills/kaggle-training-notebook/references/veremi-dataset-layout.md):

* Only the 66 `f_*` columns enter the model, in the frozen order from `meta.json`.
* Train partitions are ALREADY z-scored -- never standardize them a second time.
* Test is raw and must be scaled with the same train-fitted `scaler.json`.
* Roots are resolved by a unique sentinel file, not by a fixed mount prefix, so the
  same code runs against the local tree and the Kaggle mount.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

FP16_MAX = 65504.0


# --------------------------------------------------------------------------- paths

def find_root(sentinel, search_roots, max_depth=5):
    """Locate the unique directory containing `sentinel` under any of `search_roots`.

    `sentinel` is a path relative to the root, e.g. "20_client/client_stats.json".
    Descends up to `max_depth` levels because a Kaggle mount adds an unpredictable
    prefix. Two shapes have been seen and both must resolve: the flat
    `/kaggle/input/<dataset>/upload/test/` and the owner-prefixed
    `/kaggle/input/datasets/<owner>/<dataset>/upload/test/`, which is one level deeper.
    Raises unless exactly one root matches: an ambiguous mount must fail loudly, not
    train on the wrong partition.
    """
    hits, seen = [], set()

    def walk(base, depth):
        base = Path(base)
        if not base.is_dir() or base in seen:
            return
        seen.add(base)
        if (base / sentinel).exists():
            hits.append(base)
            return                       # do not descend past a match
        if depth == 0:
            return
        for child in sorted(base.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                walk(child, depth - 1)

    for root in search_roots:
        walk(root, max_depth)
    hits = sorted({h.resolve() for h in hits})
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one root containing {sentinel!r} under {list(search_roots)}, "
            f"found {len(hits)}: {hits}")
    return hits[0]


# --------------------------------------------------------------------- frozen schema

class Schema:
    """Feature order, class names and the train-fitted scaler, loaded once."""

    def __init__(self, meta_path, scaler_path, label_mapping_path=None):
        meta = json.loads(Path(meta_path).read_text())
        self.features = list(meta["feature_cols"])
        self.class_names = list(meta["class_names"])
        self.num_classes = int(meta["num_classes"])
        assert len(self.features) == 66, f"expected 66 features, got {len(self.features)}"
        assert len(self.class_names) == self.num_classes

        scaler = json.loads(Path(scaler_path).read_text())["features"]
        self.mean = np.array([scaler[f]["mean"] for f in self.features], dtype=np.float64)
        self.std = np.array([scaler[f]["std_used"] for f in self.features], dtype=np.float64)
        self.fill = np.array([scaler[f].get("fill_value", 0.0) for f in self.features],
                             dtype=np.float64)
        assert (self.std > 0).all(), "scaler has a non-positive std_used"

        if label_mapping_path is not None:
            lm = json.loads(Path(label_mapping_path).read_text())
            assert lm["classes"] == self.class_names, "label_mapping disagrees with meta.json"

    def fingerprint(self):
        payload = json.dumps({"features": self.features, "classes": self.class_names,
                              "mean": self.mean.tolist(), "std": self.std.tolist()},
                             sort_keys=True)
        import hashlib
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ------------------------------------------------------------------------- reading

def _read_files(files, schema, out_X, out_y, row0, scale, log=None):
    """Stream `files` into preallocated arrays starting at `row0`; return rows written."""
    cols = schema.features + ["label"]
    n = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=262_144, columns=cols, use_threads=True):
            m = batch.num_rows
            lo = row0 + n
            for j, name in enumerate(schema.features):
                v = batch.column(name).to_numpy(zero_copy_only=False)
                if scale:
                    v = np.nan_to_num(v.astype(np.float64), nan=np.nan)
                    bad = ~np.isfinite(v)
                    if bad.any():
                        v[bad] = schema.fill[j]
                    v = (v - schema.mean[j]) / schema.std[j]
                out_X[lo:lo + m, j] = v            # numpy downcasts to the array dtype
            out_y[lo:lo + m] = batch.column("label").to_numpy(zero_copy_only=False)
            n += m
        if log:
            log(f"    read {Path(path).name}: {n:,} rows so far")
    return n


def _check_fp16(X, tag):
    """A z-scored feature stored as fp16 must not overflow; report the observed extreme."""
    peak = float(np.abs(X[:: max(1, len(X) // 2_000_000)]).max())
    assert np.isfinite(X[:: max(1, len(X) // 2_000_000)]).all(), f"{tag}: non-finite after cast"
    assert peak < FP16_MAX, f"{tag}: |x|max {peak} exceeds fp16 range"
    return peak


def _file_identity(path):
    """Content identity of one parquet file, from its footer only.

    Row counts do not identify data: a file can be rewritten with different features and
    permuted labels while keeping every count intact, and a fingerprint built on counts
    alone then declares the new data identical to the old -- reusing a stale prepack cache
    and permitting a resume onto different mathematics.

    The footer carries enough to tell those apart cheaply: the byte size of the file, the
    row count, and per row group per column the compressed size plus the min/max/null
    statistics. Rewriting values moves at least one of those in practice, and reading them
    costs a footer seek rather than a column decode -- which matters, because this runs
    before the multi-minute prepack and `--require-resume` has to fail in seconds.
    """
    md = pq.ParquetFile(path).metadata
    parts = [path.name, str(path.stat().st_size), str(md.num_rows),
             str(md.num_row_groups), str(md.num_columns)]
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for c in range(rg.num_columns):
            col = rg.column(c)
            st = col.statistics
            parts += [str(col.total_compressed_size),
                      "" if st is None else f"{st.min}|{st.max}|{st.null_count}"]
    return "\x1f".join(parts)


def _digest(items):
    h = hashlib.sha256()
    for it in items:
        h.update(it.encode())
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def scan_counts(train_root, test_root):
    """Row counts and content digests from parquet footers only -- no column decode.

    The fingerprint has to cover the data identity, and `--require-resume` has to fail
    before the multi-minute prepack, so both need these numbers cheaply.

    Returns (rows, n_test, digests) with digests = {"train": ..., "test": ...}.
    """
    train_root, test_root = Path(train_root), Path(test_root)
    client_dirs = sorted(d for d in train_root.iterdir()
                         if d.is_dir() and d.name.startswith("client_id="))
    assert client_dirs, f"no client_id=* directories under {train_root}"
    rows, train_ids = [], []
    for d in client_dirs:
        files = sorted(d.glob("part-*.parquet"))
        assert files, f"no parquet parts in {d}"
        rows.append(sum(pq.ParquetFile(f).metadata.num_rows for f in files))
        train_ids.append(d.name)
        train_ids += [_file_identity(f) for f in files]
    test_files = sorted(test_root.glob("part-*.parquet"))
    assert test_files, f"no parquet parts in {test_root}"
    n_test = sum(pq.ParquetFile(f).metadata.num_rows for f in test_files)
    digests = {"train": _digest(train_ids),
               "test": _digest(_file_identity(f) for f in test_files)}
    return np.asarray(rows, dtype=np.int64), int(n_test), digests


def load_clients(train_root, schema, log=print):
    """Read every `client_id=NNN` partition into one contiguous fp16 matrix.

    Returns (X, y, spans, rows) where `spans[i] = (lo, hi)` addresses client i's
    rows and clients appear in ascending client_id order.
    """
    train_root = Path(train_root)
    client_dirs = sorted(d for d in train_root.iterdir()
                         if d.is_dir() and d.name.startswith("client_id="))
    assert client_dirs, f"no client_id=* directories under {train_root}"

    per_client_files, rows = [], []
    for d in client_dirs:
        files = sorted(d.glob("part-*.parquet"))
        assert files, f"no parquet parts in {d}"
        per_client_files.append(files)
        rows.append(sum(pq.ParquetFile(f).metadata.num_rows for f in files))
    total = int(sum(rows))
    log(f"  {len(client_dirs)} clients, {total:,} rows, "
        f"{total * 66 * 2 / 2**30:.2f} GiB as fp16")

    X = np.empty((total, 66), dtype=np.float16)
    y = np.empty(total, dtype=np.uint8)
    spans, cursor, t0 = [], 0, time.perf_counter()
    for i, files in enumerate(per_client_files):
        got = _read_files(files, schema, X, y, cursor, scale=False)
        assert got == rows[i], f"client {i}: expected {rows[i]} rows, read {got}"
        spans.append((cursor, cursor + got))
        cursor += got
        if (i + 1) % 10 == 0 or i == len(per_client_files) - 1:
            log(f"  client {i + 1}/{len(per_client_files)}  "
                f"{cursor:,} rows  {time.perf_counter() - t0:.0f}s")
    assert cursor == total
    peak = _check_fp16(X, "train")
    assert y.max() < schema.num_classes, "train label out of range"
    log(f"  train ready in {time.perf_counter() - t0:.0f}s, |x|max~{peak:.1f}, "
        f"classes present {len(np.unique(y))}/{schema.num_classes}")
    return X, y, spans, np.asarray(rows, dtype=np.int64)


def load_test(test_root, schema, log=print):
    """Read the fixed test set and apply the train-fitted scaler."""
    test_root = Path(test_root)
    files = sorted(test_root.glob("part-*.parquet"))
    assert files, f"no parquet parts in {test_root}"
    total = int(sum(pq.ParquetFile(f).metadata.num_rows for f in files))
    log(f"  test {total:,} rows from {len(files)} files")
    X = np.empty((total, 66), dtype=np.float16)
    y = np.empty(total, dtype=np.uint8)
    t0 = time.perf_counter()
    got = _read_files(files, schema, X, y, 0, scale=True)
    assert got == total
    peak = _check_fp16(X, "test")
    assert set(np.unique(y).tolist()) == set(range(schema.num_classes)), \
        "test does not contain all 16 classes"
    log(f"  test ready in {time.perf_counter() - t0:.0f}s, |x|max~{peak:.1f}")
    return X, y


def steps_per_round(rows, batch):
    """Optimizer steps a full round costs; the tail batch is a real step."""
    return int(sum(math.ceil(int(r) / batch) for r in rows))

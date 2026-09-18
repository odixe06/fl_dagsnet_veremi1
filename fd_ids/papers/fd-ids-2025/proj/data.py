"""Parquet -> resident fp16 tensors. One pass per session, then no input pipeline at all.

Two facts from knowledge/DATASET.md that produce silent corruption if ignored:
  * train/ is ALREADY z-scored; test/ is NOT. Applying scaler.json to train a second time
    destroys it and raises nothing.
  * the integer label is in column `label` (int8, 0..15). Decoding `attack_type` strings
    over 43 M rows costs tens of seconds per pass for the same information.
"""
import json
from pathlib import Path
import numpy as np
import pyarrow.dataset as ds

CACHE_FILES = ("train_X.f16.npy", "train_y.u8.npy", "test_X.f16.npy",
               "test_y.u8.npy", "spans.json")


def find_root(sentinel, bases=("/kaggle/input",)):
    """Kaggle mounts are nested by kind and owner; the prefix is not /kaggle/input/<slug>/.
    Resolve by locating the sentinel instead of hard-coding a depth."""
    depth = len(Path(sentinel).parts)          # strip the whole sentinel, not one level
    hits = []
    for b in bases:
        p = Path(b)
        if p.exists():
            for q in p.rglob(sentinel):
                if q.is_dir():
                    r = q
                    for _ in range(depth): r = r.parent
                    hits.append(r)
    hits = sorted(set(hits))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {sentinel!r} under {bases}, got {hits}")
    return hits[0]


def parquet_files(root):
    """The parquet parts of a directory, in a fixed order, and NOTHING else.

    `ds.dataset(dir, format="parquet")` opens every file it finds. The centralized test
    directory ships a `part-NNNNN.stats.json` sidecar next to each part, so `to_table()`
    died on Kaggle with "Parquet magic bytes not found in footer" after a nine-minute train
    decode. It survived locally only because the smoke fixture reads through `to_batches()`
    and breaks early, never reaching a sidecar -- a lazy reader hides exactly this.

    sorted() is not cosmetic either: the fragment order fixes the row order of the test set,
    and therefore the order of y_true and of every saved prediction vector."""
    files = sorted(str(f) for f in Path(root).rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"no .parquet files under {root}")
    return files


def _labels(col, num_classes=16):
    """astype(np.uint8) on a label of -1 gives 255 and on 300 gives 44 -- both are silent,
    and both survive every downstream assert because the counts still add up."""
    v = col.to_numpy(zero_copy_only=False)
    lo, hi = int(v.min()), int(v.max())
    if lo < 0 or hi >= num_classes:
        raise RuntimeError(f"labels out of range [{lo}, {hi}], expected 0..{num_classes-1}")
    return v.astype(np.uint8)


def load_clients(fl_root, feature_cols, n_clients, dtype=np.float16):
    """Returns X (N,66) fp16, y (N,) uint8, and spans[cid] = (lo, hi) contiguous row range.

    Contiguous spans are what make the training loop a slice + randperm instead of a
    gather over a client-id column."""
    dirs = sorted((fl_root / "train").glob("client_id=*"),
                  key=lambda p: int(p.name.split("=")[1]))
    if len(dirs) != n_clients:
        raise RuntimeError(f"expected {n_clients} client dirs, found {len(dirs)}")
    cols = list(feature_cols) + ["label"]
    xs, ys, spans, off = [], [], {}, 0
    for d in dirs:
        cid = int(d.name.split("=")[1])
        t = ds.dataset(parquet_files(d), format="parquet").to_table(columns=cols)
        n = t.num_rows
        a = np.empty((n, len(feature_cols)), dtype=dtype)
        for j, c in enumerate(feature_cols):
            a[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(dtype, copy=False)
        xs.append(a)
        ys.append(_labels(t.column("label")))
        spans[cid] = (off, off + n); off += n
        del t
    return np.concatenate(xs), np.concatenate(ys), spans


def load_test(test_root, feature_cols, scaler, dtype=np.float16):
    """test/ is raw: apply scaler.json here, and nowhere else."""
    t = ds.dataset(parquet_files(test_root), format="parquet").to_table(
        columns=list(feature_cols) + ["label"])
    n = t.num_rows
    X = np.empty((n, len(feature_cols)), dtype=dtype)
    for j, c in enumerate(feature_cols):
        v = t.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        s = scaler[c]
        X[:, j] = ((v - s["mean"]) / s["std_used"]).astype(dtype)
    y = _labels(t.column("label"))
    return X, y


def assert_fp16_safe(X, name):
    """knowledge/DATASET.md measured max|x| = 570.44 on both splits, two orders below the
    fp16 ceiling. Assert it rather than inherit the assumption."""
    m = float(np.abs(X).max())
    if not np.isfinite(m) or m >= 65504:
        raise RuntimeError(f"{name}: max|x| = {m} is not fp16-safe")
    return m


def cache_ok(cache, want, n_clients, n_features=66):
    """Is the prepack cache complete, current and self-consistent?

    A matching manifest is a claim, not evidence. Running the notebook's own hit branch with
    a matching manifest and no train_X printed "cache reusable"; the worker then died opening
    a file that had never been written. Headers are read through mmap, so this costs a few
    stat calls and no data."""
    cache = Path(cache)
    mf = cache / "manifest.json"
    if not mf.is_file():
        return False
    try:
        if json.loads(mf.read_text()) != want:
            return False
        for f in CACHE_FILES:
            if not (cache / f).is_file() or (cache / f).stat().st_size == 0:
                return False
        spans = {int(k): tuple(v)
                 for k, v in json.load(open(cache / "spans.json")).items()}
        X = np.load(cache / "train_X.f16.npy", mmap_mode="r")
        Y = np.load(cache / "train_y.u8.npy", mmap_mode="r")
        TX = np.load(cache / "test_X.f16.npy", mmap_mode="r")
        TY = np.load(cache / "test_y.u8.npy", mmap_mode="r")
    except Exception as e:
        print("[cache] unreadable:", e)
        return False
    n = sum(hi - lo for lo, hi in spans.values())
    r = sorted(spans.values())
    return bool(
        len(spans) == n_clients
        and X.dtype == np.float16 and Y.dtype == np.uint8
        and TX.dtype == np.float16 and TY.dtype == np.uint8
        and X.shape == (n, n_features) and Y.shape == (n,)
        and TX.ndim == 2 and TX.shape[1] == n_features and len(TX) == len(TY)
        and r and r[0][0] == 0 and r[-1][1] == n
        and all(a[1] == b[0] for a, b in zip(r, r[1:])))

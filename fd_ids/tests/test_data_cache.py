"""The two production loaders and the prepack cache guard, on a real parquet fixture.

The smoke test builds its arrays with code written inside the test, so it never touched
load_clients/load_test. These are the functions that decide whether the training data is
scaled once, twice or not at all, and whether a label of -1 quietly becomes class 255.
"""
import json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/fd-ids-2025"))
from proj.data import (find_root, load_clients, load_test, cache_ok,
                       assert_fp16_safe, parquet_files)

ROOT = Path(__file__).resolve().parents[1]
FEATS = json.loads((ROOT / "knowledge/meta.json").read_text())["feature_cols"]
CLIENTS = {0: 40, 1: 25, 2: 35}          # deliberately unequal
NT = 50


def write_parquet(path, n, seed, labels=None, order=None, scale=1.0, offset=0.0):
    rng = np.random.default_rng(seed)
    cols = order or FEATS
    data = {c: (rng.standard_normal(n) * scale + offset).astype(np.float64) for c in cols}
    data["label"] = (rng.integers(0, 16, n) if labels is None
                     else np.full(n, labels)).astype(np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(data), path)


def build(tmp, labels=None, order=None):
    fl = tmp / "fl"
    for cid, n in CLIENTS.items():
        write_parquet(fl / "train" / f"client_id={cid:03d}" / "p.parquet", n, 100 + cid,
                      labels=labels, order=order)
    cen = tmp / "cen"
    # test is RAW: give it a large offset so double-scaling is impossible to miss
    write_parquet(cen / "upload" / "test" / "p.parquet", NT, 7, offset=1000.0, scale=5.0)
    scaler = {"features": {c: {"mean": 1000.0, "std_used": 5.0} for c in FEATS}}
    (cen / "upload" / "scaler.json").write_text(json.dumps(scaler))
    return fl, cen, scaler["features"]


def main():
    tmp = Path(tempfile.mkdtemp())
    fl, cen, scaler = build(tmp)

    # 1. find_root strips the whole sentinel, not one level
    assert find_root("train/client_id=000", bases=(str(tmp),)) == fl, "sentinel depth"
    assert find_root("upload/test", bases=(str(tmp),)) == cen
    print("   find_root resolves both sentinels to the right roots")

    # 2. an ambiguous sentinel must refuse rather than pick one
    dup = tmp / "fl2"
    shutil.copytree(fl, dup)
    try:
        find_root("train/client_id=000", bases=(str(tmp),))
        raise AssertionError("two candidate roots were silently reduced to one")
    except RuntimeError as e:
        assert "got" in str(e)
    shutil.rmtree(dup)
    print("   two matching roots -> refuses instead of guessing")

    # 3. load_clients: contiguous spans, right dtypes, and NO rescaling of train
    X, Y, spans = load_clients(fl, FEATS, len(CLIENTS))
    n = sum(CLIENTS.values())
    assert X.shape == (n, 66) and X.dtype == np.float16, (X.shape, X.dtype)
    assert Y.shape == (n,) and Y.dtype == np.uint8
    assert [spans[c] for c in sorted(spans)] == [(0, 40), (40, 65), (65, 100)], spans
    raw = pq.read_table(fl / "train/client_id=000/p.parquet").column(FEATS[0]).to_numpy()
    assert np.allclose(X[:40, 0].astype(np.float64), raw.astype(np.float16), atol=0), \
        "train was transformed on the way in; it is already standardized at source"
    print(f"   load_clients: {X.shape} fp16, spans contiguous, train untouched")

    # 4. load_test applies the scaler exactly once, here and nowhere else
    TX, TY = load_test(cen / "upload" / "test", FEATS, scaler)
    raw_t = pq.read_table(cen / "upload/test/p.parquet").column(FEATS[0]).to_numpy()
    want = ((raw_t - 1000.0) / 5.0).astype(np.float16)
    assert np.array_equal(TX[:, 0], want), "test scaling is not (x - mean) / std_used"
    assert abs(float(TX.mean())) < 1.0, "test still looks unscaled"
    print(f"   load_test: z-scored exactly once, mean {float(TX.mean()):+.3f}")

    # 4b. sidecars. The real centralized test directory ships part-NNNNN.stats.json next to
    #     every part, and ds.dataset(dir) opens whatever it finds: on Kaggle that ended a
    #     nine-minute decode with "Parquet magic bytes not found in footer". It passed here
    #     only because the smoke fixture reads lazily and breaks before reaching one.
    tdir = cen / "upload" / "test"
    for part in sorted(tdir.glob("*.parquet")):
        part.with_suffix(".stats.json").write_text('{"rows": 0}')
    (tdir / "_SUCCESS").write_text("")
    files = parquet_files(tdir)
    assert files == [str(f) for f in sorted(tdir.glob("*.parquet"))], files
    TX2, TY2 = load_test(tdir, FEATS, scaler)
    assert np.array_equal(TX2, TX) and np.array_equal(TY2, TY), \
        "sidecars changed the decoded test set"
    print(f"   {len(list(tdir.iterdir()))} files in test/, "
          f"{len(files)} parquet parts read, sidecars ignored")

    # 4c. and against the real dataset on this machine, which has the sidecars
    real = Path("/home/odixe/nckh/dataset/centralized/test")
    if real.is_dir():
        rf = parquet_files(real)
        assert all(f.endswith(".parquet") for f in rf) and rf == sorted(rf)
        assert len(rf) < len(list(real.iterdir())), "fixture assumption: real dir has extras"
        print(f"   real test dir: {len(list(real.iterdir()))} files -> {len(rf)} parquet parts")

    # 5. feature ORDER comes from the list, not from the file
    tmp2 = Path(tempfile.mkdtemp())
    fl2, _, _ = build(tmp2, order=list(reversed(FEATS)))
    X2, _, _ = load_clients(fl2, FEATS, len(CLIENTS))
    a = pq.read_table(fl2 / "train/client_id=000/p.parquet")
    assert a.column_names[0] != FEATS[0], "fixture did not actually reorder"
    assert np.array_equal(X2[:40, 0], a.column(FEATS[0]).to_numpy().astype(np.float16)), \
        "columns were taken positionally instead of by name"
    shutil.rmtree(tmp2)
    print("   reordered parquet columns -> still selected by name")

    # 6. labels outside 0..15 must raise, not wrap. astype(uint8) turns -1 into 255.
    for bad in (-1, 16):
        t3 = Path(tempfile.mkdtemp())
        build(t3, labels=bad)
        try:
            load_clients(t3 / "fl", FEATS, len(CLIENTS))
            raise AssertionError(f"label {bad} accepted")
        except RuntimeError as e:
            assert "out of range" in str(e), e
        shutil.rmtree(t3)
    print("   labels of -1 and 16 both rejected before the uint8 cast")

    # 7. wrong client count
    try:
        load_clients(fl, FEATS, 99)
        raise AssertionError("client count not checked")
    except RuntimeError as e:
        assert "client dirs" in str(e)

    # 8. the cache guard: a matching manifest with a missing or broken file is not a cache
    cache = tmp / "cache"; cache.mkdir()
    want_mf = {"data_id": "x", "n_clients": 3}
    np.save(cache / "train_X.f16.npy", X); np.save(cache / "train_y.u8.npy", Y)
    np.save(cache / "test_X.f16.npy", TX); np.save(cache / "test_y.u8.npy", TY)
    json.dump({str(k): list(v) for k, v in spans.items()}, open(cache / "spans.json", "w"))
    (cache / "manifest.json").write_text(json.dumps(want_mf))
    assert cache_ok(cache, want_mf, 3), "a complete cache was rejected"
    print("   complete cache accepted")

    for name, break_it in (
            ("train_X missing", lambda: (cache / "train_X.f16.npy").unlink()),
            ("train_X truncated", lambda: (cache / "train_X.f16.npy").write_bytes(b"")),
            ("train_X is garbage", lambda: (cache / "train_X.f16.npy").write_bytes(b"x" * 999)),
            ("y has fewer rows than X",
             lambda: np.save(cache / "train_y.u8.npy", Y[:-1])),
            ("wrong dtype for X",
             lambda: np.save(cache / "train_X.f16.npy", X.astype(np.float32))),
            ("spans leave a gap",
             lambda: json.dump({"0": [0, 40], "1": [41, 65], "2": [65, 100]},
                               open(cache / "spans.json", "w"))),
            ("manifest describes another scenario",
             lambda: (cache / "manifest.json").write_text(json.dumps({"data_id": "other",
                                                                     "n_clients": 3}))),
            ("manifest missing", lambda: (cache / "manifest.json").unlink())):
        backup = {f: (cache / f).read_bytes() for f in
                  ("train_X.f16.npy", "train_y.u8.npy", "spans.json", "manifest.json")
                  if (cache / f).is_file()}
        break_it()
        assert not cache_ok(cache, want_mf, 3), f"cache guard accepted: {name}"
        print(f"   rejected: {name}")
        for f, b in backup.items(): (cache / f).write_bytes(b)

    assert assert_fp16_safe(X, "train") < 65504
    shutil.rmtree(tmp)
    print("\nALL 8 DATA/CACHE CHECKS PASSED")


if __name__ == "__main__":
    main()

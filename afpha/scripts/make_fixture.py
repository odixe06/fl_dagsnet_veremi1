#!/usr/bin/env python
"""Build small fixtures for the end-to-end smoke tests.

Usage: conda run -n nckh python scripts/make_fixture.py <dir> [--clients 20 50 100]
                                                              [--rows 1200] [--nested]

Real schema and real values, so the whole driver -- parquet decode, scaler, resident
tensors, workers, aggregation, artifacts -- runs on a small laptop GPU.

Memory: never call `pq.read_table` on a source part. A single VeReMi part can be a
gigabyte once decoded, and this box has 8 GB of RAM shared with the training test that
follows. Everything here streams record batches and keeps at most a few of them alive.

`--nested` also writes the owner-prefixed mount shape Kaggle sometimes produces
(`datasets/<owner>/<dataset>/...`) so `fldata.find_root` is tested against both.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FL_ROOT = Path("/home/odixe/nckh/dataset/fl_client/alpha05")
TEST_PART = Path("/home/odixe/nckh/dataset/centralized/test")
OWNER = "odixe0502"
BATCH = 16_384


SPREAD = 8          # number of record batches a sample is drawn from


def _sample_batches(path, want, rng, label_col=None, per_class=None, n_classes=16):
    """Collect ~`want` rows spread across a part file, one record batch at a time.

    Parts are time-ordered, so a head slice yields one or two classes and makes the
    smoke test degenerate; striding across batches keeps the class mix. The stride is
    derived from the file's own size -- a fixed stride skips nearly every batch of a
    small partition and silently returns a fraction of the rows asked for.
    """
    pf = pq.ParquetFile(path)
    n_batches = max(1, -(-pf.metadata.num_rows // BATCH))
    # Striding is for spreading a plain sample over a time-ordered file. The stratified
    # branch must not stride: rare classes live in a few batches, and skipping 7 of
    # every 8 drops them from the fixture entirely.
    stride = 1 if per_class is not None else max(1, n_batches // SPREAD)
    chunks = min(SPREAD, n_batches)
    kept, counts = [], {}
    for i, batch in enumerate(pf.iter_batches(batch_size=BATCH)):
        if i % stride:
            continue
        if per_class is None:
            take = min(-(-want // chunks), batch.num_rows)
            idx = np.sort(rng.choice(batch.num_rows, size=take, replace=False))
            kept.append(batch.take(pa.array(idx)))
            if sum(b.num_rows for b in kept) >= want:
                break
        else:
            lab = batch.column(label_col).to_numpy(zero_copy_only=False)
            pick = []
            for c in np.unique(lab):
                need = per_class - counts.get(int(c), 0)
                if need <= 0:
                    continue
                where = np.flatnonzero(lab == c)
                sel = rng.choice(where, size=min(need, len(where)), replace=False)
                counts[int(c)] = counts.get(int(c), 0) + len(sel)
                pick.append(sel)
            if pick:
                idx = np.sort(np.concatenate(pick))
                kept.append(batch.take(pa.array(idx)))
            if len(counts) >= n_classes and min(counts.values()) >= per_class:
                break
    assert kept, f"no rows sampled from {path}"
    table = pa.Table.from_batches([b for b in kept], schema=pf.schema_arrow)
    if per_class is None and table.num_rows > want:
        idx = np.sort(rng.choice(table.num_rows, size=want, replace=False))
        table = table.take(pa.array(idx))
    return table


def build_scenario(fix, n, rows_per_client, rng):
    src = FL_ROOT / f"{n}_client"
    dst = fix / "fl" / f"{n}_client"
    (dst / "train").mkdir(parents=True)
    client_dirs = sorted(p for p in (src / "train").iterdir()
                         if p.name.startswith("client_id="))
    got = 0
    for d in client_dirs:
        part = sorted(d.glob("part-*.parquet"))[0]
        table = _sample_batches(part, rows_per_client, rng)
        out = dst / "train" / d.name
        out.mkdir()
        pq.write_table(table, out / "part-00000.parquet")
        got = table.num_rows
        del table
    shutil.copy(src / "client_stats.json", dst / "client_stats.json")
    return len(client_dirs), got


def build_test(fix, test_rows, rng, n_classes=16):
    tp = sorted(TEST_PART.glob("part-*.parquet"))[0]
    table = _sample_batches(tp, test_rows, rng, label_col="label",
                            per_class=max(1, test_rows // n_classes), n_classes=n_classes)
    out = fix / "central" / "test"
    out.mkdir(parents=True)
    pq.write_table(table, out / "part-00000.parquet")
    lab = table.column("label").to_numpy(zero_copy_only=False)
    return int(table.num_rows), int(len(np.unique(lab)))


def build_nested(fix, clients):
    """The owner-prefixed mount shape, as symlinks so it costs no extra disk."""
    nested = fix / "nested"
    for n in clients:
        dst = nested / "datasets" / OWNER / f"veremi-fl-{n}client" / f"{n}_client"
        dst.mkdir(parents=True)
        for item in (fix / "fl" / f"{n}_client").iterdir():
            (dst / item.name).symlink_to(item.resolve())
    dst = nested / "datasets" / OWNER / "veremi-nextgen2026-centralized" / "upload"
    dst.mkdir(parents=True)
    (dst / "test").symlink_to((fix / "central" / "test").resolve())
    return str(nested)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--clients", type=int, nargs="+", default=[20])
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument("--test-rows", type=int, default=16_000)
    ap.add_argument("--nested", action="store_true")
    a = ap.parse_args()

    fix = Path(a.dir)
    if fix.exists():
        shutil.rmtree(fix)
    rng = np.random.default_rng(7)
    report = {"scenarios": {}}
    for n in a.clients:
        got, rows = build_scenario(fix, n, a.rows, rng)
        report["scenarios"][n] = {"clients": got, "rows_per_client": rows}
    report["test_rows"], report["test_classes"] = build_test(fix, a.test_rows, rng)
    assert report["test_classes"] == 16, "fixture test set is missing classes"
    if a.nested:
        report["nested_root"] = build_nested(fix, a.clients)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

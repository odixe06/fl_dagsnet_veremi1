"""Execute a generated notebook's cells locally, against the fixture, before spending GPU quota.

The modules in `src/` are covered by `unit_test.py`, `run_e2e.py` and `resume_test.py`. What is
NOT covered by those is the notebook's own glue: resolving mounts by sentinel, reading the
sidecars, cross-checking footer rows, building `cfg`/`paths`, the disk estimate, and the results
cell. Every one of those has failed in this project's history *after* the algorithm was correct,
and each failure costs a Kaggle version and a session start.

This runs the real cells with three substitutions: the Kaggle paths point at the fixture, the
scenario is shrunk to the fixture's size, and training uses one CPU worker in FP32. The
2 x T4 probe cell is skipped -- it is the one cell that cannot pass here by construction.

    python tests/notebook_sim.py tinyproto-fp-train-20client [--rounds 2]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "papers" / "tinyproto-lee-2026" / "notebook"


def substitutions(fixture: Path, work: Path, n_clients: int, rounds: int,
                  total_rows: int, test_rows: int, is_mu: bool) -> list[tuple[str, str]]:
    subs = [
        ('SEARCH = [Path("/kaggle/input")]', f'SEARCH = [Path("{fixture}")]'),
        ('Path("/kaggle/input").rglob', f'Path("{fixture}").rglob'),
        ('"/kaggle/working/proj"', f'"{work}/proj"'),
        ('/kaggle/working/proj/', f'{work}/proj/'),
        ('Path("/kaggle/working")', f'Path("{work}")'),
        ('"/kaggle/working"', f'"{work}"'),
        ('Path("/kaggle/temp")', f'Path("{work}/temp")'),
        ('"/kaggle/temp/', f'"{work}/temp/'),
        ('shutil.disk_usage("/kaggle/working")', f'shutil.disk_usage("{work}")'),
        ("shutil.disk_usage('/kaggle/working')", f"shutil.disk_usage('{work}')"),
        ("'/kaggle/working'", f"'{work}'"),
        ("'/kaggle/input'", f"'{fixture}'"),
        ("'/kaggle/temp'", f"'{work}/temp'"),
        # generic prefixes: any quoted path under the three Kaggle roots
        ('"/kaggle/working/', f'"{work}/'),
        ("'/kaggle/working/", f"'{work}/"),
        ('"/kaggle/temp/', f'"{work}/temp/'),
        ("'/kaggle/temp/", f"'{work}/temp/"),
        ('"/kaggle/input/', f'"{fixture}/'),
        ("'/kaggle/input/", f"'{fixture}/"),
        ('"import_roots": ["/kaggle/input"]', '"import_roots": []'),
        # scenario shrunk to the fixture
        (f'"n_clients": 20,', f'"n_clients": {n_clients},'),
        ('assert TOTAL == 43_045_415', f'assert TOTAL == {total_rows}'),
        ('assert TEST_ROWS == 10_761_343', f'assert TEST_ROWS == {test_rows}'),
        ('"devices": ["cuda:0", "cuda:1"]', '"devices": ["cpu"]'),
        ('"assignment": ASSIGNMENT', '"assignment": [sorted(c for group in ASSIGNMENT for c in group)]'),
        ('"compile": True', '"compile": False'),
        ('"amp": True', '"amp": False'),
        ('"require_mu_selection": True', '"require_mu_selection": False'),
        ('"batch": 512', '"batch": 64'),
        ('"batch": 256', '"batch": 64'),
        # Exercise the W&B cell's no-key path: it must degrade to WANDB=None and keep going,
        # and a local fixture run must never create real W&B runs.
        ("_INLINE_WANDB_KEY = ", "_INLINE_WANDB_KEY = ''  #"),
        ('"eval_batch": 8192', '"eval_batch": 512'),
        ('"eval_batch": 16384', '"eval_batch": 512'),   # older notebooks
        ('"expected_round_s": 800.0', '"expected_round_s": 1.0'),
        ('"expected_round_s": 1300.0', '"expected_round_s": 1.0'),
        ('"expected_round_s": 2500.0', '"expected_round_s": 1.0'),
        ('"expected_round_s": 600.0', '"expected_round_s": 1.0'),
        # The calibration benchmarks are written for 2 x T4 with 29 GB of host RAM. Run locally
        # at full size they take WSL down: Inductor's compile workers alone exceed this box's
        # 7.6 GB. Smoke mode exercises the same code path with tiny sizes and no torch.compile.
        ("BENCH_SMOKE = False", "BENCH_SMOKE = True"),
    ]
    for n in (20, 50, 100):
        subs.append((f'"scenario": "{n}client", "n_clients": {n},',
                     f'"scenario": "{n}client", "n_clients": {n_clients},'))
        subs.append((f'"n_clients": {n},', f'"n_clients": {n_clients},'))
        subs.append((f'"run_name": "tinyproto_fp_{n}client"', f'"run_name": "sim_{n}client"'))
    subs.append(('"rounds": 50,', f'"rounds": {rounds},'))
    subs.append(('"rounds": 3,', f'"rounds": {rounds},'))
    subs.append(('"save_preds_rounds": [50]', f'"save_preds_rounds": [{rounds}]'))
    if is_mu:
        subs.append(("MU_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]", "MU_GRID = [0.1, 1.0]"))
        subs.append(("SWEEP_ROUNDS = 4", f"SWEEP_ROUNDS = {rounds}"))
    return subs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--fixture", default="/tmp/tinyproto_fixture")
    ap.add_argument("--work", default="/tmp/tinyproto_nbsim")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--run-bench", action="store_true",
                    help="also execute the calibration micro-benchmarks (slow, "
                         "and their timings mean nothing on this GPU)")
    a = ap.parse_args()

    if a.run_bench:
        avail = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1])
        print(f"available RAM: {avail/2**20:.1f} GiB (benchmarks run in SMOKE mode, no compile)")
        if avail / 2**20 < 2.0:
            print("refusing: under 2 GiB available; free memory first"); return 1

    fixture = Path(a.fixture)
    info = json.loads((fixture / "fixture.json").read_text())
    work = Path(a.work)
    shutil.rmtree(work, ignore_errors=True)
    (work / "temp").mkdir(parents=True, exist_ok=True)

    nb = nbf.read(NB_DIR / a.slug / f"{a.slug}.ipynb", as_version=4)
    is_mu = "mu-sweep" in a.slug
    subs = substitutions(fixture, work, info["n_clients"], a.rounds,
                         info["total_rows"], info["test_rows"], is_mu)

    g: dict = {"__name__": "__main__", "display": lambda *x: print(*x)}
    cells = [c for c in nb.cells if c.cell_type == "code"]
    print(f"simulating {a.slug}: {len(cells)} code cells, "
          f"{info['n_clients']} clients x {info['rows_per_client']} rows, {a.rounds} rounds\n")

    for i, c in enumerate(cells):
        src = c.source
        head = src.split("\n", 1)[0]
        if "torch.cuda.device_count()" in src and "assert n == 2" in src:
            print(f"  [skip] cell {i}: 2 x T4 probe (cannot pass on one local GPU)")
            g.update({"os": __import__("os"), "sys": sys, "json": json, "time": __import__("time"),
                      "math": __import__("math"), "shutil": shutil,
                      "np": __import__("numpy"), "torch": __import__("torch")})
            continue
        if head.startswith("%%writefile"):
            target = Path(head.split()[-1])
            for old, new in subs:
                target = Path(str(target).replace(old.strip('"'), new.strip('"')))
            target.parent.mkdir(parents=True, exist_ok=True)
            body = src.split("\n", 1)[1]
            if is_mu and target.name == "sweep.py":
                # The notebook asserts its knob cell equals the frozen MU_PROTOCOL. The fixture
                # cannot afford the real 5x4 grid, so shrink the PROTOCOL to match the shrunken
                # knobs rather than skipping the assertion -- that keeps the check executing, and
                # a real notebook (whose knobs are the real protocol) is still verified by it.
                body = body.replace('"grid": [0.03, 0.1, 0.3, 1.0, 3.0],', '"grid": [0.1, 1.0],')
                body = body.replace('"rounds": 4,', f'"rounds": {a.rounds},')
            target.write_text(body)
            print(f"  [write] cell {i}: {target.name}")
            continue
        if not a.run_bench and ("Micro-benchmark" in src or 'report["step_ms"]' in src):
            print(f"  [skip] cell {i}: micro-benchmarks (pass --run-bench for a smoke run)")
            continue
        for old, new in subs:
            src = src.replace(old, new)
        print(f"  [exec] cell {i}: {head[:76]}")
        try:
            exec(compile(src, f"<cell {i}>", "exec"), g)
        except BaseException as e:      # SystemExit too -- a bare traceback prints nothing for it
            print(f"\n  CELL {i} FAILED: {type(e).__name__}: {e}\n" + "-" * 70)
            traceback.print_exception(type(e), e, e.__traceback__)
            print("-" * 70)
            return 1

    print("\nnotebook simulation completed with no error")
    run_name = g.get("CFG", {}).get("run_name")
    if run_name and (work / "runs" / run_name / "complete").exists():
        done = sorted((work / "runs" / run_name / "complete").glob("*.done"))
        print(f"committed rounds: {[p.name for p in done]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

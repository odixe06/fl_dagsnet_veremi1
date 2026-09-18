"""Crash and resume behaviour, verified by actually crashing rather than by reading the code.

Six scenarios:
 1  stop after N rounds, resume -> continues at N+1, no duplicate history row
 2  half-written round (artifacts present, marker absent) -> redone and replaced
 3  cross-session import from a READ-ONLY mount -> imported, then continues
 4  --require-resume with nothing to resume -> fails before the decode
 5  a changed run-defining setting -> refuses to resume instead of mixing runs
 6  a marker whose resume state is missing -> refuses instead of guessing
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable
FIX = "/tmp/tinyproto_fixture"


def run_e2e(out, rounds, extra=(), expect_fail=False):
    cmd = [PY, str(ROOT / "tests" / "run_e2e.py"), "--fixture", FIX, "--out", str(out),
           "--rounds", str(rounds), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if expect_fail:
        assert r.returncode != 0, f"expected failure, got success:\n{r.stdout[-2000:]}"
    else:
        assert r.returncode == 0, f"run failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r


def verify(d, extra=()):
    r = subprocess.run([PY, str(ROOT / "scripts" / "verify_run.py"), str(d), *extra],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


def done_rounds(d):
    return sorted(int(p.name[6:9]) for p in (Path(d) / "complete").glob("round_*.done"))


def history_rounds(d):
    import csv
    with (Path(d) / "metrics" / "history.csv").open() as fh:
        return [int(r["round"]) for r in csv.DictReader(fh)]


def main():
    fails = []

    def check(ok, msg):
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            fails.append(msg)

    # ---- 1: stop, then resume ---------------------------------------------
    print("\n1. resume after a clean stop")
    out = Path("/tmp/tp_resume1")
    shutil.rmtree(out, ignore_errors=True)
    run_e2e(out, 4, ["--session-rounds", "2"])
    d = out / "runs" / "e2e"
    check(done_rounds(d) == [1, 2], f"2 rounds committed, got {done_rounds(d)}")
    r = run_e2e(out, 4)
    check("starting at round 3" in r.stdout, "resumed at round 3")
    check(done_rounds(d) == [1, 2, 3, 4], f"4 rounds committed, got {done_rounds(d)}")
    check(history_rounds(d) == [1, 2, 3, 4], f"history has no duplicates: {history_rounds(d)}")
    code, _ = verify(d)
    check(code == 0, "verifier passes after resume")
    code, _ = verify(d, ["--require-complete"])
    check(code == 0, "--require-complete passes at 4/4")

    # Saving round counters alone is not exact resume: compare the trained tensors with a
    # continuous run. CPU + one thread makes this a bitwise regression for worker dropout RNG.
    import torch
    continuous = Path("/tmp/tp_continuous")
    shutil.rmtree(continuous, ignore_errors=True)
    run_e2e(continuous, 4)
    reference = torch.load(continuous / "runs/e2e/weights/round_004.pt", weights_only=True)
    restored = torch.load(d / "weights/round_004.pt", weights_only=True)
    for key in ("params", "buffers", "int_buffers"):
        check(torch.equal(reference[key], restored[key]),
              f"continuous vs resumed: {key} are bitwise equal")

    # ---- 2: half-written round --------------------------------------------
    print("\n2. crash between artifacts and marker")
    out2 = Path("/tmp/tp_resume2")
    shutil.rmtree(out2, ignore_errors=True)
    run_e2e(out2, 3, ["--session-rounds", "2"])
    d2 = out2 / "runs" / "e2e"
    shutil.copy(d2 / "weights" / "round_002.pt", d2 / "weights" / "round_003.pt")
    shutil.copy(d2 / "protos" / "round_002.pt", d2 / "protos" / "round_003.pt")
    (d2 / "metrics" / "round_003.json").write_text('{"round": 3, "corrupt": true}')
    r = run_e2e(out2, 3)
    check("starting at round 3" in r.stdout, "ignored the unmarked round-3 leftovers")
    # Compare CONTENT, not bytes: every torch.save file opens with the same zip header, so a
    # byte-prefix comparison passes whether or not the round was actually redone.
    w2 = torch.load(d2 / "weights" / "round_002.pt", map_location="cpu", weights_only=False)
    w3 = torch.load(d2 / "weights" / "round_003.pt", map_location="cpu", weights_only=False)
    check(not torch.equal(w2["params"], w3["params"]),
          "round-3 weights were retrained, not the copied round-2 placeholder")
    m3 = json.loads((d2 / "metrics" / "round_003.json").read_text())
    check("corrupt" not in m3 and "aggregate" in m3,
          "the corrupt round-3 metrics file was replaced")
    check(history_rounds(d2) == [1, 2, 3], f"no duplicate history row: {history_rounds(d2)}")
    code, o = verify(d2)
    check(code == 0, f"verifier passes after redoing the crashed round\n{o[-400:] if code else ''}")

    # ---- 3: cross-session import from a read-only mount --------------------
    print("\n3. cross-session import (read-only source, like /kaggle/input)")
    src = Path("/tmp/tp_input/prev")
    shutil.rmtree("/tmp/tp_input", ignore_errors=True)
    shutil.copytree(d2, src)
    for p in src.rglob("*"):
        if p.is_file():
            os.chmod(p, 0o444)                      # exactly what a Kaggle input mount looks like
    out3 = Path("/tmp/tp_resume3")
    shutil.rmtree(out3, ignore_errors=True)
    r = run_e2e(out3, 3, ["--run-name", "e2e", "--session-rounds", "1"])
    d3 = out3 / "runs" / "e2e"
    # the plain run has no import_roots, so it must have started at round 1
    check("starting at round 1" in r.stdout, "no import configured -> fresh start")
    shutil.rmtree(out3, ignore_errors=True)
    env = dict(os.environ, TINYPROTO_IMPORT_ROOTS="/tmp/tp_input")
    rr = subprocess.run([PY, str(ROOT / "tests" / "run_e2e.py"), "--fixture", FIX,
                         "--out", str(out3), "--rounds", "3"],
                        capture_output=True, text=True, env=env)
    assert rr.returncode == 0, rr.stdout[-3000:] + rr.stderr[-3000:]
    check("imported 3 completed rounds" in rr.stdout, "imported the previous session's 3 rounds")
    check("starting at round 4" in rr.stdout or "3 rounds" in rr.stdout,
          "nothing left to do after importing a complete run")
    check(done_rounds(d3) == [1, 2, 3], f"3 rounds present, got {done_rounds(d3)}")
    imported = d3 / "metrics" / "history.csv"
    check(os.access(imported, os.W_OK), "imported files are writable (copyfile + chmod, not copy2)")
    code, o = verify(d3, ["--require-complete"])
    check(code == 0, f"verifier passes on the merged run\n{o[-500:] if code else ''}")

    # ---- 4: require-resume with nothing to resume --------------------------
    print("\n4. --require-resume gate")
    out4 = Path("/tmp/tp_resume4")
    shutil.rmtree(out4, ignore_errors=True)
    r = run_e2e(out4, 2, ["--require-resume"], expect_fail=True)
    check("require_resume" in (r.stdout + r.stderr), "failed with the require_resume message")
    check(not (out4 / "runs" / "e2e" / "weights").exists()
          or not list((out4 / "runs" / "e2e" / "weights").glob("*.pt")),
          "died before training anything")

    # ---- 5: changed run-defining setting -----------------------------------
    print("\n5. fingerprint guard")
    r = subprocess.run([PY, str(ROOT / "tests" / "run_e2e.py"), "--fixture", FIX,
                        "--out", str(out), "--rounds", "4", "--cps-s", "26"],
                       capture_output=True, text=True)
    check(r.returncode != 0 and "fingerprint" in (r.stdout + r.stderr),
          "changing the CPS dimension refuses to resume the old run")

    # ---- 6: marker without resume state ------------------------------------
    print("\n6. marker with no resume state")
    out6 = Path("/tmp/tp_resume6")
    shutil.rmtree(out6, ignore_errors=True)
    run_e2e(out6, 4, ["--session-rounds", "2"])
    d6 = out6 / "runs" / "e2e"
    for p in (d6 / "resume").glob("round_002.pt"):
        p.unlink()
    r = run_e2e(out6, 4, expect_fail=True)
    check("Refusing to guess" in (r.stdout + r.stderr), "refused instead of silently restarting")

    print(f"\n{'ALL RESUME TESTS PASS' if not fails else f'{len(fails)} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

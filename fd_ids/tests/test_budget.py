"""The session deadline must be a deadline, not a suggestion.

Three things were wrong at once: the clock started after spawn/prepack/compile, so minutes
the 12 h cap charges for were invisible; the budget was only consulted after a round, so a
session with nothing left still began one; and `seconds` stopped before the commit it was
supposed to be paying for. Small synthetic dataset, one CPU worker -- this measures the
control flow, never the speed.
"""
import json, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/fd-ids-2025"))
from proj.model import CFG
from proj import ckpt as C
from proj import driver as D

CLIENTS = {0: 300, 1: 200}
NT = 400


def make(tmp):
    cache = tmp / "cache"; cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = sum(CLIENTS.values())
    np.save(cache / "train_X.f16.npy", rng.standard_normal((n, 66)).astype(np.float16))
    np.save(cache / "train_y.u8.npy", rng.integers(0, 16, n, dtype=np.uint8))
    np.save(cache / "test_X.f16.npy", rng.standard_normal((NT, 66)).astype(np.float16))
    np.save(cache / "test_y.u8.npy", rng.integers(0, 16, NT, dtype=np.uint8))
    spans, off = {}, 0
    for c, k in sorted(CLIENTS.items()):
        spans[c] = (off, off + k); off += k
    return cache, spans


def cfg_for(cache, name, **over):
    base = {**CFG, "n_features": 66, "device": "cpu", "world_size": 1, "compile": False,
            "batch": 64, "eval_batch": 256, "lr": 1e-3, "lam": 0.5, "beta": 0.1,
            "mu": 0.01, "temperature": 3.0, "clip": 1.0, "seed": 42, "rounds": 2,
            "local_epochs": 1, "n_clients": len(CLIENTS), "data_id": "b" * 16,
            "n_test": NT, "cache": str(cache), "run_name": name,
            "max_seconds": 3600, "require_resume": False,
            "max_skips_per_client": 16, "finalize_reserve_seconds": 0}
    base.update(over)
    return base


def main():
    tmp = Path(tempfile.mkdtemp())
    cache, spans = make(tmp)
    C.run_dir = lambda name, _t=tmp: _mk(_t / "runs" / name)
    names = [f"c{i}" for i in range(16)]

    # 1. the budget is already gone before the first round: start nothing, say so
    cfg = cfg_for(cache, "gone", max_seconds=60, finalize_reserve_seconds=10)
    hist = D.run(cfg, spans, names, t_origin=time.monotonic() - 3600)
    assert hist == [], f"a round was started with no budget left: {hist}"
    assert C.last_complete_round(tmp / "runs" / "gone") is None
    print("   startup already over budget -> 0 rounds, nothing committed")

    # 2. a normal run completes both rounds and `seconds` covers the commit
    cfg = cfg_for(cache, "ok")
    hist = D.run(cfg, spans, names, t_origin=time.monotonic())
    assert len(hist) == 2, hist
    d = tmp / "runs" / "ok"
    assert C.last_complete_round(d, C.fingerprint(cfg)) == 2
    for r in hist:
        assert r["seconds"] > 0 and r["applied"] + r["skipped"] == r["steps"]
        assert "vram_train_gb" in r and "ce_client_mean" in r and "backend" in r
    js = json.loads((d / "metrics" / "round_002.json").read_text())
    for k in ("seconds", "ce_client_mean", "kd_client_mean", "steps", "applied",
              "skipped", "teacher_sec", "backend"):
        assert k in js, f"metrics json cannot rebuild history: no {k}"
    print(f"   full run: 2 rounds, seconds={[round(r['seconds'], 2) for r in hist]}")

    # 3. a budget that fits fewer rounds than asked must stop early and cleanly. Sized
    #    from the run above rather than from a guess, so the test does not depend on how
    #    fast this machine happens to be.
    # Round 1 carries warm-up, so the steady round is the last one. The driver reports
    # what startup actually cost on this box; guessing it is what made an earlier version
    # of this test pass or fail with the machine's mood.
    per_round = hist[-1]["seconds"]
    budget = cfg["startup_seconds"] + 2.5 * per_round
    cfg = cfg_for(cache, "reserve", rounds=5, max_seconds=budget,
                  finalize_reserve_seconds=0)
    t = time.monotonic()
    hist = D.run(cfg, spans, names, t_origin=t)
    assert 1 <= len(hist) < 5, f"budget did not stop the loop early: {len(hist)} rounds"
    assert time.monotonic() - t < budget + 10 * per_round + 5, "ran far past its budget"
    d3 = tmp / "runs" / "reserve"
    assert C.last_complete_round(d3, C.fingerprint(cfg)) == len(hist), \
        "stopped mid-round and left a half-committed one behind"
    print(f"   budget of {budget:.1f}s at {per_round:.2f}s/round -> stopped after "
          f"{len(hist)} of 5 rounds, every committed round intact")

    # 4. resuming that run continues where it stopped rather than redoing anything
    done = len(hist)
    cfg = cfg_for(cache, "reserve", rounds=done + 1)
    hist2 = D.run(cfg, spans, names, t_origin=time.monotonic())
    assert [r["round"] for r in hist2] == [done + 1], hist2
    rounds = [int(l.split(",")[0]) for l in
              (d3 / "history.csv").read_text().splitlines()[1:]]
    assert rounds == list(range(1, done + 2)), rounds
    print(f"   resumed at round {done + 1}; history holds exactly 1..{done + 1}")

    # 5. nothing left to do is not an error
    hist3 = D.run(cfg_for(cache, "reserve", rounds=done + 1), spans, names,
                  t_origin=time.monotonic())
    assert hist3 == [], hist3
    print("   already complete -> returns cleanly with no rounds")

    shutil.rmtree(tmp)
    print("\nALL 5 BUDGET CHECKS PASSED")


def _mk(d):
    for s in C.SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d


# spawn re-imports this module in the worker; main() must not run there
if __name__ == "__main__":
    main()

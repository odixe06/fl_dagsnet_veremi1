"""The client schedule must not change the round.

Clients are handed out longest-first to whichever GPU is free, so which rank trains which
client, and how many ran before it, depends on a completion race. Before every stochastic
input was derived from (seed, round, client), the Dropout masks came from a generator seeded
once per rank -- and the same round on the same data produced different weights whenever the
schedule shifted. Two workers against one is the cheapest way to shift it.

Its own file on purpose: two CPU workers plus this parent sit near the local memory ceiling,
and the end-to-end smoke test already spends most of it.
"""
import shutil, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/fd-ids-2025"))
from proj.model import CFG
from proj import ckpt as C
from proj import driver as D

CLIENTS = {0: 700, 1: 300, 2: 450, 3: 180}       # unequal, so LPT actually reorders them
NT = 512


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


def cfg_for(cache, name, world):
    return {**CFG, "n_features": 66, "device": "cpu", "world_size": world,
            "compile": False, "batch": 128, "eval_batch": 256, "lr": 1e-3, "lam": 0.5,
            "beta": 0.1, "mu": 0.01, "temperature": 3.0, "clip": 1.0, "seed": 42,
            "rounds": 2, "local_epochs": 1, "n_clients": len(CLIENTS),
            "data_id": "s" * 16, "n_test": NT, "cache": str(cache), "run_name": name,
            "max_seconds": 3600, "require_resume": False,
            "max_skips_per_client": 16, "finalize_reserve_seconds": 0}


def weights(d, r):
    return torch.load(d / "weights" / f"round_{r:03d}.pt", map_location="cpu",
                      weights_only=True)["model"]


def main():
    tmp = Path(tempfile.mkdtemp())
    cache, spans = make(tmp)
    C.run_dir = lambda name, _t=tmp: _mk(_t / "runs" / name)
    names = [f"c{i}" for i in range(16)]

    h1 = D.run(cfg_for(cache, "w1", 1), spans, names, t_origin=time.monotonic())
    h2 = D.run(cfg_for(cache, "w2", 2), spans, names, t_origin=time.monotonic())
    assert len(h1) == len(h2) == 2, (len(h1), len(h2))
    d1, d2 = tmp / "runs" / "w1", tmp / "runs" / "w2"

    for r in (1, 2):
        a, b = weights(d1, r), weights(d2, r)
        assert set(a) == set(b)
        dmax = max((a[k].float() - b[k].float()).abs().max().item() for k in a)
        assert dmax == 0.0, f"round {r}: 1 worker and 2 workers differ by {dmax}"
        print(f"   round {r}: 1 worker vs 2 workers, max|delta weight| = {dmax}")

    for r1, r2 in zip(h1, h2):
        for k in ("f1_macro", "accuracy", "ce_client_mean", "kd_client_mean",
                  "grad_norm", "steps", "applied", "skipped"):
            assert r1[k] == r2[k], f"round {r1['round']} {k}: {r1[k]} vs {r2[k]}"
    print("   every reported metric and step count identical across both topologies")

    # and the whole thing is reproducible: a third run of the 2-worker config must land
    # on the same weights again, not merely agree with the 1-worker one by luck.
    D.run(cfg_for(cache, "w2b", 2), spans, names, t_origin=time.monotonic())
    d3 = tmp / "runs" / "w2b"
    dmax = max((weights(d2, 2)[k].float() - weights(d3, 2)[k].float()).abs().max().item()
               for k in weights(d2, 2))
    assert dmax == 0.0, f"repeating the same config gave different weights: {dmax}"
    print(f"   repeat of the same 2-worker config: max|delta weight| = {dmax}")

    shutil.rmtree(tmp)
    print("\nALL 3 SCHEDULE-INDEPENDENCE CHECKS PASSED")


def _mk(d):
    for s in C.SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    main()

"""Unit checks on the paper's equations, each against an independent reference implementation.

These test the FORMULAS. They cannot certify a run -- that is `scripts/verify_run.py`'s job.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cps import build_masks, compress, decompress, mask_index, mask_stats  # noqa: E402
from src.data import stratified_holdout  # noqa: E402
from src.driver import resolve_mu  # noqa: E402
from src.evaluate import fold_bn, verify_fold  # noqa: E402
from src.metrics import METRIC_KEYS, metrics_from_confusion  # noqa: E402
from src.model import FlatPacker, build_model  # noqa: E402
from src.protos import (ProtoAccumulator, ProtoRegularizer, aggregate_global, proto_logits,
                        scale_and_compress)  # noqa: E402

FAIL: list[str] = []


def check(ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        FAIL.append(msg)


def t_masks():
    print("\nCPS masks — §4.1, Def. 1 and 2")
    for s in (26, 50, 128):
        M = build_masks(256, 16, s, seed=42)
        st = mask_stats(M)
        check((M.sum(1) == s).all(), f"s={s}: every mask has exactly s ones")
        check(st["overlap_max"] <= st["overlap_search_target"],
              f"s={s}: max overlap {st['overlap_max']} <= search target {st['overlap_search_target']}")
        check(st["overlap_mean"] < st["overlap_random_expectation"],
              f"s={s}: mean overlap {st['overlap_mean']:.2f} beats random "
              f"{st['overlap_random_expectation']:.2f}")
        c = np.random.default_rng(s).normal(size=(16, 256))
        check(np.array_equal(decompress(compress(c, M), M), c * M),
              f"s={s}: decompress(compress(c)) == c ⊙ m (Eq. 7 and 8)")
        idx = mask_index(M)
        check(all(set(idx[j]) == set(np.flatnonzero(M[j])) for j in range(16)),
              f"s={s}: mask_index agrees with the boolean mask")
    check(np.array_equal(build_masks(256, 16, 50, seed=42), build_masks(256, 16, 50, seed=42)),
          "mask construction is deterministic under a fixed seed")
    check(not np.array_equal(build_masks(256, 16, 50, seed=1), build_masks(256, 16, 50, seed=2)),
          "different seeds give different masks")


def t_aggregate():
    print("\nAPS aggregation — Eq. (10)")
    rng = np.random.default_rng(0)
    K, d, s, n = 16, 256, 50, 7
    M = build_masks(d, K, s, seed=42)
    idx = mask_index(M)
    c = rng.normal(size=(n, K, d))
    cnt = rng.integers(0, 500, size=(n, K)).astype(np.int64)
    cnt[0, 3] = 0
    cnt[1, 3] = 0                                     # class 3 held by 5 of 7 clients

    ups = []
    for i in range(n):
        sc, pres = scale_and_compress(c[i], cnt[i], idx)
        ups.append((i, sc, pres))
    g, nj = aggregate_global(ups, K, s)

    ref = np.zeros((K, s))
    for j in range(K):
        who = [i for i in range(n) if cnt[i, j] > 0]
        if who:
            ref[j] = np.mean([cnt[i, j] * c[i, j, M[j].astype(bool)] for i in who], axis=0)
    check(np.allclose(g, ref, atol=1e-9), "ĉ_G equals the naive mean of n_ij ĉ_L over N_j")
    check(nj[3] == 5 and (nj[[0, 1, 2]] == n).all(), "|N_j| counts only clients holding class j")

    try:
        aggregate_global([ups[2], ups[0]], K, s)
        check(False, "unsorted uploads must be rejected (fp addition is not associative)")
    except AssertionError:
        check(True, "unsorted uploads are rejected")

    # A class no client holds gets an all-zero prototype AND |N_j| = 0, so the caller can tell
    # "no target" from "a target that happens to be zero".
    cnt2 = cnt.copy()
    cnt2[:, 5] = 0
    ups2 = [(i, *scale_and_compress(c[i], cnt2[i], idx)) for i in range(n)]
    g2, nj2 = aggregate_global(ups2, K, s)
    check(nj2[5] == 0 and np.abs(g2[5]).max() == 0.0, "an absent class yields |N_j| = 0 and zeros")


def t_mu():
    print("\nAPS scaling constant μ — Eq. (11)")
    rng = np.random.default_rng(1)
    K, n = 16, 12
    cnt = rng.integers(1, 5000, size=(n, K)).astype(np.int64)
    cnt[0, 7] = 0
    mu_pc, prov = resolve_mu("per_class", 1.0, cnt, K)
    # With mu_j = 1 / mean_{i in N_j}(n_ij), mu_j·ĉ_G[j] is EXACTLY the sample-weighted mean of
    # the local prototypes -- the quantity Eq. (4) is written to produce.
    d, s = 256, 50
    M = build_masks(d, K, s, seed=42)
    idx = mask_index(M)
    c = rng.normal(size=(n, K, d))
    ups = [(i, *scale_and_compress(c[i], cnt[i], idx)) for i in range(n)]
    g, _ = aggregate_global(ups, K, s)
    for j in range(K):
        who = [i for i in range(n) if cnt[i, j] > 0]
        w = np.array([cnt[i, j] for i in who], dtype=np.float64)
        want = (w[:, None] * np.stack([c[i, j, M[j].astype(bool)] for i in who])).sum(0) / w.sum()
        check(np.allclose(mu_pc[j] * g[j], want, atol=1e-9),
              f"class {j}: μ_j·ĉ_G is the sample-weighted mean prototype") if j < 2 else None
    ok = all(np.allclose(mu_pc[j] * g[j],
                         (np.array([cnt[i, j] for i in range(n) if cnt[i, j] > 0], np.float64)[:, None]
                          * np.stack([c[i, j, M[j].astype(bool)] for i in range(n) if cnt[i, j] > 0])
                          ).sum(0) / np.array([cnt[i, j] for i in range(n) if cnt[i, j] > 0]).sum(),
                         atol=1e-9) for j in range(K))
    check(ok, "per-class μ makes the target the sample-weighted mean for EVERY class")

    mu_s, _ = resolve_mu("inv_mean_nij", 1.0, cnt, K)
    check(len(set(np.round(mu_s, 15))) == 1, "inv_mean_nij gives one scalar for all classes")
    mu_a, _ = resolve_mu("absolute", 3.5e-6, cnt, K)
    check(np.allclose(mu_a, 3.5e-6), "absolute uses the value verbatim")


def t_regularizer():
    print("\nPrototype regularizer — Eq. (5) and (11)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = torch.Generator(device="cpu").manual_seed(0)
    K, d, s, B = 16, 256, 50, 64
    M = torch.from_numpy(build_masks(d, K, s, seed=42).astype(np.float32)).to(dev)
    g = (torch.randn(K, d, generator=rng) * M.cpu()).to(dev)
    has = torch.ones(K, dtype=torch.bool, device=dev)
    has[4] = False
    mu = torch.full((K,), 3e-4, device=dev)
    reg = ProtoRegularizer(g, has, M, mu, s)

    h = torch.randn(B, d, generator=rng).to(dev)
    y = torch.randint(0, K, (B,), generator=rng).to(dev)
    got = float(reg(h, y))

    tot, cnt = 0.0, 0
    for b in range(B):
        j = int(y[b])
        if not bool(has[j]):
            continue
        diff = (h[b] - mu[j] * g[j]) * M[j]
        tot += float((diff ** 2).sum())
        cnt += 1
    check(abs(got - tot / (cnt * s)) < 1e-5, "R_i equals the naive per-sample masked MSE / s")

    yb = torch.full((B,), 4, device=dev)                 # every row's class has no prototype
    check(float(reg(h, yb)) == 0.0, "rows with no global prototype contribute exactly 0, not NaN")

    # with s = d and mu = 1 the term must reduce to plain nn.MSELoss on dense prototypes
    Mf = torch.ones(K, d, device=dev)
    gd = torch.randn(K, d, generator=rng).to(dev)
    reg2 = ProtoRegularizer(gd, torch.ones(K, dtype=torch.bool, device=dev), Mf,
                            torch.ones(K, device=dev), d)
    want = torch.nn.functional.mse_loss(h, gd.index_select(0, y))
    check(abs(float(reg2(h, y)) - float(want)) < 1e-5,
          "with s = d and μ = 1 the term is exactly nn.MSELoss (FedProto's form)")


def t_eq12():
    print("\nPrediction rule — Eq. (12)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K, d, B = 16, 256, 256
    torch.manual_seed(0)
    h = torch.randn(B, d, device=dev)
    c = torch.randn(K, d, device=dev).abs()
    present = torch.ones(K, dtype=torch.bool, device=dev)
    present[[2, 9]] = False
    got = proto_logits(h, c, present).argmax(1)
    dist = torch.cdist(h, c)
    dist[:, ~present] = float("inf")
    want = dist.argmin(1)
    check(int((got != want).sum()) == 0, "argmax of the score equals argmin of the L2 distance")
    check(int((got == 2).sum()) == 0 and int((got == 9).sum()) == 0,
          "a class the client has no prototype for is never predicted")


def t_accumulator():
    print("\nLocal prototypes — Eq. (3)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K, d, N = 16, 256, 5000
    torch.manual_seed(0)
    f = torch.randn(N, d, device=dev)
    y = torch.randint(0, K, (N,), device=dev)
    y[y == 11] = 0                                        # class 11 absent
    acc = ProtoAccumulator(K, d, dev)
    for i in range(0, N, 333):                            # ragged batches
        acc.update(f[i:i + 333], y[i:i + 333])
    c, cnt = acc.finish()
    ok = True
    for j in range(K):
        m = y == j
        if int(m.sum()) == 0:
            ok &= float(c[j].abs().max()) == 0.0 and float(cnt[j]) == 0.0
        else:
            ok &= bool(torch.allclose(c[j], f[m].mean(0), atol=1e-4))
            ok &= int(cnt[j]) == int(m.sum())
    check(ok, "class means and counts are exact and batch-boundary independent; absent class = 0")


def t_packer():
    print("\nFlat weight packing")
    torch.manual_seed(0)
    m = build_model()
    for b in m.modules():
        if isinstance(b, torch.nn.BatchNorm1d):
            b.running_mean.normal_(0, 1)
            b.running_var.uniform_(0.5, 2)
            b.num_batches_tracked.fill_(7)
    p = FlatPacker(m)
    a, b_, i = p.pack(m)
    check(a.numel() == 395_024, f"param vector is 395,024 long, got {a.numel()}")
    sd = p.to_state_dict(a, b_, i)
    m2 = build_model()
    m2.load_state_dict(sd, strict=True)
    x = torch.randn(32, 66)
    m.eval()
    m2.eval()
    with torch.no_grad():
        check(float((m(x) - m2(x)).abs().max()) == 0.0, "round trip is bit-exact")
    check(all(int(v) == 7 for k, v in m2.state_dict().items() if "num_batches_tracked" in k),
          "int64 BatchNorm counters survive the round trip")
    m3 = build_model()
    p.load_into(m3, a, b_, i)
    m3.eval()
    with torch.no_grad():
        check(float((m(x) - m3(x)).abs().max()) == 0.0, "in-place load_into matches too")


def t_fold():
    print("\nBatchNorm folding (evaluation path)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    m = build_model().to(dev).eval()
    for b in m.modules():
        if isinstance(b, torch.nn.BatchNorm1d):
            b.running_mean.normal_(0, 1)
            b.running_var.uniform_(0.3, 3.0)
    f = fold_bn(m)
    v = verify_fold(m, f, torch.randn(4096, 66, device=dev))
    check(v["max_abs_dlogit"] < 1e-3, f"folded logits match within fp32 noise: {v}")
    check(not any(isinstance(x, torch.nn.BatchNorm1d) for x in f.modules()),
          "no BatchNorm survives in the folded module")


def t_holdout():
    print("\nValidation holdout for the μ sweep")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 16, 20000).astype(np.int8)
    spans = {0: (0, 8000), 1: (8000, 20000)}
    tr, va = stratified_holdout(y, spans, 0.02, seed=42)
    check(len(tr) + len(va) == len(y), "train+val partition the rows exactly")
    check(len(np.intersect1d(tr, va)) == 0, "train and val are disjoint")
    check(abs(len(va) / len(y) - 0.02) < 0.01, f"val is ~2% of rows ({len(va)}/{len(y)})")
    for cid, (lo, hi) in spans.items():
        for c in np.unique(y[lo:hi]):
            n_va = int((y[va[(va >= lo) & (va < hi)]] == c).sum())
            check(n_va >= 1, f"client {cid} class {c} has at least one validation row") \
                if c < 2 else None
    tr2, va2 = stratified_holdout(y, spans, 0.02, seed=42)
    check(np.array_equal(va, va2), "the split is reproducible under a fixed seed")


def t_metrics():
    print("\n10 metrics vs sklearn")
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    rng = np.random.default_rng(0)
    for name, (yt, yp) in {
        "typical": (rng.integers(0, 16, 5000), rng.integers(0, 16, 5000)),
        "one-class prediction": (rng.integers(0, 16, 5000), np.zeros(5000, int)),
        "classes absent from truth": (rng.integers(0, 4, 5000), rng.integers(0, 16, 5000)),
        "perfect": (lambda a: (a, a))(rng.integers(0, 16, 5000)),
    }.items():
        cm = confusion_matrix(yt, yp, labels=np.arange(16))
        mine = metrics_from_confusion(cm)
        ref = {"accuracy": float(accuracy_score(yt, yp))}
        for avg in ("macro", "micro", "weighted"):
            p, r, f, _ = precision_recall_fscore_support(yt, yp, average=avg,
                                                        labels=np.arange(16), zero_division=0)
            ref[f"precision_{avg}"], ref[f"recall_{avg}"], ref[f"f1_{avg}"] = map(float, (p, r, f))
        dmax = max(abs(mine[k] - ref[k]) for k in METRIC_KEYS)
        check(dmax < 1e-12, f"{name}: max|Δ| vs sklearn = {dmax:.2e}")


if __name__ == "__main__":
    for fn in (t_masks, t_aggregate, t_mu, t_regularizer, t_eq12, t_accumulator,
               t_packer, t_fold, t_holdout, t_metrics):
        fn()
    print(f"\n{'ALL UNIT TESTS PASS' if not FAIL else f'{len(FAIL)} FAILED'}")
    sys.exit(1 if FAIL else 0)

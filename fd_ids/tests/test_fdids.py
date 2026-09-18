"""The FD-IDS math: closed-form proximal gradient, flat round-trip, aggregation, KD loss.

Each of these fails silently if wrong -- the model still trains and still emits 16 logits.
"""
import sys, copy
from pathlib import Path
import math
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/fd-ids-2025"))
from proj.model import build_model, CFG
from proj.fdids import layout, flatten, unflatten_into, aggregate, client_update
from proj.driver import check_updates

CFGD = {**CFG, "n_features": 66}
LAM, BETA, MU, T = 0.5, 0.1, 0.01, 3.0


def grads_of(model, x, y, zt, proximal_mode, anchor_flat, n_params):
    """One backward. proximal_mode: 'autograd' puts Eq.(3) in the graph, 'closed' adds
    its gradient afterwards. Everything else is held identical between the two."""
    torch.manual_seed(1234)                      # same Dropout mask on both paths
    bn = copy.deepcopy({k: v.clone() for k, v in model.state_dict().items()
                        if not v.is_floating_point() or "running" in k})
    model.zero_grad(set_to_none=True)
    zs = model(x)
    l_hard = F.cross_entropy(zs, y)
    l_soft = (T * T) * F.kl_div(F.log_softmax(zs / T, 1), F.log_softmax(zt / T, 1),
                                reduction="batchmean", log_target=True)
    loss = LAM * l_hard + (1 - LAM) * l_soft
    params = list(model.parameters())
    if proximal_mode == "autograd":
        o = 0
        prox = 0.0
        for p in params:
            n = p.numel()
            prox = prox + ((p.reshape(-1) - anchor_flat[o:o + n]) ** 2).sum()
            o += n
        loss = loss + BETA * (MU / 2) * prox
    loss.backward()
    if proximal_mode == "closed":
        flat = torch.cat([p.detach().reshape(-1) for p in params])
        d = flat - anchor_flat
        o = 0
        for p in params:
            n = p.numel()
            p.grad.add_(d[o:o + n].view_as(p), alpha=BETA * MU)
            o += n
    g = torch.cat([p.grad.reshape(-1).clone() for p in params])
    # restore BN buffers AFTER backward: load_state_dict during forward bumps version
    # counters and autograd then refuses to run
    with torch.no_grad():
        sd = model.state_dict()
        for k, v in bn.items(): sd[k].copy_(v)
    return g


def reference_client(model, opt, X, Y, lo, hi, anchor, ZT, cfg, seed):
    """Eq. (6) written out with the proximal term IN the graph, stepped by the same Adam.

    This is the reference client_update must match: same shuffle, same batching, same clip,
    same optimizer. Only the proximal path differs -- autograd here, closed form there."""
    params = list(model.parameters())
    B, T = cfg["batch"], cfg["temperature"]
    lam, beta, mu, clip = cfg["lam"], cfg["beta"], cfg["mu"], cfg["clip"]
    gen = torch.Generator(device=X.device); gen.manual_seed(seed)
    torch.manual_seed(seed)
    seen = []
    opt.zero_grad(set_to_none=True)
    for _e in range(cfg["local_epochs"]):
        perm = lo + torch.randperm(hi - lo, generator=gen, device=X.device)
        for i in range(0, hi - lo, B):
            idx = perm[i:i + B]
            seen.append(idx.clone())
            zs = model(X[idx].float()).float()
            l_hard = F.cross_entropy(zs, Y[idx].long())
            l_soft = (T * T) * F.kl_div(F.log_softmax(zs / T, 1),
                                        F.log_softmax(ZT[idx - lo].float() / T, 1),
                                        reduction="batchmean", log_target=True)
            o, prox = 0, 0.0
            for pm in params:
                n = pm.numel()
                prox = prox + ((pm.reshape(-1) - anchor[o:o + n]) ** 2).sum()
                o += n
            (lam * l_hard + (1 - lam) * l_soft + beta * (mu / 2) * prox).backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
    return seen


def check_client_update(n_rows, batch, epochs, label):
    """One client, end to end, through the production function."""
    cfg = {**CFGD, "batch": batch, "local_epochs": epochs, "lam": LAM, "beta": BETA,
           "mu": MU, "temperature": T, "clip": 1.0, "device": "cpu", "lr": 1e-3}
    torch.manual_seed(7)
    X = torch.randn(n_rows, 66); Y = torch.randint(0, 16, (n_rows,), dtype=torch.uint8)
    ZT = (torch.randn(n_rows, 16) * 3).half()
    base = build_model(cfg).train()
    fkeys, ikeys, n_params = layout(base)
    w0, i0 = flatten(base, fkeys, ikeys)
    anchor = w0[:n_params] + torch.randn(n_params) * 0.02      # a drifted global model
    seed = 12345

    got = build_model(cfg).train(); unflatten_into(got, w0, i0, fkeys, ikeys)
    opt_g = torch.optim.Adam(got.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    gen = torch.Generator(); gen.manual_seed(seed)
    torch.manual_seed(seed)
    ce, kd, gn, sk, nf, nsteps = client_update(
        got, got, opt_g, scaler, X, Y, 0, n_rows,
        torch.cat([anchor, w0[n_params:]]), n_params, ZT, cfg, gen)

    want = build_model(cfg).train(); unflatten_into(want, w0, i0, fkeys, ikeys)
    opt_w = torch.optim.Adam(want.parameters(), lr=cfg["lr"])
    seen = reference_client(want, opt_w, X, Y, 0, n_rows, anchor, ZT, cfg, seed)

    g = torch.cat([p.detach().reshape(-1) for p in got.parameters()])
    w = torch.cat([p.detach().reshape(-1) for p in want.parameters()])
    rel = (g - w).abs().max().item() / (w.abs().max().item() + 1e-12)
    idx = torch.cat(seen)
    per_epoch = sorted(idx[:n_rows].tolist())
    print(f"   {label}: {nsteps} steps (tail {n_rows % batch or batch}), "
          f"skipped={int(sk.item())} nonfinite={int(nf.item())}, "
          f"max|dw|/|w| vs autograd Eq.(6) = {rel:.3e}")
    assert nsteps == epochs * math.ceil(n_rows / batch), nsteps
    assert int(sk.item()) == 0 and int(nf.item()) == 0
    assert per_epoch == list(range(n_rows)), "an epoch did not cover every row exactly once"
    assert len(idx) == epochs * n_rows, "epoch count wrong"
    assert rel < 2e-5, rel


def main():
    torch.manual_seed(42)
    model = build_model(CFGD).train()
    fkeys, ikeys, n_params = layout(model)
    print(f"   layout: {len(fkeys)} float keys, {len(ikeys)} int keys, "
          f"n_params={n_params:,}")
    assert n_params == 395_024, n_params

    x = torch.randn(256, 66); y = torch.randint(0, 16, (256,))
    zt = torch.randn(256, 16) * 3
    anchor = torch.cat([p.detach().reshape(-1) for p in model.parameters()]) \
        + torch.randn(n_params) * 0.05          # a genuinely drifted global model

    # 1. closed-form proximal gradient == the same term through autograd
    ga = grads_of(model, x, y, zt, "autograd", anchor, n_params)
    gc = grads_of(model, x, y, zt, "closed", anchor, n_params)
    ad = (ga - gc).abs().max().item()
    rel = ad / ga.abs().max().item()
    print(f"   proximal closed-form vs autograd: max|delta|={ad:.3e} rel={rel:.3e}")
    assert rel < 1e-6, (ad, rel)

    # 2. the proximal term must NOT touch BatchNorm buffers
    sd = model.state_dict()
    assert all(k in {n for n, _ in model.named_parameters()}
               for k in fkeys[:len([1 for n, _ in model.named_parameters()])])
    assert n_params < sum(sd[k].numel() for k in fkeys), "no float buffers after params"

    # 3. flat round-trip is exact
    model.eval()
    with torch.no_grad(): ref = model(x).clone()
    fv, iv = flatten(model, fkeys, ikeys)
    m2 = build_model(CFGD).eval()
    unflatten_into(m2, fv, iv, fkeys, ikeys)
    with torch.no_grad(): got = m2(x)
    d = (got - ref).abs().max().item()
    print(f"   flat vector round-trip: max|delta logits| = {d}")
    assert d == 0.0, d

    # 4. aggregation is the sample-weighted mean of Eq. (2)
    vs = [(i, n, torch.randn(1000), torch.zeros(0, dtype=torch.long))
          for i, n in enumerate([100, 300, 600])]
    agg, _ = aggregate(vs, 1000)
    want = sum(v[2] * (v[1] / 1000) for v in vs)
    print(f"   aggregate vs weighted mean: max|delta| = {(agg-want).abs().max():.3e}")
    assert torch.allclose(agg, want, atol=1e-6)

    # 5. KD loss equals Eq. (4) written out directly
    zs = torch.randn(64, 16)
    zt2 = torch.randn(64, 16)
    got_kd = (T * T) * F.kl_div(F.log_softmax(zs / T, 1), F.log_softmax(zt2 / T, 1),
                                reduction="batchmean", log_target=True)
    pt = F.softmax(zt2 / T, 1); ps = F.softmax(zs / T, 1)
    want_kd = (T * T) * (pt * (pt.log() - ps.log())).sum(1).mean()   # KL(teacher||student)
    print(f"   L_soft vs Eq.(4) direct: {got_kd.item():.8f} vs {want_kd.item():.8f}")
    assert abs(got_kd - want_kd) < 1e-5, (got_kd, want_kd)

    # 6. client_update, end to end, against Eq. (6) through autograd. Two client sizes,
    #    a partial last batch, and E=1 vs E=2 -- the epoch loop was silently a no-op once.
    check_client_update(700, 256, 1, "client 700 rows / B=256 / E=1")
    check_client_update(517, 128, 2, "client 517 rows / B=128 / E=2")

    # 7. a client that applied no optimizer step must be rejected, not averaged in.
    #    `applied = max(1, ...)` made this branch unreachable for the life of the code.
    ok = [(0, 10, torch.zeros(4), torch.zeros(0, dtype=torch.long)),
          (1, 10, torch.zeros(4), torch.zeros(0, dtype=torch.long))]
    st = {0: {"steps": 5, "applied": 5, "skipped": 0, "nonfinite": 0},
          1: {"steps": 5, "applied": 5, "skipped": 0, "nonfinite": 0}}
    check_updates(1, ok, st, 2)                                   # the healthy case passes
    for label, mutate in (
            ("zero applied steps", lambda s: s[1].update(applied=0, skipped=5)),
            ("non-finite gradient", lambda s: s[1].update(nonfinite=3))):
        bad = {k: dict(v) for k, v in st.items()}; mutate(bad)
        try:
            check_updates(1, ok, bad, 2); raise AssertionError(f"{label} accepted")
        except RuntimeError:
            pass
    nan = [(0, 10, torch.tensor([1., float("nan"), 0., 0.]), torch.zeros(0, dtype=torch.long)),
           ok[1]]
    try:
        check_updates(1, nan, st, 2); raise AssertionError("NaN weights accepted")
    except RuntimeError:
        pass
    try:
        check_updates(1, ok[:1], {0: st[0]}, 2); raise AssertionError("missing client accepted")
    except RuntimeError:
        pass
    print("   check_updates rejects: applied=0, non-finite grads, NaN weights, short round")

    print("\nALL 7 CHECKS PASSED")


main()

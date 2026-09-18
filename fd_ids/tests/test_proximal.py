"""The foreach proximal must equal the loop it replaced, exactly -- and keep equalling it.

Eq. (3)'s gradient is beta*mu*(w - w_G), added to .grad after unscale_. It used to be a
torch.cat over 99 parameters followed by 99 slice-and-add_ calls; it is now two foreach
launches. That is a speed change only if it is a no-op numerically, so this asserts bitwise
equality, not a tolerance.

The optimisation rests on one assumption that is easy to break later: the anchor and
parameter views are built ONCE per client, so p.data must keep its storage across
optimizer steps. Adam updates in place today. Check 3 fails the day something replaces a
parameter's storage instead of writing into it, which would otherwise apply the proximal
term against a stale copy of the weights and corrupt Eq. (3) silently.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/fd-ids-2025"))
from proj.model import build_model, CFG
from proj.fdids import layout, flatten

CFGD = {**CFG, "n_features": 66, "device": "cpu"}
BETA, MU = 0.1, 0.01


def reference(params, anchor, grads):
    """The exact code this replaced, kept here as the oracle."""
    flat = torch.cat([p.detach().reshape(-1) for p in params])
    d = flat - anchor
    o = 0
    for p, g in zip(params, grads):
        n = p.numel()
        g.add_(d[o:o + n].view_as(p), alpha=BETA * MU)
        o += n
    return grads


def foreach(params, anchor, grads):
    """The production form, built the way client_update builds it."""
    pdata, aview, o = [], [], 0
    for p in params:
        pdata.append(p.data)
        aview.append(anchor[o:o + p.numel()].view_as(p)); o += p.numel()
    torch._foreach_add_(grads, torch._foreach_sub(pdata, aview), alpha=BETA * MU)
    return grads


def setup(seed=0):
    torch.manual_seed(seed)
    m = build_model(CFGD).train()
    fk, ik, npar = layout(m)
    w0, _ = flatten(m, fk, ik)
    params = list(m.parameters())
    anchor = w0[:npar] + torch.randn(npar) * 0.02      # a global that differs from w
    grads = [torch.randn_like(p) for p in params]
    return m, params, anchor, grads


def main():
    n = 0

    # 1. Bitwise identical on a fresh model.
    _, params, anchor, g = setup()
    a = reference(params, anchor, [x.clone() for x in g])
    b = foreach(params, anchor, [x.clone() for x in g])
    worst = max(float((x - y).abs().max()) for x, y in zip(a, b))
    assert worst == 0.0, f"foreach differs from the loop by {worst}"
    print(f"1. {len(params)} parameters, max|delta| = {worst}"); n += 1

    # 2. Still identical after the weights have moved away from the anchor.
    m, params, anchor, g = setup(1)
    opt = torch.optim.Adam(params, lr=1e-3)
    for p, x in zip(params, g): p.grad = x.clone()
    opt.step()
    a = reference(params, anchor, [x.clone() for x in g])
    b = foreach(params, anchor, [x.clone() for x in g])
    worst = max(float((x - y).abs().max()) for x, y in zip(a, b))
    assert worst == 0.0, f"after an Adam step, foreach differs by {worst}"
    print(f"2. after an Adam step, max|delta| = {worst}"); n += 1

    # 3. THE ASSUMPTION: views are built once per client, so storage must survive steps.
    m, params, anchor, g = setup(2)
    before = [p.data.data_ptr() for p in params]
    opt = torch.optim.Adam(params, lr=1e-3)
    for _ in range(3):
        for p, x in zip(params, g): p.grad = x.clone()
        opt.step(); opt.zero_grad(set_to_none=True)
    after = [p.data.data_ptr() for p in params]
    assert before == after, "Adam replaced a parameter's storage; cached views are now stale"
    print("3. parameter storage stable across 3 Adam steps -- cached views stay valid"); n += 1

    # 4. A cached view really does see the updated weights (the failure mode check 3 guards).
    m, params, anchor, g = setup(3)
    pdata = [p.data for p in params]
    opt = torch.optim.Adam(params, lr=1e-3)
    for p, x in zip(params, g): p.grad = x.clone()
    opt.step()
    assert all(a.data_ptr() == p.data.data_ptr() and bool((a == p.data).all())
               for a, p in zip(pdata, params)), "cached p.data no longer tracks the parameter"
    print("4. cached p.data tracks the parameter after an update"); n += 1

    print(f"\nALL {n} PROXIMAL CHECKS PASSED")


if __name__ == "__main__":
    main()

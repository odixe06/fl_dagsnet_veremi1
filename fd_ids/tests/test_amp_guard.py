"""AMP overflow is not divergence, and the guard must tell them apart.

Each client starts a fresh GradScaler at 2**16, so the first steps of nearly every client
overflow while it calibrates. The scaler discards those steps and halves the scale; nothing
reaches the weights. Counting them as "non-finite gradient" made check_updates abort a round
in which every client had trained perfectly well -- on T4 that fires in round 1.

What still has to be rejected: a client that applied no step, one that skipped far past
calibration, and one whose APPLIED step carried a non-finite gradient. The last is real:
the proximal term is added after unscale_, so the scaler's own overflow check never sees it.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/fd-ids-2025"))
from proj.model import build_model, CFG
from proj.fdids import layout, flatten, client_update
from proj.driver import check_updates

CFGD = {**CFG, "n_features": 66, "batch": 64, "local_epochs": 1, "lam": 0.5, "beta": 0.1,
        "mu": 0.01, "temperature": 3.0, "clip": 1.0, "device": "cpu", "lr": 1e-3}
N = 256


def run_client(scaler, inf_on_step=None):
    """The production client_update, with an optional Inf injected into one backward."""
    torch.manual_seed(0)
    X = torch.randn(N, 66); Y = torch.randint(0, 16, (N,), dtype=torch.uint8)
    ZT = (torch.randn(N, 16) * 3).half()
    m = build_model(CFGD).train()
    fk, ik, npar = layout(m)
    w0, i0 = flatten(m, fk, ik)
    opt = torch.optim.Adam(m.parameters(), lr=CFGD["lr"])
    h = None
    if inf_on_step is not None:
        box = [0]
        def hook(g, box=box):
            box[0] += 1
            return g * float("inf") if box[0] in inf_on_step else g
        h = m.head[-1].bias.register_hook(hook)
    g = torch.Generator(); g.manual_seed(1)
    ce, kd, gn, sk, nf, nsteps = client_update(m, m, opt, scaler, X, Y, 0, N, w0, npar,
                                               ZT, CFGD, g)
    if h is not None: h.remove()
    fv, _ = flatten(m, fk, ik)
    skipped = int(sk.item())
    return fv, {"steps": nsteps, "applied": nsteps - skipped, "skipped": skipped,
                "nonfinite": int(nf.item())}


def accepts(fv, st, max_skips=16):
    try:
        check_updates(1, [(0, N, fv, torch.zeros(0, dtype=torch.long))], {0: st}, 1,
                      max_skips)
        return True, ""
    except RuntimeError as e:
        return False, str(e)


def main():
    # 1. A real scaler, a real overflow on the first backward, recovery afterwards.
    #    Before the fix this reported nonfinite=1 and the round was aborted.
    fv, st = run_client(torch.amp.GradScaler("cpu", enabled=True, init_scale=2.0 ** 16),
                        inf_on_step={1})
    print(f"   overflow+recovery: {st}")
    assert st["skipped"] == 1 and st["applied"] == st["steps"] - 1, st
    assert st["nonfinite"] == 0, "a scaler-skipped overflow was counted as divergence"
    assert torch.isfinite(fv).all(), "weights should be finite: the step was discarded"
    ok, why = accepts(fv, st)
    assert ok, f"a healthy client was rejected: {why}"

    # 2. A clean client with no scaler at all.
    fv2, st2 = run_client(torch.amp.GradScaler("cpu", enabled=False))
    print(f"   clean, scaler off: {st2}")
    assert st2 == {"steps": 4, "applied": 4, "skipped": 0, "nonfinite": 0}, st2
    assert accepts(fv2, st2)[0]

    # 3. Scaler OFF and an Inf gradient: nothing skips it, so it DID reach the weights.
    #    clip_grad_norm_ turns an infinite total norm into clip_coef 0 and Inf*0 = NaN, so
    #    the weights break too and the weight check -- deliberately first, being the more
    #    severe finding -- reports it. The counter is the diagnostic, not the only defence;
    #    case 7 exercises it alone, against finite weights.
    fv3, st3 = run_client(torch.amp.GradScaler("cpu", enabled=False), inf_on_step={1})
    print(f"   applied a non-finite gradient: {st3}")
    assert st3["nonfinite"] >= 1 and st3["skipped"] == 0, st3
    assert not torch.isfinite(fv3).all(), "an applied Inf should have reached the weights"
    ok, why = accepts(fv3, st3)
    assert not ok and "non-finite weights" in why, why

    # 4-7. The synthetic rejections, on the same guard.
    base = {"steps": 100, "applied": 100, "skipped": 0, "nonfinite": 0}
    good = torch.zeros(4)
    for name, st, needle in (
            ("every step skipped", {**base, "applied": 0, "skipped": 100}, "zero optimizer"),
            ("skips past the budget", {**base, "applied": 83, "skipped": 17}, "warm-up budget"),
            ("step arithmetic does not close", {**base, "applied": 50, "skipped": 10},
             "applied+skipped"),
            ("applied a non-finite gradient", {**base, "nonfinite": 2}, "APPLIED")):
        ok, why = accepts(good, st)
        assert not ok and needle in why, f"{name}: {ok} {why}"
        print(f"   rejected: {name}")

    # 8. Exactly at the budget is still fine; one over is not.
    assert accepts(good, {**base, "applied": 84, "skipped": 16})[0], "16 skips is the budget"
    assert not accepts(good, {**base, "applied": 83, "skipped": 17})[0]
    print("   budget boundary: 16 accepted, 17 rejected")

    print("\nALL 8 AMP-GUARD CHECKS PASSED")


if __name__ == "__main__":
    main()

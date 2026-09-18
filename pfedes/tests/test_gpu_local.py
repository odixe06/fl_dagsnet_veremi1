"""Real torch.compile(reduce-overhead) on the local GPU (sm_86): both pFedES steps and the
eval template under CUDA graphs, graph reuse across clients, tail-batch fallback, AMP
skip accounting, and compiled-vs-eager agreement. sm_86 passing does NOT prove sm_75; it
proves the graph scheme (4 captured graphs, requires_grad/mode toggles) is sound."""
import json, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/pfedes-yi-2025"))
import tests.test_smoke_real as S
from proj.model import build_model, build_proxy, CFG
from proj.pfedes import layout, flatten, unflatten_into, client_update
from proj.evaluate import fold_bn, load_folded, eval_model
from proj import driver as D

S.ROWS_PER_CLIENT, S.TEST_ROWS = 3_000, 12_000


def train_one(Fc, Fe, Gc, Ge, X, Y, lo, hi, cfg, dev, seed=7):
    optF = torch.optim.AdamW(Fe.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], fused=True)
    optG = torch.optim.AdamW(Ge.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], fused=True)
    scF = torch.amp.GradScaler("cuda"); scG = torch.amp.GradScaler("cuda")
    scF.scale(torch.zeros(1, device=dev)); scG.scale(torch.zeros(1, device=dev))
    torch.manual_seed(seed); g = torch.Generator(device=dev); g.manual_seed(seed)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    a1, n1, a2, n2 = client_update(Fc, Fe, Gc, Ge, optF, optG, scF, scG, X, Y, lo, hi, cfg, g)
    torch.cuda.synchronize()
    return a1, n1, a2, n2, (time.perf_counter() - t0) / (n1 + n2) * 1000


def main():
    assert torch.cuda.is_available(), "needs the local GPU"
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    for _n in ('recompile_limit', 'cache_size_limit'):
        if hasattr(torch._dynamo.config, _n): setattr(torch._dynamo.config, _n, 64)
    tmp = Path(tempfile.mkdtemp()); cache = tmp / "cache"; cache.mkdir()
    spans, n_test = S.prepack(cache, [0, 1])
    cfg = S.make_cfg(cache, 2, n_test, rounds=1)
    cfg.update(device="cuda", compile=True, batch=256, eval_batch=4096)
    X = D._resident(cache / "train_X.f16.npy", dev); Y = D._resident(cache / "train_y.u8.npy", dev)
    TX = D._resident(cache / "test_X.f16.npy", dev); TY = D._resident(cache / "test_y.u8.npy", dev)
    B = cfg["batch"]

    torch.manual_seed(42)
    Fe = build_model(cfg).to(dev).train(); Ge = build_proxy(cfg).to(dev).train()
    fk, ik, _ = layout(Fe); gk, gik, _ = layout(Ge)
    f0, i0 = flatten(Fe, fk, ik); g0, gi0 = flatten(Ge, gk, gik)

    # 1. the real gate on real rows
    Fc, Gc = D._compile_train(Fe, Ge, cfg, dev, X[:B].float(), Y[:B].long())
    assert Fc is not Fe and Gc is not Ge, "compile gate fell back to eager on sm_86"
    print("1. train gate certified with real torch.compile(reduce-overhead)")

    # 2. two clients back to back on the compiled modules (graph reuse, tail batch, AMP)
    stats = []
    for cid in (0, 1):
        unflatten_into(Fe, f0.to(dev), i0.to(dev), fk, ik); unflatten_into(Ge, g0.to(dev), gi0.to(dev), gk, gik)
        a1, n1, a2, n2, ms = train_one(Fc, Fe, Gc, Ge, X, Y, *spans[cid], cfg, dev)
        s = dict(cid=cid, steps=(n1, n2), skips=(int(a1["skips"]), int(a2["skips"])),
                 loss_w=float(a1["loss"]) / max(1, n1 - int(a1["skips"])),
                 loss_theta=float(a2["loss"]) / max(1, n2 - int(a2["skips"])), ms_per_step=ms)
        assert n1 == n2 == 12 and all(np.isfinite([s["loss_w"], s["loss_theta"]])), s
        assert int(a1["nonfin"]) == 0 and int(a2["nonfin"]) == 0
        fv, _ = flatten(Fe, fk, ik); gv, _ = flatten(Ge, gk, gik)
        assert torch.isfinite(fv).all() and torch.isfinite(gv).all()
        stats.append(s); print(f"2. client {cid}: {s}")
    comp_w = flatten(Fe, fk, ik)[0].clone(); comp_g = flatten(Ge, gk, gik)[0].clone()

    # 3. compiled vs eager on the same client from the same weights with dropout off:
    #    the two paths must land within fp16 accumulation distance of each other
    for m in list(Fe.modules()) + list(Ge.modules()):
        if isinstance(m, torch.nn.Dropout): m.p = 0.0
    outs = []
    for Fm, Gm in ((Fc, Gc), (Fe, Ge)):
        unflatten_into(Fe, f0.to(dev), i0.to(dev), fk, ik); unflatten_into(Ge, g0.to(dev), gi0.to(dev), gk, gik)
        a1, n1, a2, n2, ms = train_one(Fm, Fe, Gm, Ge, X, Y, *spans[1], cfg, dev)
        outs.append((flatten(Fe, fk, ik)[0].clone(), flatten(Ge, gk, gik)[0].clone(),
                     float(a1["loss"]), float(a2["loss"]), ms))
    dw = (outs[0][0] - outs[1][0]).abs().max().item(); dg = (outs[0][1] - outs[1][1]).abs().max().item()
    rel = dw / (outs[1][0] - f0.to(dev)).abs().max().item()
    print(f"3. compiled vs eager (dropout 0): max|dw|={dw:.2e} max|dtheta|={dg:.2e} "
          f"(rel. to the update {rel:.2e}) | loss_w {outs[0][2]:.4f} vs {outs[1][2]:.4f} | "
          f"{outs[0][4]:.2f} vs {outs[1][4]:.2f} ms/step -> {outs[1][4]/outs[0][4]:.2f}x")
    assert rel < 0.5 and abs(outs[0][2] - outs[1][2]) / outs[1][2] < 5e-2, (rel, outs[0][2], outs[1][2])
    for m, p in zip([m for m in list(Fe.modules()) + list(Ge.modules()) if isinstance(m, torch.nn.Dropout)], [0.1]*4):
        m.p = p

    # 4. eval template: compile, gate, run on the whole test with a tail batch
    Te = fold_bn(build_model(cfg).to(dev))
    Tc = D._compile_eval(Te, cfg, dev, TX[:cfg["eval_batch"]].float())
    assert Tc is not Te, "eval gate fell back to eager on sm_86"
    unflatten_into(Fe, comp_w, i0.to(dev), fk, ik)
    load_folded(Te, Fe)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    cm_c, nf, preds = eval_model(Tc, Te, TX, TY, cfg, want_preds=True)
    torch.cuda.synchronize(); tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    cm_e, nf2, preds2 = eval_model(Te, Te, TX, TY, cfg, want_preds=True)
    torch.cuda.synchronize(); te = time.perf_counter() - t0
    assert cm_c.sum().item() == n_test == cm_e.sum().item() and nf == nf2 == 0
    dis = int((preds != preds2).sum())
    print(f"4. eval: compiled {n_test/tc:,.0f} rows/s vs eager-folded {n_test/te:,.0f} rows/s "
          f"({te/tc:.2f}x); {dis} of {n_test} predictions differ ({dis/n_test:.4%})")
    assert dis / n_test < 1e-3
    # a second client through the SAME compiled template (weights swapped in place)
    unflatten_into(Fe, f0.to(dev), i0.to(dev), fk, ik); load_folded(Te, Fe)
    cm2, _, _ = eval_model(Tc, Te, TX, TY, cfg)
    assert cm2.sum().item() == n_test and not torch.equal(cm2, cm_c)
    print("5. second client through the same compiled eval template: different, complete CM")
    print(f"   peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    shutil.rmtree(tmp)
    print("\nLOCAL GPU COMPILE TEST PASSED (sm_86; not evidence for sm_75)")


if __name__ == "__main__":
    main()

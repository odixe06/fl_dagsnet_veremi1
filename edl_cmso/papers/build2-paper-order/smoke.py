#!/usr/bin/env python
"""Local pre-flight for build 2. Run in the `nckh` conda env before any Kaggle push:

    source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
    python papers/build2-paper-order/smoke.py

Every check here exists because getting it wrong costs a GPU session, not a test run.
The two that matter most are marked CONTRACT (the mask and the trained network must refer
to the same §4.8 projection) and DEADLOCK (a mask that starves the GAT hangs DDP with no
traceback).
"""
import json, re, sys, tempfile, shutil
from pathlib import Path

import numpy as np
import torch

import os
NB = Path(__file__).parent / os.environ.get("EDL_NB", "notebook-final-wd5e2/edl_cmso_v2_final.ipynb")
ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1; print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
    else:
        fail += 1; print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))


def extract_modules(dst):
    """Pull every %%writefile cell out of the notebook into an importable proj/ package."""
    nb = json.loads(NB.read_text())
    pkg = dst / "proj"; pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    written = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        m = re.match(r"%%writefile /kaggle/working/proj/(\w+\.py)\n", src)
        if m:
            (pkg / m.group(1)).write_text(src[m.end():])
            written.append(m.group(1))
    return written


CFG = dict(patch_len=6, d_model=128, vit_layers=2, vit_heads=4, vit_mlp_ratio=2,
           gat_heads=4, gat_slope=0.2, dropout=0.1, stem_ch=96, dense_growth=32,
           dense_layers=3, incep_modules=2, fire_modules=3, num_classes=16, seed=42,
           cmso_pop=20, cmso_iters=50, cmso_C1=0.2, cmso_mu=25.0, cmso_sigma=3.0,
           cmso_mutation=0.05, cmso_size_penalty=0.01)
N_FEAT, FUSED = 66, 6 + 2 * 128


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        mods = extract_modules(tmp)
        sys.path.insert(0, str(tmp))
        print(f"extracted {len(mods)} modules from the notebook: {', '.join(sorted(mods))}\n")

        import proj.model as M
        import proj.cmso as CM

        # ── 1. Eq. (19) ──────────────────────────────────────────────────────────
        print("1. HaarDWT — Eq. (19)")
        dwt = M.HaarDWT()
        x = torch.randn(64, 66)
        xw = dwt(x)
        check("output length preserved at even n", xw.shape == (64, 66), f"{tuple(xw.shape)}")
        check("orthonormal: energy preserved",
              torch.allclose(xw.pow(2).sum(1), x.pow(2).sum(1), atol=1e-4))
        check("odd n is padded to n+1", dwt(torch.randn(8, 41)).shape == (8, 42))

        # ── 2. Eq. (20)-(21), and decision C ─────────────────────────────────────
        print("\n2. PatchEmbed — Eq. (20)-(21), raw patch returned for Eq. (28)")
        pe = M.PatchEmbed(66, 6, 128)
        check("66 = 11 x 6 exactly, no padding", (pe.k, pe.pad) == (11, 0), f"k={pe.k} pad={pe.pad}")
        xp, zp = pe(xw)
        check("raw patch shape", xp.shape == (64, 11, 6), f"{tuple(xp.shape)}")
        check("raw patch IS the wavelet coefficients, not a projection",
              torch.equal(xp, xw.view(64, 11, 6)))
        check("embedded token shape", zp.shape == (64, 11, 128), f"{tuple(zp.shape)}")

        # ── 3. Eq. (28) ──────────────────────────────────────────────────────────
        print("\n3. Fusion — Eq. (28) read literally")
        torch.manual_seed(0)
        full = M.build_model(CFG, N_FEAT, sel_ch=None).eval()
        check("fused_dim = P + 2d", full.fused_dim == FUSED, f"{full.fused_dim} = 6 + 128 + 128")
        with torch.no_grad():
            F0 = full.fuse(x)
        check("fuse shape (B, fused, k)", F0.shape == (64, FUSED, 11), f"{tuple(F0.shape)}")
        with torch.no_grad():
            expect = dwt_patch = full.dwt(x).view(64, 11, 6).transpose(1, 2)
        check("channels 0:6 are the raw wavelet term",
              torch.allclose(F0[:, :6], expect, atol=1e-5))

        # ── 4. forward/backward under a mask; every parameter must get gradient ──
        print("\n4. Masked forward/backward — DDP needs every parameter to get gradient")
        rng = np.random.default_rng(0)
        for n_sel in (FUSED, 131, 40, 16):
            sel = sorted(rng.choice(FUSED, n_sel, replace=False).tolist())
            for a, b in ((0, 6), (6, 134), (134, FUSED)):          # keep all three terms
                if not any(a <= s < b for s in sel):
                    sel = sorted(set(sel) | {a})
            torch.manual_seed(0)
            mdl = M.build_model(CFG, N_FEAT, sel_ch=sel)
            out = mdl(torch.randn(8, 66))
            out.sum().backward()
            dead = [n for n, p in mdl.named_parameters() if p.grad is None]
            check(f"|S|={len(sel):3d}: logits (8,16), all {sum(1 for _ in mdl.parameters())} "
                  f"params have grad", out.shape == (8, 16) and not dead,
                  f"{sum(p.numel() for p in mdl.parameters()):,} params"
                  + (f"  DEAD: {dead[:4]}" if dead else ""))

        # ── 5. FROZEN BRANCH: the failure the per-term floor exists to prevent ──
        # Not a DDP hang. cat + index_select keep gat.* in the autograd graph, so a
        # GAT-free mask gives them a ZERO gradient, not None — find_unused_parameters=False
        # stays valid. The cost is that the branch never moves off its random init while
        # still paying a forward pass every step. Both halves are asserted here so the
        # claim in the notebook prose stays measured rather than assumed.
        print("\n5. FROZEN-BRANCH guard — a mask with no GAT channel")
        torch.manual_seed(0)
        bad = M.build_model(CFG, N_FEAT, sel_ch=list(range(0, 134)))   # wavelet + ViT only
        bad(torch.randn(8, 66)).sum().backward()
        gat_g = {n: p.grad for n, p in bad.named_parameters() if n.startswith("gat.")}
        check("DDP is safe: gat params stay in the graph (grad is not None)",
              len(gat_g) == 3 and all(g is not None for g in gat_g.values()),
              f"{len(gat_g)} gat tensors")
        check("but the branch is frozen: every gat gradient is exactly zero",
              all(bool((g == 0).all()) for g in gat_g.values()),
              f"absmax {max(float(g.abs().max()) for g in gat_g.values()):.1e}")
        others = [n for n, p in bad.named_parameters()
                  if not n.startswith("gat.") and (p.grad is None or bool((p.grad == 0).all()))]
        check("no other parameter is frozen by that mask", not others, f"{others[:4]}")

        bounds = [(0, 6), (6, 134), (134, FUSED)]
        worst = np.full(FUSED, -8.0); worst[:134] = 8.0      # sigmoid drops every GAT channel
        m = CM.binarise(worst, bounds)
        check("binarise floor rescues it", m[134:].sum() >= CM.MIN_PER_TERM,
              f"gat kept {int(m[134:].sum())}, wavelet {int(m[:6].sum())}, "
              f"vit {int(m[6:134].sum())}")
        for trial in range(500):                              # never violated, over 500 draws
            mm = CM.binarise(rng.normal(0, 3, FUSED), bounds)
            if not (mm[:6].sum() >= CM.MIN_PER_TERM and mm[6:134].sum() >= CM.MIN_PER_TERM
                    and mm[134:].sum() >= CM.MIN_PER_TERM and mm.sum() >= CM.MIN_CHANNELS):
                break
        else:
            trial = None
        check("500 random candidates all honour both floors", trial is None,
              "" if trial is None else f"violated at draw {trial}")

        # ── 6. CONTRACT: mask and trained network share one §4.8 projection ─────
        print("\n6. CONTRACT — extractor_init.pt pins the projection CMSO selected against")
        torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
        probe = M.build_model(CFG, N_FEAT, sel_ch=None).eval()
        with torch.no_grad():
            F_probe = probe.fuse(x)
        sd = M.extractor_state(probe)
        check("extractor_state covers only §4.8",
              all(k.startswith(M.EXTRACTOR_PREFIXES) for k in sd)
              and not any(k.startswith(("stems", "dense", "google", "alex", "squeeze", "head"))
                          for k in sd),
              f"{len(sd)} tensors, {sum(v.numel() for v in sd.values()):,} params")

        torch.manual_seed(999)                       # deliberately a DIFFERENT init
        trainee = M.build_model(CFG, N_FEAT, sel_ch=list(range(0, FUSED, 2))).eval()
        with torch.no_grad():
            drift = (trainee.fuse(x) - F_probe).abs().max().item()
        check("without the load, §4.8 is a different projection", drift > 1e-3,
              f"max|Δ| = {drift:.4f}")
        M.load_extractor_state(trainee, sd)
        with torch.no_grad():
            F_after = trainee.fuse(x)
        check("after load_extractor_state, fuse() is bit-identical",
              torch.equal(F_after, F_probe),
              f"max|Δ| = {(F_after - F_probe).abs().max().item():.2e}")
        try:
            M.load_extractor_state(trainee, {"nonexistent.weight": torch.zeros(1)})
            check("a foreign key is rejected", False)
        except AssertionError:
            check("a foreign key is rejected", True)

        # ── 7. CMSO optimiser logic, on a planted answer ────────────────────────
        print("\n7. CMSO — Eq. (29)-(37) on a problem with a known answer")
        planted = np.zeros(FUSED, bool)
        planted[[1, 3, 10, 40, 77, 140, 200, 255]] = True     # spans all three terms

        def planted_fitness(mask):
            hit = (mask & planted).sum() / planted.sum()
            noise = (mask & ~planted).sum() / (~planted).sum()
            return float(hit - 0.35 * noise - CFG["cmso_size_penalty"] * mask.sum() / FUSED)

        best, hist = CM.run_cmso(planted_fitness, FUSED, CFG, term_bounds=bounds,
                                 log=lambda *a, **k: None)
        f_first, f_last = hist[0]["best_fitness"], hist[-1]["best_fitness"]
        check("fitness improves over T iterations", f_last > f_first,
              f"{f_first:.4f} -> {f_last:.4f}")
        check("monotone (run_cmso only ever replaces the global best upward)",
              all(b["best_fitness"] >= a["best_fitness"]
                  for a, b in zip(hist, hist[1:])))
        recall = (best & planted).sum() / planted.sum()
        check("recovers the planted channels", recall >= 0.75,
              f"{int((best & planted).sum())}/{int(planted.sum())} planted, "
              f"|S| = {int(best.sum())}/{FUSED}")
        check("best mask honours the per-term floor",
              all(best[a:b].sum() >= CM.MIN_PER_TERM for a, b in bounds),
              f"wavelet {int(best[:6].sum())}, vit {int(best[6:134].sum())}, "
              f"gat {int(best[134:].sum())}")

        # ── 8. the surrogate the fitness actually uses ──────────────────────────
        print("\n8. SurrogateCNN — the fitness evaluator's shapes")
        for cin in (16, 131, FUSED):
            s = M.SurrogateCNN(cin, 16)
            o = s(torch.randn(32, cin, 11))
            check(f"cin={cin:3d}: (32,16) logits", o.shape == (32, 16),
                  f"{sum(p.numel() for p in s.parameters()):,} params")

        # ── 9. AMP path ────────────────────────────────────────────────────────
        print("\n9. autocast — fp16 on T4, softmax forced back to fp32")
        sel = list(range(0, FUSED, 2))
        torch.manual_seed(0)
        mdl = M.build_model(CFG, N_FEAT, sel_ch=sel).eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            lo = mdl(torch.randn(8, 66))
        pr = torch.softmax(lo.float(), 1)
        check("logits finite under autocast", torch.isfinite(lo).all().item())
        check("softmax rows sum to 1 in fp32",
              torch.allclose(pr.sum(1), torch.ones(8), atol=1e-5))

        # ── 10. shape agreement with the launch-cell asserts ───────────────────
        print("\n10. launch-cell invariants")
        check("k is 11 for the full input, every run", M.PatchEmbed(66, 6, 128).k == 11)
        check("fused = 262 matches CFG arithmetic", 6 + 2 * 128 == FUSED, str(FUSED))
        check("MIN_CHANNELS <= any admissible |S|", CM.MIN_CHANNELS >= 3 * CM.MIN_PER_TERM,
              f"MIN_CHANNELS={CM.MIN_CHANNELS}, 3 x MIN_PER_TERM={3*CM.MIN_PER_TERM}")

        # ── 11. the CMSO cell's chunked extraction ─────────────────────────────
        # Cell 16 fills Fbuf in 16k-row chunks off a fancy-indexed memmap. An off-by-one
        # there would corrupt CMSO's input silently — the search would still "converge",
        # just on the wrong rows.
        print("\n11. chunked F extraction — cell 16's loop")
        torch.manual_seed(CFG["seed"])
        pm = M.build_model(CFG, N_FEAT, sel_ch=None).eval()
        store = np.random.default_rng(1).standard_normal((997, 66)).astype(np.float16)
        take = np.sort(np.random.default_rng(2).choice(997, 500, replace=False))
        buf = torch.empty((len(take), FUSED, 11), dtype=torch.float16)
        with torch.no_grad():
            for i in range(0, len(take), 64):                 # the cell's chunk loop
                j = min(i + 64, len(take))
                buf[i:j] = pm.fuse(torch.from_numpy(store[take[i:j]]).float()).half()
            one = pm.fuse(torch.from_numpy(store[take]).float()).half()
        # fp16 matmul reduces in a different order at a different batch size, so the two
        # paths agree to fp16 rounding, not bit-for-bit. The tolerance is set where noise
        # ends and misalignment begins — the shifted control below proves it has teeth.
        d = (buf.float() - one.float()).abs().max().item()
        check("chunked == single pass, to fp16 rounding", d < 5e-3,
              f"{tuple(buf.shape)}, max|Δ| = {d:.1e}")
        shifted = (buf[1:].float() - one[:-1].float()).abs().max().item()
        check("a one-row shift would NOT pass that tolerance", shifted > 5e-3,
              f"max|Δ| = {shifted:.2f}")
        nf = 400                                  # cell 16 splits Fbuf[:nf] / Fbuf[nf:]
        check("fit/val slices partition the subsample exactly, no overlap, no gap",
              torch.equal(torch.cat([buf[:nf], buf[nf:]]), buf)
              and buf[:nf].shape[0] + buf[nf:].shape[0] == len(take),
              f"{nf} fit + {len(take)-nf} val = {len(take)}")
        check("channel axis is dim 1, ready for Conv1d and index_select",
              buf.shape[1] == FUSED and buf.shape[2] == 11)

        # ── 12. no unbound name at notebook top level ──────────────────────────
        # A NameError in a top-level cell surfaces AFTER ~17 min of parquet prep and
        # costs the whole GPU session. Build 2 shipped exactly one (autocast in cell 16)
        # and this check is why it never reached Kaggle.
        print("\n12. static scan — every top-level name is bound before use")
        import ast as _ast
        nbj = json.loads(NB.read_text())
        bound, unbound = set(dir(__builtins__)) | {"display", "get_ipython"}, {}
        for i, c in enumerate(nbj["cells"]):
            if c["cell_type"] != "code":
                continue
            s = "".join(c["source"])
            if s.startswith("%%writefile"):
                continue                      # runs in its own module namespace
            body = "\n".join(l for l in s.split("\n") if not l.startswith(("%", "!")))
            tree = _ast.parse(body)
            for n in _ast.walk(tree):
                if isinstance(n, _ast.Import):
                    for a in n.names: bound.add((a.asname or a.name).split(".")[0])
                elif isinstance(n, _ast.ImportFrom):
                    for a in n.names: bound.add(a.asname or a.name)
                elif isinstance(n, (_ast.FunctionDef, _ast.ClassDef)): bound.add(n.name)
                elif isinstance(n, _ast.Name) and isinstance(n.ctx, _ast.Store): bound.add(n.id)
                elif isinstance(n, _ast.arg): bound.add(n.arg)
            for n in _ast.walk(tree):
                if isinstance(n, _ast.Name) and isinstance(n.ctx, _ast.Load) and n.id not in bound:
                    unbound.setdefault(n.id, i)
        check("no top-level cell reads a name nothing has bound", not unbound,
              f"{len(nbj['cells'])} cells scanned" if not unbound
              else "  ".join(f"cell {v}: {k}" for k, v in unbound.items()))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'='*70}\n{ok} passed, {fail} failed\n{'='*70}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

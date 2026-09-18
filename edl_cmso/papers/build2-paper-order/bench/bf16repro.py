"""Why does the TPU run blow up at step ~250-500 when the GPU run survives ~147,000?

Both use batch 4,096, lr 1e-3 and the same frozen extractor. The differences are the
precision (bf16 on XLA, fp16 on CUDA) and the guard (clip-to-norm on XLA, GradScaler on
CUDA). Hypothesis: fp16's 65,504 ceiling was an ACCIDENTAL TRIPWIRE. An exploding
attention gradient overflows fp16, GradScaler drops the step and reports nothing, and the
run keeps training on the steps that stayed in range. bf16 reaches 3.39e38, so nothing
overflows, the bad gradient is finite, the clip-guard scales it to norm 1 -- and Adam is
scale-invariant to a uniform rescale, so the corrupted DIRECTION is applied in full.

If that is right, the bf16 arm reproduces the TPU blow-up locally and the fp16 arm does
not, with the difference visible in the skip count.

Runs on the 4 GB RTX 3050 at batch 1,024 with 300,000 real rows (~0.8 GB of the 8 GB box).
Zero Kaggle quota.
"""
import sys, math, time, json, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R

dev, B, STEPS, EVERY = "cuda", 1024, 900, 50
CLIP, LR, WD = 1.0, 1e-3, 5e-2                       # the fast50 CFG
X, Y = R.load_train(300_000, seed=11)
crit = torch.nn.CrossEntropyLoss()
print(f"rows {len(X):,}  classes {len(np.unique(Y))}/16  batch {B}  steps {STEPS}")


def probe(m, xb):
    """|logit|max and softmax entropy inside vit.0 -- computed in fp32, like the notebook."""
    blk = m.vit[0]
    with torch.no_grad():
        _, z = m.patch(m.dwt(xb.float())); h = blk.n1(z)
        qkv = torch.nn.functional.linear(h, blk.att.in_proj_weight, blk.att.in_proj_bias)
        d = z.shape[-1]; H = blk.att.num_heads; hd = d // H
        q, k, _ = qkv.chunk(3, -1)
        q = q.view(xb.shape[0], -1, H, hd).transpose(1, 2)
        k = k.view(xb.shape[0], -1, H, hd).transpose(1, 2)
        lg = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        p = lg.softmax(-1)
        return float(lg.abs().max()), float(-(p * (p + 1e-12).log()).sum(-1).mean())


def arm(name, dtype, guard, gmax=None):
    """guard: 'scaler' = fp16 GradScaler (CUDA run), 'clip' = clip-to-norm (XLA run),
    'clip+thresh' = clip-to-norm plus a skip when the norm exceeds gmax."""
    torch.manual_seed(0); np.random.seed(0)
    m, _ = R.build(dev, load_init=True); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sc = torch.amp.GradScaler("cuda", enabled=(guard == "scaler"))
    W = m.vit[0].att.in_proj_weight
    rows, skipped, t0 = [], 0, time.time()
    print(f"\n=== {name} ===")
    print(f"  {'step':>5} {'loss':>8} {'grad':>11} {'|logit|max':>11} {'H':>7} "
          f"{'||Wqkv||':>9} {'max|w|':>7} {'skip':>5}")
    for s in range(STEPS + 1):
        i = (s * B) % (len(X) - B)
        xb = torch.from_numpy(X[i:i+B]).to(dev); yb = torch.from_numpy(Y[i:i+B]).to(dev)
        if s % EVERY == 0:
            lg, ent = probe(m, xb)
            with torch.no_grad():
                mw = float(max(p.abs().max() for p in m.parameters()))
            rows.append({"step": s, "abs_logit_max": lg, "entropy": ent,
                         "Wqkv": float(W.norm()), "max_w": mw, "skipped": skipped})
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype is not torch.float32):
            logits = m(xb)
        loss = crit(logits.float(), yb)
        if guard == "scaler":
            sc.scale(loss).backward()
            sc.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
            prev = sc.get_scale(); sc.step(opt); sc.update()
            if sc.get_scale() < prev:                  # GradScaler dropped this step
                skipped += 1
        else:
            loss.backward()
            grads = [p.grad for p in m.parameters() if p.grad is not None]
            gn = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
            ok = torch.isfinite(gn)
            if gmax is not None:
                ok = ok & (gn < gmax)
            scale = torch.where(ok, torch.clamp(CLIP / (gn + 1e-6), max=1.0),
                                torch.zeros_like(gn))
            for gr in grads:
                gr.nan_to_num_().mul_(scale)
            skipped += int(not bool(ok))
            opt.step()
        if s % EVERY == 0:
            rows[-1].update(loss=float(loss), grad=float(gn))
            r = rows[-1]
            print(f"  {r['step']:5d} {r['loss']:8.4f} {r['grad']:11.3e} "
                  f"{r['abs_logit_max']:11.3e} {r['entropy']:7.4f} {r['Wqkv']:9.3f} "
                  f"{r['max_w']:7.3f} {r['skipped']:5d}")
    print(f"  ({time.time()-t0:.0f}s, {skipped} steps skipped of {STEPS})")
    return rows


out = {}
out["bf16 + clip  (the TPU run)"]  = arm("bf16 + clip  (the TPU run)", torch.bfloat16, "clip")
out["fp16 + GradScaler (the GPU run)"] = arm("fp16 + GradScaler (the GPU run)", torch.float16, "scaler")
out["bf16 + clip + skip if grad>1e3"] = arm("bf16 + clip + skip if grad>1e3", torch.bfloat16,
                                            "clip+thresh", gmax=1e3)
out["fp32 + clip  (control)"] = arm("fp32 + clip  (control)", torch.float32, "clip")

p = "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench/bf16repro.json"
open(p, "w").write(json.dumps(out, indent=2))
print(f"\nwrote {p}")
print(f"\n{'arm':<34} {'|logit|max end':>15} {'grad end':>11} {'loss end':>9} {'skipped':>8}")
for k, v in out.items():
    print(f"{k:<34} {v[-1]['abs_logit_max']:15.3e} {v[-1]['grad']:11.3e} "
          f"{v[-1]['loss']:9.4f} {v[-1]['skipped']:8d}")

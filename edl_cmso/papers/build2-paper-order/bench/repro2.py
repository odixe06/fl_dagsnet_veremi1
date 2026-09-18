"""Is the round-24 model actually diverging, or only overflowing fp16?

Same weights, same real batches, three regimes:
  fp32          — the true gradient, no scaling anywhere
  fp16 autocast — the run's regime, gradient measured after unscaling
Compared against an early, healthy checkpoint so the numbers have a baseline.
"""
import sys, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R

dev, B, NB = "cuda", 4096, 12
X, Y = R.load_train(B * NB, seed=11)
NB = len(X) // B
crit = torch.nn.CrossEntropyLoss()

def probe(rnd):
    m, _ = R.build(dev, load_init=False)
    ck = torch.load(R.RUN / "checkpoints" / f"ckpt_round_{rnd:03d}.pt",
                    map_location="cpu", weights_only=False)
    m.load_state_dict({k.replace("module.", ""): v for k, v in ck["model"].items()})
    m.train()
    w = torch.cat([p.detach().flatten() for p in m.parameters()])
    out = {"round": rnd, "max|w|": float(w.abs().max()), "||w||": float(w.norm())}

    g32, g16, act, losses = [], [], [], []
    for i in range(NB):
        xb = torch.from_numpy(X[i*B:(i+1)*B]).to(dev)
        yb = torch.from_numpy(Y[i*B:(i+1)*B]).to(dev)
        # --- true gradient, fp32 end to end ---
        m.zero_grad(set_to_none=True)
        logits = m(xb)
        l = crit(logits, yb); l.backward()
        gn = torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad) for p in m.parameters() if p.grad is not None]))
        g32.append(float(gn)); losses.append(float(l.detach()))
        act.append(float(logits.detach().abs().max()))
        # --- same batch, fp16 autocast, gradient unscaled by hand ---
        m.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lo = m(xb)
        crit(lo.float(), yb).backward()
        gn16 = torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad.float()) for p in m.parameters()
             if p.grad is not None]))
        g16.append(float(gn16))
    m.zero_grad(set_to_none=True)
    out.update({"loss": np.mean(losses), "max|logit|": max(act),
                "grad_fp32": np.mean(g32), "grad_fp32_max": max(g32),
                "grad_fp16_nonfinite": int(sum(not np.isfinite(v) for v in g16)),
                "grad_fp16": np.nanmean([v for v in g16 if np.isfinite(v)])
                             if any(np.isfinite(v) for v in g16) else float("nan")})
    del m; torch.cuda.empty_cache()
    return out

hdr = f"{'round':>5} {'loss':>8} {'max|w|':>8} {'max|logit|':>11} {'grad fp32':>11} {'grad fp32 max':>14} {'fp16 nonfin':>12}"
print(hdr); print("-" * len(hdr))
for rnd in (0, 5, 15, 21, 23, 24):
    r = probe(rnd)
    print(f"{r['round']:5d} {r['loss']:8.4f} {r['max|w|']:8.3f} {r['max|logit|']:11.1f} "
          f"{r['grad_fp32']:11.3f} {r['grad_fp32_max']:14.3f} {r['grad_fp16_nonfinite']:>7}/{NB}")

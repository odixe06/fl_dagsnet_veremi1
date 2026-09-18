"""Where inside epoch 0 do non-finite gradients start, and WHICH parameter goes first?

The TPU run reported 66-72% of round 0's 10,509 steps skipped, with steps_ok=3563 — which
points at an onset a few thousand steps in. onset.py only ran 2,000 steps and saw nothing,
so it was measuring too short a window. This runs long enough to cross it.
"""
import sys, math, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R
dev, B, STEPS, WIN = "cuda", 4096, 9000, 500
X, Y = R.load_train(300_000, seed=5)
NB = len(X) // B
print(f"{len(X):,} rows, {NB} batches cycled, {STEPS} steps at batch {B}", flush=True)
crit = torch.nn.CrossEntropyLoss()

def attn_stats(m, xb):
    blk = m.vit[0]
    with torch.no_grad():
        _, z = m.patch(m.dwt(xb)); h = blk.n1(z)
        qkv = torch.nn.functional.linear(h, blk.att.in_proj_weight, blk.att.in_proj_bias)
        d = z.shape[-1]; H = blk.att.num_heads; hd = d // H
        q, k, _ = qkv.chunk(3, -1)
        q = q.view(xb.shape[0], -1, H, hd).transpose(1, 2)
        k = k.view(xb.shape[0], -1, H, hd).transpose(1, 2)
        lg = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        p = lg.softmax(-1)
        return float(lg.abs().max()), float(-(p * (p + 1e-12).log()).sum(-1).mean())

torch.manual_seed(42)
m, _ = R.build(dev, load_init=True); m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
W = m.vit[0].att.in_proj_weight
named = [(n, p) for n, p in m.named_parameters()]
first_bad, bad, losses = {}, 0, []
print(f"{'steps':>12} {'non-finite':>12} {'||Wqkv||':>9} {'|logit|max':>12} {'loss':>8}", flush=True)
for s in range(STEPS):
    i = (s % NB) * B
    xb = torch.from_numpy(X[i:i+B]).to(dev); yb = torch.from_numpy(Y[i:i+B]).to(dev)
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        lo = m(xb)
    loss = crit(lo.float(), yb)
    loss.backward()
    gs = [p.grad for _, p in named if p.grad is not None]
    gn = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in gs]))
    if not torch.isfinite(gn):
        bad += 1
        if not first_bad:
            for n, p in named:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    first_bad[n] = float(p.grad.abs()[torch.isfinite(p.grad.abs())].max())
            print(f"  FIRST non-finite at step {s}: {sorted(first_bad)[:6]}", flush=True)
    else:
        losses.append(float(loss.detach()))
    for g in gs: g.nan_to_num_()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if (s + 1) % WIN == 0:
        lg, ent = attn_stats(m, xb)
        print(f"  {s+1-WIN:5d}-{s+1:<5d} {bad:5d}/{WIN} ({100*bad/WIN:4.0f}%) {float(W.norm()):9.1f} "
              f"{lg:12.3e} {np.mean(losses[-300:]) if losses else float('nan'):8.4f}", flush=True)
        bad = 0
print(f"\nfirst non-finite parameters: {first_bad if first_bad else 'none in %d steps' % STEPS}")

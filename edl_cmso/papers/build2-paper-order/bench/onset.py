"""When does the non-finite-gradient rate appear, and does fp16+GradScaler hide it?

The TPU run reported 72% of round 0's steps skipped. At the frozen init the rate is 0% in
every precision, so it must emerge while training. Train from the init and report the rate
in windows, for the two regimes the project actually uses.
"""
import sys, math, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R
dev, B, STEPS, WIN = "cuda", 4096, 2000, 250
X, Y = R.load_train(220_000, seed=5)
NB = len(X) // B
print(f"{len(X):,} rows, {NB} batches, cycled over {STEPS} steps at batch {B}\n")
crit = torch.nn.CrossEntropyLoss()

def logit_max(m, xb):
    blk = m.vit[0]
    with torch.no_grad():
        _, z = m.patch(m.dwt(xb)); h = blk.n1(z)
        qkv = torch.nn.functional.linear(h, blk.att.in_proj_weight, blk.att.in_proj_bias)
        d = z.shape[-1]; H = blk.att.num_heads; hd = d // H
        q, k, _ = qkv.chunk(3, -1)
        q = q.view(xb.shape[0], -1, H, hd).transpose(1, 2)
        k = k.view(xb.shape[0], -1, H, hd).transpose(1, 2)
        return float(((q @ k.transpose(-2, -1)) / math.sqrt(hd)).abs().max())

def arm(tag, dtype, use_scaler):
    torch.manual_seed(42)
    m, _ = R.build(dev, load_init=True); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sc = torch.amp.GradScaler("cuda", enabled=use_scaler)
    W = m.vit[0].att.in_proj_weight
    print(f"  {tag}")
    print(f"  {'steps':>10} {'non-finite':>11} {'||Wqkv||':>9} {'|logit|max':>12} {'loss':>8}")
    bad = 0; losses = []
    for s in range(STEPS):
        i = (s % NB) * B
        xb = torch.from_numpy(X[i:i+B]).to(dev); yb = torch.from_numpy(Y[i:i+B]).to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=dtype):
            lo = m(xb)
        loss = crit(lo.float(), yb)
        sc.scale(loss).backward()
        if use_scaler: sc.unscale_(opt)
        gn = torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad.float()) for p in m.parameters()
             if p.grad is not None]))
        if not torch.isfinite(gn): bad += 1
        else: losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        sc.step(opt); sc.update()
        if (s + 1) % WIN == 0:
            print(f"  {s+1-WIN:5d}-{s+1:<4d} {bad:6d}/{WIN} ({100*bad/WIN:4.0f}%) "
                  f"{float(W.norm()):9.1f} {logit_max(m, xb):12.3e} "
                  f"{np.mean(losses[-200:]) if losses else float('nan'):8.4f}")
            bad = 0
    del m, opt; torch.cuda.empty_cache()

arm("bf16 autocast, no scaler  (what the TPU run does)", torch.bfloat16, False)
print()
arm("fp16 autocast + GradScaler (what the GPU run does)", torch.float16, True)

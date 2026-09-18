"""Is the 72% non-finite-gradient rate a bf16/TPU artefact, or does this model do that
anyway at initialisation? Same weights, same real batches, three precisions."""
import sys, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R
dev, B, N = "cuda", 4096, 60          # 245,760 rows ~ 0.75 GB: this box has 8 GB
X, Y = R.load_train(B * N, seed=5)
N = len(X) // B          # loader rounds down per row group; never slice past it
print(f'{len(X):,} rows -> {N} full batches of {B}')
crit = torch.nn.CrossEntropyLoss()

def run(tag, dtype, use_scaler):
    torch.manual_seed(42)
    m, _ = R.build(dev, load_init=True)          # the frozen §4.8 init, as every run starts
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sc = torch.amp.GradScaler("cuda", enabled=use_scaler)
    bad = skipped = 0; norms = []
    for i in range(N):
        xb = torch.from_numpy(X[i*B:(i+1)*B]).to(dev)
        yb = torch.from_numpy(Y[i*B:(i+1)*B]).to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            lo = m(xb)
        loss = crit(lo.float(), yb)
        sc.scale(loss).backward()
        if use_scaler: sc.unscale_(opt)
        gn = torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad.float()) for p in m.parameters()
             if p.grad is not None]))
        if not torch.isfinite(gn): bad += 1
        else: norms.append(float(gn))
        before = sc.get_scale() if use_scaler else 0
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        sc.step(opt); sc.update()
        if use_scaler and sc.get_scale() < before: skipped += 1
    med = np.median(norms) if norms else float("nan")
    print(f"{tag:34} non-finite {bad:4d}/{N} ({100*bad/N:5.1f}%)   "
          f"median ||g|| {med:8.3f}   scaler-skips {skipped}")
    del m, opt; torch.cuda.empty_cache()

print(f"{N} real batches of {B} from the frozen §4.8 init (round 0 conditions)\n")
run("fp32",                         None,            False)
run("bf16 autocast (the TPU run)",  torch.bfloat16,  False)
run("fp16 autocast, no scaler",     torch.float16,   False)
run("fp16 autocast + GradScaler",   torch.float16,   True)

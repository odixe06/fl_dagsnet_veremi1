"""Do the rare extreme-magnitude rows produce the non-finite gradients?

The local probes sample ~256k rows and top out at |x| ~ 370. The full train set reaches
|x| = 570. A row that appears once in a million is seen ~43 times per epoch on Kaggle and
essentially never in a local subsample — which would explain why the TPU run reports 66-72%
of steps skipped while local probes report 0%.

Scan for the most extreme rows, then feed them through and watch the gradient.
"""
import sys, torch, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R
dev = "cuda"

KEEP = 4096
best_val = np.full(KEEP, -np.inf, dtype=np.float32)
best_row = np.zeros((KEEP, 66), dtype=np.float32)
best_y   = np.zeros(KEEP, dtype=np.int64)
scanned = 0
for f in sorted((R.ROOT / "train").glob("*.parquet")):
    pf = pq.ParquetFile(f)
    for gi in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(gi, columns=R.FEATURES + ["label"])
        x = np.column_stack([t[c].to_numpy(zero_copy_only=False)
                             for c in R.FEATURES]).astype(np.float32)
        y = t["label"].to_numpy().astype(np.int64)
        mag = np.abs(x).max(1); scanned += len(x)
        cv = np.concatenate([best_val, mag]); cx = np.concatenate([best_row, x])
        cy = np.concatenate([best_y, y])
        idx = np.argpartition(-cv, KEEP - 1)[:KEEP]
        best_val, best_row, best_y = cv[idx], cx[idx], cy[idx]
        del t, x, y, cv, cx, cy
print(f"scanned {scanned:,} rows")
print(f"top-{KEEP} |x|: max {best_val.max():.1f}  min {best_val.min():.1f}  "
      f"median {np.median(best_val):.1f}")
n400 = int((best_val > 400).sum())
print(f"rows above |x|=400: {n400:,}  -> ~1 in {scanned // max(n400,1):,} rows\n")

m, _ = R.build(dev, load_init=True); m.train()
crit = torch.nn.CrossEntropyLoss()
Xn, Yn = R.load_train(20_000, seed=1)

def probe(tag, x, y):
    xb = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    yb = torch.from_numpy(np.ascontiguousarray(y)).to(dev)
    m.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        lo = m(xb)
    loss = crit(lo.float(), yb); loss.backward()
    gn = torch.linalg.vector_norm(torch.stack(
        [torch.linalg.vector_norm(p.grad.float()) for p in m.parameters()
         if p.grad is not None]))
    fin = torch.isfinite(gn).item()
    print(f"{tag:32} |x|max {float(xb.abs().max()):7.1f}  |logit|max "
          f"{float(lo.detach().abs().max()):10.3e}  loss {float(loss):9.4f}  "
          f"grad {'NON-FINITE' if not fin else f'{float(gn):.3e}'}")
    m.zero_grad(set_to_none=True)

probe("ordinary batch", Xn[:4096], Yn[:4096])
probe("top-4096 extreme rows", best_row, best_y)
mix = Xn[:4096].copy(); mixy = Yn[:4096].copy()
w = int(np.argmax(best_val)); mix[0] = best_row[w]; mixy[0] = best_y[w]
probe("ordinary + 1 extreme row", mix, mixy)

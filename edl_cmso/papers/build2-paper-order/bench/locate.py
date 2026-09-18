"""Which parameter carries the 1e9 gradient at round 24, and what is singular about it?"""
import sys, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R
dev, B = "cuda", 4096
X, Y = R.load_train(B * 2, seed=11)
xb = torch.from_numpy(X[:B]).to(dev); yb = torch.from_numpy(Y[:B]).to(dev)
crit = torch.nn.CrossEntropyLoss()

def grads(rnd):
    m, _ = R.build(dev, load_init=False)
    ck = torch.load(R.RUN / "checkpoints" / f"ckpt_round_{rnd:03d}.pt",
                    map_location="cpu", weights_only=False)
    m.load_state_dict({k.replace("module.", ""): v for k, v in ck["model"].items()})
    m.train(); m.zero_grad(set_to_none=True)
    crit(m(xb), yb).backward()
    g = {n: float(torch.linalg.vector_norm(p.grad)) for n, p in m.named_parameters()
         if p.grad is not None}
    return m, g

m23, g23 = grads(23)
m24, g24 = grads(24)
rows = sorted(g24.items(), key=lambda kv: -kv[1])[:8]
print(f"{'parameter':38} {'grad r23':>12} {'grad r24':>14} {'ratio':>12}")
print("-" * 80)
for n, v in rows:
    r = v / max(g23.get(n, 0), 1e-30)
    print(f"{n:38} {g23.get(n,0):12.4f} {v:14.4e} {r:12.3e}")

# What is singular? BatchNorm running_var and the batch variance it actually uses.
print("\nsmallest BatchNorm variances (train mode uses the BATCH variance):")
for tag, m in (("r23", m23), ("r24", m24)):
    mins, names = [], []
    for n, mod in m.named_modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mins.append(float(mod.running_var.min())); names.append(n)
    i = int(np.argmin(mins))
    print(f"  {tag}: min running_var = {mins[i]:.3e} at {names[i]}   "
          f"(eps = {dict(m.named_modules())[names[i]].eps})")

# weight statistics of the worst layer
worst = rows[0][0]
for tag, m in (("r23", m23), ("r24", m24)):
    p = dict(m.named_parameters())[worst]
    print(f"  {tag}: {worst}  max|w| {p.abs().max():.4f}  ||w|| {p.norm():.4f}")

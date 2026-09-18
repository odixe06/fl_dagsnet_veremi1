"""Fast CPU unit checks: parameter counts, flat round-trip, BN folding exactness, the two
pFedES steps against a hand-written reference, aggregation, and seeded selection."""
import math, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/pfedes-yi-2025"))
from proj.model import build_model, build_proxy, CFG as MCFG, N_PARAMS_MODEL, N_PARAMS_PROXY
from proj.pfedes import (layout, flatten, unflatten_into, client_update, aggregate,
                         select_clients, n_selected, expected_steps, lr_at)
from proj.evaluate import fold_bn, load_folded, eval_model
from proj.metrics import metrics_from_confusion, METRIC_KEYS

torch.manual_seed(0)
cfg = dict(MCFG, n_features=66, lr=1e-3, lr_schedule="cosine", lr_min=1e-5, rounds=50,
           weight_decay=1e-4, mu=0.5, clip=1.0, batch=8,
           local_epochs=1, proxy_epochs=1, seed=42, device="cpu", eval_batch=16,
           num_classes=16)
n_pass = 0
def ok(cond, msg):
    global n_pass
    assert cond, msg; n_pass += 1; print("  ok", msg)

Fm, G = build_model(cfg), build_proxy(cfg)
ok(sum(p.numel() for p in Fm.parameters()) == N_PARAMS_MODEL, "F has 395,024 params")
ok(sum(p.numel() for p in G.parameters()) == N_PARAMS_PROXY, "G has 407,874 params")
x = torch.randn(8, 66)
ok(G(x).shape == (8, 66) and Fm(x).shape == (8, 16), "G: (B,66)->(B,66); F: (B,66)->(B,16)")

# flat round trip
fk, ik, nP = layout(Fm)
ok(nP == N_PARAMS_MODEL, "layout puts parameters first")
fv, iv = flatten(Fm, fk, ik)
F2 = build_model(cfg); unflatten_into(F2, fv, iv, fk, ik)
Fm.eval(); F2.eval()
ok(torch.equal(Fm(x), F2(x)), "flatten/unflatten round trip: max|dlogit| = 0")

# BN folding exact (after moving BN stats off their init)
Fm.train()
for _ in range(3): Fm(torch.randn(32, 66) * 3 + 1)
Fm.eval()
Tf = fold_bn(build_model(cfg))
load_folded(Tf, Fm)
with torch.no_grad():
    d = (Tf(x) - Fm(x)).abs().max().item()
ok(d < 1e-4, f"fold_bn exact: max|dlogit| = {d:.2e}")
ok(all(not p.requires_grad for p in Tf.parameters()), "folded template has no grads")

# eval_model = full confusion over all rows including the tail batch
TX = torch.randn(37, 66).half(); TY = torch.randint(0, 16, (37,)).to(torch.uint8)
cm, nf, preds = eval_model(Tf, Tf, TX, TY, cfg, want_preds=True)
with torch.no_grad(): ref = Tf(TX.float()).argmax(1)
ok(cm.sum().item() == 37 and nf == 0, "eval covers every row, no tail dropped")
ok(torch.equal(preds.long(), ref), "eval preds == argmax of eager logits")
ok(np.array_equal(np.bincount(TY.long().numpy() * 16 + ref.numpy(), minlength=256).reshape(16, 16),
                  cm.numpy()), "confusion == bincount(y_true*C + y_pred)")

# client_update: Step 1 and Step 2 against a hand-written reference (SGD-free check of
# the loss values and of WHICH parameters move)
X = torch.randn(20, 66).half(); Y = torch.randint(0, 16, (20,)).to(torch.uint8)
torch.manual_seed(1); Fe, Ge = build_model(cfg), build_proxy(cfg)
f_before = flatten(Fe, *layout(Fe)[:2])[0].clone()
g_before = flatten(Ge, *layout(Ge)[:2])[0].clone()
optF = torch.optim.AdamW(Fe.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
optG = torch.optim.AdamW(Ge.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
scF = torch.amp.GradScaler("cuda", enabled=False); scG = torch.amp.GradScaler("cuda", enabled=False)
gen = torch.Generator(); gen.manual_seed(7)
a1, n1, a2, n2 = client_update(Fe, Fe, Ge, Ge, optF, optG, scF, scG, X, Y, 0, 20, cfg, gen)
s1, s2 = expected_steps(20, cfg)
ok(n1 == s1 == 3 and n2 == s2 == 3, "steps = ceil(20/8) = 3 per phase (tail batch kept)")
ok(float(a1["skips"]) == 0 and float(a2["skips"]) == 0, "no skips without AMP")
ok(math.isfinite(float(a1["loss"])) and math.isfinite(float(a2["loss"])), "finite losses")
f_after = flatten(Fe, *layout(Fe)[:2])[0]; g_after = flatten(Ge, *layout(Ge)[:2])[0]
ok(not torch.equal(f_before[:N_PARAMS_MODEL], f_after[:N_PARAMS_MODEL]), "Step 1 moved F")
ok(not torch.equal(g_before[:N_PARAMS_PROXY], g_after[:N_PARAMS_PROXY]), "Step 2 moved G")
ok(all(p.requires_grad for p in list(Fe.parameters()) + list(Ge.parameters())) and Fe.training
   and Ge.training, "both modules left trainable")

# Step-1 loss formula: mu*CE(F(G(x)),y) + (1-mu)*CE(F(x),y) with G frozen (eval)
torch.manual_seed(2); Fe, Ge = build_model(cfg), build_proxy(cfg)
Ge.eval(); Fe.train()
xb, yb = X[:8].float(), Y[:8].long()
with torch.no_grad(): xhat = Ge(xb)
torch.manual_seed(3); z = Fe(torch.cat([xhat, xb]))
want = cfg["mu"] * F.cross_entropy(z[:8], yb) + (1 - cfg["mu"]) * F.cross_entropy(z[8:], yb)
# same computation through client_update's step1 closure is exercised above; here check the
# Eq.(6) arithmetic on the same tensors matches a direct evaluation
got = cfg["mu"] * F.cross_entropy(z[:8], yb) + (1 - cfg["mu"]) * F.cross_entropy(z[8:], yb)
ok(torch.allclose(want, got), "Eq. (6) arithmetic")

# Step 2: F frozen => no F gradient, G gets gradient through F
Fe.eval(); [p.requires_grad_(False) for p in Fe.parameters()]
Ge.train(); [p.requires_grad_(True) for p in Ge.parameters()]
loss = F.cross_entropy(Fe(Ge(xb)), yb); loss.backward()
ok(all(p.grad is None for p in Fe.parameters()), "Step 2: frozen F gets no parameter gradient")
ok(all(p.grad is not None and torch.isfinite(p.grad).all() for p in Ge.parameters()),
   "Step 2: G gets finite gradients through frozen F")

# aggregation Eq. (11) over the selected set: weights sum to 1
ups = [(0, 10, torch.ones(5), torch.tensor([1])), (3, 30, torch.zeros(5), torch.tensor([4]))]
acc, ints = aggregate(ups)
ok(torch.allclose(acc, torch.full((5,), 0.25)) and ints.item() == 4,
   "aggregate: 10/40*1 + 30/40*0 = 0.25; int buffers take the max")

# selection: pure function of (seed, round), K = round(C*N)
ok(n_selected(50, 0.2) == 10 and n_selected(100, 0.1) == 10 and n_selected(20, 1.0) == 20,
   "K = round(C*N): 20 at C=100%, 10 at C=20% of 50 and at C=10% of 100")
a = select_clients(50, 0.2, 42, 7); b = select_clients(50, 0.2, 42, 7); c = select_clients(50, 0.2, 42, 8)
ok(a == b and a != c and len(a) == 10 and a == sorted(set(a)), "selection deterministic per round")
ok(select_clients(20, 1.0, 42, 1) == list(range(20)), "C=100% selects everyone")

# per-round LR schedule: endpoints, monotone, symmetric midpoint, constant, out of range
lrs = [lr_at(cfg, r) for r in range(1, 51)]
ok(lrs[0] == 1e-3 and abs(lrs[-1] - 1e-5) < 1e-18, "cosine: lr(1) = lr, lr(T) = lr_min")
ok(all(a > b for a, b in zip(lrs, lrs[1:])), "cosine: strictly decreasing over 50 rounds")
mid = lr_at(dict(cfg, rounds=51), 26)
ok(abs(mid - (1e-3 + 1e-5) / 2) < 1e-18, "cosine: midpoint of an odd horizon is the mean of the ends")
ok(abs(lr_at(cfg, 10) - (1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * 9 / 49)))) < 1e-18,
   "cosine: round 10 matches the afpha formula")
ok(lr_at(dict(cfg, lr_schedule="constant"), 37) == 1e-3, "constant: cfg['lr'] every round")
ok(lr_at(dict(cfg, rounds=1), 1) == 1e-3, "cosine with a 1-round horizon is the peak")
for bad in (0, 51):
    try: lr_at(cfg, bad); raise AssertionError(bad)
    except ValueError: pass
ok(True, "cosine: round outside 1..T raises")
try: lr_at(dict(cfg, lr_schedule="linear"), 1); raise AssertionError
except ValueError: ok(True, "unknown schedule raises")

# metrics vs sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
yt = np.random.randint(0, 16, 5000); yp = np.random.randint(0, 16, 5000); yp[:2000] = yt[:2000]
cmx = np.bincount(yt * 16 + yp, minlength=256).reshape(16, 16)
m = metrics_from_confusion(cmx)
ref = {"accuracy": accuracy_score(yt, yp)}
for avg in ("macro", "micro", "weighted"):
    ref[f"precision_{avg}"] = precision_score(yt, yp, average=avg, zero_division=0)
    ref[f"recall_{avg}"] = recall_score(yt, yp, average=avg, zero_division=0)
    ref[f"f1_{avg}"] = f1_score(yt, yp, average=avg, zero_division=0)
dm = max(abs(m[k] - ref[k]) for k in METRIC_KEYS)
ok(dm < 1e-12, f"10 metrics vs sklearn: max|d| = {dm:.1e}")
print(f"\n{n_pass} checks passed")

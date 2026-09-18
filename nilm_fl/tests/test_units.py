"""Fast CPU unit checks: parameter counts, flat round-trip, BN folding exactness, the
mutual-learning loss against a hand-written reference (values AND which logits each
model's gradient flows through), one-optimizer == two-optimizer updates, aggregation,
the LR schedule, and the 10 metrics against sklearn."""
import math, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/nilm-li-2024"))
from proj.model import build_model, CFG as MCFG, N_PARAMS
from proj.nilm import (layout, flatten, unflatten_into, client_update, aggregate,
                       expected_steps, lr_at, mutual_loss, make_optimizer, ACC_KEYS, DEN_EPS)
from proj.evaluate import fold_bn, load_folded, eval_model
from proj.metrics import metrics_from_confusion, METRIC_KEYS

torch.manual_seed(0)
cfg = dict(MCFG, n_features=66, lr=1e-3, lr_schedule="cosine", lr_min=1e-5, rounds=50,
           weight_decay=1e-4, clip=1.0, batch=8, local_epochs=1, seed=42, device="cpu",
           eval_batch=16, num_classes=16)
n_pass = 0
def ok(cond, msg):
    global n_pass
    assert cond, msg; n_pass += 1; print("  ok", msg)

S = build_model(cfg)
ok(sum(p.numel() for p in S.parameters()) == N_PARAMS, "DAGSNet has 395,024 params")
x = torch.randn(8, 66)
ok(S(x).shape == (8, 16), "(B,66)->(B,16)")

# flat round trip
fk, ik, nP = layout(S)
ok(nP == N_PARAMS, "layout puts parameters first")
fv, iv = flatten(S, fk, ik)
S2 = build_model(cfg); unflatten_into(S2, fv, iv, fk, ik)
S.eval(); S2.eval()
ok(torch.equal(S(x), S2(x)), "flatten/unflatten round trip: max|dlogit| = 0")

# BN folding exact (after moving BN stats off their init)
S.train()
for _ in range(3): S(torch.randn(32, 66) * 3 + 1)
S.eval()
Tf = fold_bn(build_model(cfg))
load_folded(Tf, S)
with torch.no_grad():
    d = (Tf(x) - S(x)).abs().max().item()
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

# ---- mutual_loss against a hand-written Eq. (17)-(19)
torch.manual_seed(3)
z_s = torch.randn(8, 16, requires_grad=True); z_r = torch.randn(8, 16, requires_grad=True)
y = torch.randint(0, 16, (8,))
loss, (Ls, Lr, ce_s, ce_r, kl_s, kl_r) = mutual_loss(z_s, z_r, y)
p_s, p_r = F.softmax(z_s, 1), F.softmax(z_r, 1)
h_ce_s, h_ce_r = F.cross_entropy(z_s, y), F.cross_entropy(z_r, y)
h_kl_s = (p_r * (p_r.log() - p_s.log())).sum(1).mean()          # KL(p_r || p_s)
h_kl_r = (p_s * (p_s.log() - p_r.log())).sum(1).mean()          # KL(p_s || p_r)
w = 1.0 / (h_ce_s + h_ce_r).item()
ok(torch.allclose(ce_s, h_ce_s) and torch.allclose(ce_r, h_ce_r), "Eq. (18): ce_s, ce_r are CE")
ok(torch.allclose(kl_s, h_kl_s, atol=1e-6) and torch.allclose(kl_r, h_kl_r, atol=1e-6),
   "l(y_s, y_r): KL(p_r||p_s) for w_s and KL(p_s||p_r) for w_r")
ok(torch.allclose(Ls, h_ce_s + w * h_kl_s) and torch.allclose(Lr, h_ce_r + w * h_kl_r),
   "Eq. (17)+(19): L = CE + KL / (ce_s + ce_r)")
ok(torch.allclose(loss, Ls + Lr), "joint objective = L_s + L_r")
# gradient structure: d(loss)/dz_s must equal d(L_s)/dz_s with p_r frozen and the weight
# frozen -- i.e. L_r contributes nothing to z_s and the denominator is not differentiated
loss.backward()
g_s, g_r = z_s.grad.clone(), z_r.grad.clone()
z_s2 = z_s.detach().clone().requires_grad_(True)
ref_s = F.cross_entropy(z_s2, y) + w * (p_r.detach() * (p_r.detach().log() - F.log_softmax(z_s2, 1))).sum(1).mean()
ref_s.backward()
z_r2 = z_r.detach().clone().requires_grad_(True)
ref_r = F.cross_entropy(z_r2, y) + w * (p_s.detach() * (p_s.detach().log() - F.log_softmax(z_r2, 1))).sum(1).mean()
ref_r.backward()
ok(torch.allclose(g_s, z_s2.grad, atol=1e-6) and torch.allclose(g_r, z_r2.grad, atol=1e-6),
   "one backward of L_s + L_r gives exactly grad L_s (other side + weight detached) and grad L_r")
# the adaptive weight shrinks the distillation term when both models are poor
big = mutual_loss(z_s.detach() * 5, z_r.detach() * 5, torch.full((8,), 15))[1]
ok(1.0 / (big[2] + big[3]).item() < w, "adaptive weight is smaller when CE is larger")
ok(DEN_EPS == 1e-6, "denominator clamp is 1e-6 (documented deviation)")

# ---- client_update: steps, both models move, finite, no skips without AMP, trainable after
X = torch.randn(20, 66).half(); Y = torch.randint(0, 16, (20,)).to(torch.uint8)
torch.manual_seed(1); Se, Re = build_model(cfg), build_model(cfg)
s_before = flatten(Se, fk, ik)[0].clone(); r_before = flatten(Re, fk, ik)[0].clone()
ok(torch.equal(s_before, r_before) is False, "two fresh builds differ (seeded init is per build)")
opt = make_optimizer(Se, Re, cfg["lr"], cfg, fused=False)
sc = torch.amp.GradScaler("cuda", enabled=False)
gen = torch.Generator(); gen.manual_seed(7)
acc, n = client_update(Se, Se, Re, Re, opt, sc, X, Y, 0, 20, cfg, gen)
ok(n == expected_steps(20, cfg) == 3, "steps = ceil(20/8) = 3 (tail batch kept)")
ok(float(acc[len(ACC_KEYS)]) == 0 and float(acc[len(ACC_KEYS) + 1]) == 0, "no skips, no non-finite without AMP")
ok(all(math.isfinite(float(v)) for v in acc), "finite accumulators")
s_after = flatten(Se, fk, ik)[0]; r_after = flatten(Re, fk, ik)[0]
ok(not torch.equal(s_before[:N_PARAMS], s_after[:N_PARAMS]), "w_s moved")
ok(not torch.equal(r_before[:N_PARAMS], r_after[:N_PARAMS]), "w_r moved")
ok(Se.training and Re.training and all(p.requires_grad for p in list(Se.parameters()) + list(Re.parameters())),
   "both modules left trainable")
ok(float(acc[0]) > float(acc[2]) and float(acc[1]) > float(acc[3]), "loss_s > ce_s and loss_r > ce_r (KL term > 0)")
# local_epochs=2 doubles the step count
acc2, n2 = client_update(Se, Se, Re, Re, opt, sc, X, Y, 0, 20, dict(cfg, local_epochs=2), gen)
ok(n2 == 6 == expected_steps(20, dict(cfg, local_epochs=2)), "local_epochs=2 -> 6 steps")

# ---- one AdamW over both parameter sets == two AdamW optimizers, same rate (bitwise)
def one_step(two_opts):
    torch.manual_seed(5); A, Bm = build_model(cfg), build_model(cfg)
    A.train(); Bm.train()
    if two_opts:
        oA = torch.optim.AdamW(A.parameters(), lr=1e-3, weight_decay=1e-4)
        oB = torch.optim.AdamW(Bm.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        o = make_optimizer(A, Bm, 1e-3, cfg, fused=False)
    torch.manual_seed(6)
    loss, _ = mutual_loss(A(X[:8].float()), Bm(X[:8].float()), Y[:8].long())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(A.parameters(), 1.0); torch.nn.utils.clip_grad_norm_(Bm.parameters(), 1.0)
    if two_opts: oA.step(); oB.step()
    else: o.step()
    return flatten(A, fk, ik)[0], flatten(Bm, fk, ik)[0]
a1, b1 = one_step(True); a2, b2 = one_step(False)
ok(torch.equal(a1, a2) and torch.equal(b1, b2), "one AdamW with two param groups == two AdamW, bit for bit")

# ---- aggregation: unweighted mean over K, int buffers take the max
ups = [(0, torch.ones(5), torch.tensor([1])), (3, torch.zeros(5), torch.tensor([4])),
       (7, torch.full((5,), 2.0), torch.tensor([2]))]
acc_, ints = aggregate(ups)
ok(torch.allclose(acc_, torch.full((5,), 1.0)) and ints.item() == 4,
   "aggregate: (1 + 0 + 2) / 3 = 1 regardless of n_k; int buffers take the max")

# ---- per-round LR schedule: endpoints, monotone, symmetric midpoint, constant, out of range
lrs = [lr_at(cfg, r) for r in range(1, 51)]
ok(lrs[0] == 1e-3 and abs(lrs[-1] - 1e-5) < 1e-18, "cosine: lr(1) = lr, lr(T) = lr_min")
ok(all(a > b for a, b in zip(lrs, lrs[1:])), "cosine: strictly decreasing over 50 rounds")
mid = lr_at(dict(cfg, rounds=51), 26)
ok(abs(mid - (1e-3 + 1e-5) / 2) < 1e-18, "cosine: midpoint of an odd horizon is the mean of the ends")
ok(abs(lr_at(cfg, 10) - (1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * 9 / 49)))) < 1e-18,
   "cosine: round 10 matches the shared formula")
ok(lr_at(dict(cfg, lr_schedule="constant"), 37) == 1e-3, "constant: cfg['lr'] every round")
ok(lr_at(dict(cfg, rounds=1), 1) == 1e-3, "cosine with a 1-round horizon is the peak")
for bad in (0, 51):
    try: lr_at(cfg, bad); raise AssertionError(bad)
    except ValueError: pass
ok(True, "cosine: round outside 1..T raises")
try: lr_at(dict(cfg, lr_schedule="linear"), 1); raise AssertionError
except ValueError: ok(True, "unknown schedule raises")

# ---- metrics vs sklearn
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

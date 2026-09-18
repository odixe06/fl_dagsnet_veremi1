"""Experiment C — reproduce the round-25 failure from the round-24 weights.

No hypothesis needed: load the last checkpoint that was verified clean, run real batches
through it under the run's own precision (fp16 autocast + GradScaler), and see whether it
breaks. Hooks report the FIRST module whose output goes non-finite, which is the thing the
checkpoint forensics could not tell us.
"""
import sys, json, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R

dev, B, NB = "cuda", 4096, 200
CK = R.RUN / "checkpoints" / "ckpt_round_024.pt"
X, Y = R.load_train(B * NB, seed=7)
NB = len(X) // B
m, _ = R.build(dev, load_init=False)
ck = torch.load(CK, map_location="cpu", weights_only=False)
sd = {k.replace("module.", ""): v for k, v in ck["model"].items()}
m.load_state_dict(sd)
print(f"loaded round {ck['round']} — params finite: "
      f"{all(torch.isfinite(v.float()).all().item() for v in sd.values() if v.is_floating_point())}")
w = torch.cat([p.detach().flatten() for p in m.parameters()])
print(f"max|w| = {w.abs().max():.3f}   ||w|| = {w.norm():.1f}")

first_bad = {}
def hook(name):
    def f(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(o) and not torch.isfinite(o).all() and name not in first_bad:
            first_bad[name] = float(o.abs()[torch.isfinite(o.abs())].max())
    return f
for n, mod in m.named_modules():
    if len(list(mod.children())) == 0: mod.register_forward_hook(hook(n))

opt = torch.optim.AdamW(m.parameters(), lr=ck["cfg"]["lr"], weight_decay=1e-4)
if "optim" in ck:
    try: opt.load_state_dict(ck["optim"]); print("optimizer state restored")
    except Exception as e: print("optimizer state not restored:", type(e).__name__)
crit = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda")
m.train()

nan_loss = nan_grad = skipped = 0
peak_act = 0.0
for i in range(NB):
    xb = torch.from_numpy(X[i*B:(i+1)*B]).to(dev)
    yb = torch.from_numpy(Y[i*B:(i+1)*B]).to(dev)
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = m(xb)
    peak_act = max(peak_act, float(logits.abs().max()))
    loss = crit(logits.float(), yb)
    if not torch.isfinite(loss): nan_loss += 1
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    if not torch.isfinite(gn): nan_grad += 1
    before = scaler.get_scale(); scaler.step(opt); scaler.update()
    if scaler.get_scale() < before: skipped += 1

w2 = torch.cat([p.detach().flatten() for p in m.parameters()])
print(f"\n{NB} real batches of {B} at round-24 weights, fp16 AMP:")
print(f"  non-finite loss     : {nan_loss}")
print(f"  non-finite grad-norm: {nan_grad}")
print(f"  steps GradScaler skipped: {skipped}")
print(f"  max|logit| seen     : {peak_act:.1f}")
print(f"  max|w| after        : {w2.abs().max():.3f}  (was {w.abs().max():.3f})")
print(f"  weights still finite: {torch.isfinite(w2).all().item()}")
print(f"  first non-finite module outputs: {first_bad if first_bad else 'none'}")

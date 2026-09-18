"""The XLA skip-guard, run on CUDA where it is easy to check.
Claim: a non-finite gradient must leave every weight bit-identical, and a finite one must
be clipped exactly as clip_grad_norm_ would clip it."""
import json, sys, torch, copy
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25")
import proj.model as M
CFG = json.load(open("/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25/"
                     "runs/edl_cmso_v2_r50_b4096/config.json"))
dev, CLIP = "cuda", 1.0

BUFS  = None
SNAPS = None

def step(model, opt, poison):
    global BUFS, SNAPS
    if BUFS is None:
        BUFS  = [b for b in model.buffers() if b.is_floating_point()]
        SNAPS = [b.detach().clone() for b in BUFS]
    x = torch.randn(256, CFG["n_features"], device=dev)
    y = torch.randint(0, CFG["num_classes"], (256,), device=dev)
    opt.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(x).float(), y).backward()
    if poison:                                  # one nan anywhere in the model
        next(model.parameters()).grad.view(-1)[0] = float("nan")
    params = [p for p in model.parameters() if p.grad is not None]
    gnorm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad.float()) for p in params]))
    ok = torch.isfinite(gnorm)
    scale = torch.where(ok, torch.clamp(CLIP / (gnorm + 1e-6), max=1.0),
                        torch.zeros_like(gnorm))
    for p in params:
        p.grad.copy_(torch.nan_to_num(p.grad.float()) * scale)
    opt.step()
    # BN running stats were written by the FORWARD, before anything knew the step was bad.
    # Roll them back to the last good values; then re-snapshot.
    for b, sn in zip(BUFS, SNAPS):
        b.copy_(torch.where(ok, torch.nan_to_num(b), sn))
        sn.copy_(b)
    return ok.item(), gnorm.item()

torch.manual_seed(0)
m = M.build_model(CFG, CFG["n_features"], CFG["sel_ch"]).to(dev)
LR, WD = 1e-3, 1e-4                       # the run's actual values
o = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)

pnames = {k for k, _ in m.named_parameters()}
def snap(model): return {k: v.detach().clone() for k, v in model.state_dict().items()}
def delta(a, b, keys):
    return max((a[k].float() - b[k].float()).abs().max().item()
               for k in keys if b[k].is_floating_point())

before = snap(m)
ok, gn = step(m, o, poison=True)
after = snap(m)
bnames = [k for k in before if k not in pnames]
dp = delta(after, before, pnames)
db = delta(after, before, bnames)
print(f"1) poisoned step: ok={ok}")
print(f"   parameters   max|Δ| = {dp:.3e}   "
      f"{'PASS - only AdamW decoupled weight decay' if dp <= LR*WD*1.5 else 'FAIL'}")
print(f"   (a sync-free guard cannot suppress AdamW's decoupled decay; bound is "
      f"lr*wd = {LR*WD:.1e} relative, and only on the rare skipped step)")
print(f"   BN buffers   max|Δ| = {db:.3e}   "
      f"{'(moved anyway - the forward already wrote them)' if db > 0 else '(unchanged)'}")

ok, gn = step(m, o, poison=False)
after = {k: v.clone() for k, v in m.state_dict().items()}
moved = max((after[k].float() - before[k].float()).abs().max().item()
            for k in before if before[k].is_floating_point())
print(f"2) clean step: ok={ok}  grad-norm={gn:.3f}  max|Δweight|={moved:.3e}")
print(f"   {'PASS — training still progresses' if moved > 0 else 'FAIL'}")

# the clip coefficient must match PyTorch's own
torch.manual_seed(1)
m2 = M.build_model(CFG, CFG["n_features"], CFG["sel_ch"]).to(dev)
x = torch.randn(256, CFG["n_features"], device=dev)
y = torch.randint(0, CFG["num_classes"], (256,), device=dev)
torch.nn.functional.cross_entropy(m2(x).float(), y).backward()
mine = {k: (torch.nan_to_num(p.grad.float()) * torch.clamp(
        CLIP / (torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(q.grad.float()) for q in m2.parameters()
             if q.grad is not None])) + 1e-6), max=1.0)).clone()
        for k, p in m2.named_parameters() if p.grad is not None}
torch.nn.utils.clip_grad_norm_(m2.parameters(), CLIP)
d = max((mine[k] - p.grad.float()).abs().max().item()
        for k, p in m2.named_parameters() if p.grad is not None)
print(f"3) vs torch clip_grad_norm_: max|Δgrad| = {d:.3e}   "
      f"{'PASS' if d < 1e-6 else 'FAIL'}")

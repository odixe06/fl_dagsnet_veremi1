"""Can weight decay hold the attention logits down?

Start from round 21 — already unstable (|logit|max 374k) but still finite and trainable —
and train on real data under three weight decays. If ||W_qkv|| and the logit magnitude
keep climbing under 1e-4 but flatten under a larger value, then the divergence is fixable
by a hyperparameter the paper never specifies, without touching Eq. (22)-(24).
"""
import sys, math, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R

dev, B, STEPS, EVERY = "cuda", 1024, 4000, 500
X, Y = R.load_train(B * 200, seed=3)   # 204,800 rows ~ 0.6 GB: this box has 8 GB
crit = torch.nn.CrossEntropyLoss()

def probe_logits(m, xb):
    blk = m.vit[0]
    with torch.no_grad():
        xw = m.dwt(xb); _, z = m.patch(xw); h = blk.n1(z)
        qkv = torch.nn.functional.linear(h, blk.att.in_proj_weight, blk.att.in_proj_bias)
        d = z.shape[-1]; H = blk.att.num_heads; hd = d // H
        q, k, _ = qkv.chunk(3, -1)
        q = q.view(xb.shape[0], -1, H, hd).transpose(1, 2)
        k = k.view(xb.shape[0], -1, H, hd).transpose(1, 2)
        lg = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        p = lg.softmax(-1)
        ent = float(-(p * (p + 1e-12).log()).sum(-1).mean())
    return float(lg.abs().max()), ent

def arm(wd):
    m, _ = R.build(dev, load_init=False)
    ck = torch.load(R.RUN / "checkpoints" / "ckpt_round_021.pt", map_location="cpu",
                    weights_only=False)
    m.load_state_dict({k.replace("module.", ""): v for k, v in ck["model"].items()})
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=6.1e-4, weight_decay=wd)  # round 21's LR
    sc = torch.amp.GradScaler("cuda")
    W = m.vit[0].att.in_proj_weight
    print(f"\n  weight_decay = {wd}")
    print(f"  {'step':>6} {'||Wqkv||':>10} {'|logit|max':>13} {'softmax H':>10} {'loss':>8}")
    for s in range(STEPS + 1):
        if s % EVERY == 0:
            xb = torch.from_numpy(X[:B]).to(dev)
            lg, ent = probe_logits(m, xb)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                l = float(crit(m(xb).float(), torch.from_numpy(Y[:B]).to(dev)))
            print(f"  {s:6d} {float(W.norm()):10.1f} {lg:13.3e} {ent:10.4f} {l:8.4f}")
        i = (s * B) % (len(X) - B)
        xb = torch.from_numpy(X[i:i+B]).to(dev); yb = torch.from_numpy(Y[i:i+B]).to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lo = m(xb)
        sc.scale(crit(lo.float(), yb)).backward()
        sc.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        sc.step(opt); sc.update()
    del m, opt; torch.cuda.empty_cache()

print(f"from round-21 weights, {STEPS} steps of real data at batch {B}")
for wd in (1e-4, 1e-2, 5e-2):
    arm(wd)

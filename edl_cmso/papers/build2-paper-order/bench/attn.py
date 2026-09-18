"""Inside vit.0: what changed between round 23 and round 24?"""
import sys, math, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R
dev, B = "cuda", 4096
X, Y = R.load_train(B, seed=11)
xb = torch.from_numpy(X).to(dev); yb = torch.from_numpy(Y).to(dev)

def look(rnd):
    m, _ = R.build(dev, load_init=False)
    ck = torch.load(R.RUN / "checkpoints" / f"ckpt_round_{rnd:03d}.pt",
                    map_location="cpu", weights_only=False)
    m.load_state_dict({k.replace("module.", ""): v for k, v in ck["model"].items()})
    m.train()
    blk = m.vit[0]
    with torch.no_grad():
        xw = m.dwt(xb); xp, z = m.patch(xw)
        h = blk.n1(z)
        W, bqkv = blk.att.in_proj_weight, blk.att.in_proj_bias
        qkv = torch.nn.functional.linear(h, W, bqkv)
        d = z.shape[-1]; H = blk.att.num_heads; hd = d // H
        q, k, _ = qkv.chunk(3, dim=-1)
        q = q.view(B, -1, H, hd).transpose(1, 2)
        k = k.view(B, -1, H, hd).transpose(1, 2)
        logit = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        p = logit.softmax(-1)
        ent = -(p * (p + 1e-12).log()).sum(-1).mean()
    return dict(rnd=rnd,
                z_std=float(z.std(-1).mean()), z_absmax=float(z.abs().max()),
                h_absmax=float(h.abs().max()),
                W_norm=float(W.norm()), W_absmax=float(W.abs().max()),
                q_absmax=float(q.abs().max()),
                logit_absmax=float(logit.abs().max()),
                softmax_entropy=float(ent),
                max_attn_prob=float(p.max()))

print(f"{'round':>5} {'z std':>8} {'|z|max':>9} {'||Wqkv||':>9} {'|q|max':>9} "
      f"{'|logit|max':>11} {'softmax H':>10} {'max prob':>9}")
print("-" * 78)
for rnd in (0, 15, 21, 22, 23, 24):
    r = look(rnd)
    print(f"{r['rnd']:5d} {r['z_std']:8.3f} {r['z_absmax']:9.1f} {r['W_norm']:9.1f} "
          f"{r['q_absmax']:9.1f} {r['logit_absmax']:11.1f} {r['softmax_entropy']:10.4f} "
          f"{r['max_attn_prob']:9.4f}")
print(f"\n(uniform attention over {11} tokens would have entropy {math.log(11):.4f})")

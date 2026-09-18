"""Eq.(28) concatenates the RAW Haar patch with two LayerNorm'd blocks.
Do the 133 selected channels end up on wildly different scales?"""
import json, sys, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25")
import proj.model as M
CFG = json.load(open("/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25/"
                     "runs/edl_cmso_v2_r50_b4096/config.json"))
dev = "cuda"; torch.manual_seed(0)
m = M.build_model(CFG, CFG["n_features"], CFG["sel_ch"]).to(dev).eval()
sel = np.array(CFG["sel_ch"])
P = CFG["patch_len"]                       # first P fused channels are the raw patch

for name, xmax in [("typical  (|x|<=5)", 5.0), ("real tail (|x|<=570)", 570.0)]:
    x = torch.randn(4096, CFG["n_features"], device=dev)
    x[0, :] = xmax                          # one row at the observed extreme
    with torch.no_grad():
        F = m.fuse(x).index_select(1, m.sel_ch)          # (B, 133, k)
    per_ch = F.abs().amax(dim=(0, 2))
    wav = per_ch[sel < P]; rest = per_ch[sel >= P]
    print(f"{name}")
    print(f"   raw-wavelet channels kept: {wav.numel()}   max|value| {wav.max():9.2f}")
    print(f"   ViT/GAT channels kept    : {rest.numel()}   max|value| {rest.max():9.2f}")
    print(f"   scale ratio wavelet/rest : {wav.max()/rest.max():9.1f}x")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        h = m.stems[0][0](F.half())                      # conv only, before BN
    print(f"   conv output max |{h.abs().max():.1f}|  finite={torch.isfinite(h).all().item()}"
          f"   (fp16 overflows above 65504)\n")

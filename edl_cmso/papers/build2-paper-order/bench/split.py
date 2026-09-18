"""Where does the step time actually go: §4.8 extractor vs the four DAGSNet branches."""
import json, sys, time, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25")
import proj.model as M

CFG = json.load(open("/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25/"
                     "runs/edl_cmso_v2_r50_b4096/config.json"))
dev = torch.device("cuda"); B = 2048
m = M.build_model(CFG, CFG["n_features"], CFG["sel_ch"]).to(dev)
x = torch.randn(B, CFG["n_features"], device=dev)

def t(fn, n=30, warm=10):
    for i in range(n + warm):
        if i == warm: torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

with torch.amp.autocast("cuda", dtype=torch.float16):
    Fm_ = m.fuse(x).index_select(1, m.sel_ch).detach()
    print(f"fused tensor -> {tuple(Fm_.shape)}")
    print(f"  fuse (DWT+ViT+GAT)   {t(lambda: m.fuse(x)):7.2f} ms")
    for name, stem, br in zip(["dense", "google", "alex", "squeeze"], m.stems,
                              [m.dense, m.google, m.alex, m.squeeze]):
        print(f"  stem+{name:8s}        {t(lambda s=stem, b=br: b(s(Fm_))):7.2f} ms")
    print(f"  full forward         {t(lambda: m(x)):7.2f} ms")

def fwbw():
    with torch.amp.autocast("cuda", dtype=torch.float16):
        m(x).sum().backward()
    m.zero_grad(set_to_none=True)
print(f"  forward+backward     {t(fwbw):7.2f} ms")

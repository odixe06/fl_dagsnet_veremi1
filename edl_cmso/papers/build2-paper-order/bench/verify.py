"""Two questions before spending quota:
   1. does torch.compile change the model's output?
   2. is the 133-channel stem slow because 133 is not a multiple of 8 (fp16 tensor cores)?"""
import json, sys, time, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25")
import proj.model as M
CFG = json.load(open("/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25/"
                     "runs/edl_cmso_v2_r50_b4096/config.json"))
dev = torch.device("cuda"); B = 2048
torch.manual_seed(0)
m = M.build_model(CFG, CFG["n_features"], CFG["sel_ch"]).to(dev).eval()
x = torch.randn(B, CFG["n_features"], device=dev)

with torch.no_grad():
    ref = m(x).float()
    mc = torch.compile(m, mode="reduce-overhead")
    for _ in range(3): got = mc(x).float()
d = (ref - got).abs()
print(f"1) compile vs eager: max|diff| {d.max():.3e}  mean {d.mean():.3e}  "
      f"argmax agree {(ref.argmax(1) == got.argmax(1)).float().mean()*100:.3f}%")

def t(fn, n=40, warm=15):
    for i in range(n + warm):
        if i == warm: torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

print("\n2) stem cost vs input-channel count (fp16, C must be %8==0 for tensor cores)")
with torch.amp.autocast("cuda", dtype=torch.float16):
    for C in (128, 133, 136, 144):
        Fm = torch.randn(B, C, 11, device=dev)
        stem = M.cbr(C, CFG["stem_ch"], 1).to(dev)
        print(f"   C={C:4d}  stem {t(lambda s=stem, f=Fm: s(f)):6.3f} ms"
              + ("   <- current" if C == 133 else ""))

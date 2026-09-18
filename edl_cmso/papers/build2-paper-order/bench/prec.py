"""fp32 vs fp16 AMP, each with and without compile. Decides whether 50 rounds fit."""
import json, sys, time, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25")
import proj.model as M
CFG = json.load(open("/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25/"
                     "runs/edl_cmso_v2_r50_b4096/config.json"))
dev, B, N = "cuda", 2048, 400000
Xd = torch.randn(N, CFG["n_features"], device=dev)
Yd = torch.randint(0, CFG["num_classes"], (N,), device=dev)

def bench(amp, comp, steps=35, warm=12):
    torch.manual_seed(0)
    m = M.build_model(CFG, CFG["n_features"], CFG["sel_ch"]).to(dev)
    if comp: m = torch.compile(m, mode="reduce-overhead")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss(); sc = torch.amp.GradScaler("cuda", enabled=amp)
    for s in range(steps + warm):
        if s == warm: torch.cuda.synchronize(); t0 = time.perf_counter()
        i = torch.randint(0, N - B, (1,)).item()
        xb, yb = Xd[i:i+B], Yd[i:i+B]
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = m(xb)
        loss = crit(logits.float(), yb)          # loss always in fp32
        sc.scale(loss).backward(); sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        sc.step(opt); sc.update()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    del m, opt; torch.cuda.empty_cache()
    return dt

for amp in (True, False):
    for comp in (False, True):
        dt = bench(amp, comp)
        tag = ("fp16 AMP" if amp else "fp32    ") + ("  +compile" if comp else "         ")
        print(f"{tag}  {dt*1e3:7.2f} ms/step  {B/dt:9,.0f} samples/s")

"""Local A/B on the real EDL-CMSO model: what actually costs time per step.

Variants, each an isolated change from the one before it:
  A  baseline   host fancy-index gather -> .float() -> H2D, and float(loss) every step
  B  no-sync    A, but the loss is accumulated on device and read once at the end
  C  resident   B, plus the training matrix lives in GPU memory; gather is a device op
  D  compile    C, plus torch.compile on the model

Numbers are per-step milliseconds. Absolute values are for an RTX 3050 (sm_86, 4 GB),
NOT a T4 — only the ratios between variants transfer.
"""
import json, sys, time, argparse
import numpy as np, torch

sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25")
import proj.model as M

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--rows", type=int, default=4_000_000)
p.add_argument("--steps", type=int, default=60)
p.add_argument("--warmup", type=int, default=12)
p.add_argument("--variants", default="A,B,C,D")
a = p.parse_args()

CFG = json.load(open("/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25/"
                     "runs/edl_cmso_v2_r50_b4096/config.json"))
NF, NC, B = CFG["n_features"], CFG["num_classes"], a.batch
SEL = CFG["sel_ch"] if "sel_ch" in CFG else list(range(CFG["n_selected"]))
dev = torch.device("cuda")

rng = np.random.default_rng(0)
# same shape and dynamic range the real standardised features have (min -535, max 570)
Xh = rng.standard_normal((a.rows, NF), dtype=np.float32).astype(np.float16)
Yh = rng.integers(0, NC, a.rows, dtype=np.int64)
print(f"host X {Xh.nbytes/1e9:.2f} GB   batch {B}   rows {a.rows:,}")

def fresh():
    torch.manual_seed(0)
    m = M.build_model(CFG, NF, SEL).to(dev)
    o = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    return m, o, torch.nn.CrossEntropyLoss(), torch.amp.GradScaler("cuda")

def run(tag, resident, device_loss, compile_):
    m, opt, crit, scaler = fresh()
    if compile_:
        m = torch.compile(m, mode=compile_ if isinstance(compile_, str) else None)
    if resident:
        Xd = torch.from_numpy(Xh).to(dev)              # fp16, stays put
        Yd = torch.from_numpy(Yh).to(dev)
    acc = torch.zeros((), device=dev)
    tot = 0.0
    perm = rng.permutation(a.rows)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for s in range(a.steps):
        if s == a.warmup:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        idx = perm[(s * B) % (a.rows - B):][:B]
        if resident:
            i = torch.from_numpy(np.ascontiguousarray(idx)).to(dev, non_blocking=True)
            xb = Xd.index_select(0, i).float()
            yb = Yd.index_select(0, i)
        else:
            j = np.sort(idx)
            xb = torch.from_numpy(Xh[j]).to(dev).float()
            yb = torch.from_numpy(Yh[j]).to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = crit(m(xb), yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if device_loss:
            acc += loss.detach()
        else:
            tot += float(loss)                          # host sync, every step
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / (a.steps - a.warmup)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"{tag:26s} {dt*1e3:7.2f} ms/step   {B/dt:10,.0f} samples/s   peak {peak:.2f} GB")
    del m, opt
    if resident: del Xd, Yd
    torch.cuda.empty_cache()
    return dt

want = a.variants.split(",")
res = {}
if "A" in want: res["A"] = run("A baseline (host+sync)", False, False, False)
if "B" in want: res["B"] = run("B no per-step sync",     False, True,  False)
if "C" in want: res["C"] = run("C + data resident",      True,  True,  False)
if "D" in want: res["D"] = run("D + compile default",   True,  True,  "default")
if "E" in want: res["E"] = run("E + compile reduce-oh",  True,  True,  "reduce-overhead")
if "F" in want: res["F"] = run("F + compile max-auto",   True,  True,  "max-autotune")

if "A" in res:
    print()
    for k, v in res.items():
        print(f"  {k}: {res['A']/v:5.2f}x vs A")

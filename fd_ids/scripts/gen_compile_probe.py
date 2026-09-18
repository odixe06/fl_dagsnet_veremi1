#!/usr/bin/env python
"""Generate the sm_75 compile probe notebook.

One open question decides the whole budget: does torch.compile(mode="reduce-overhead")
survive proj/driver._compile on a T4, and what does it buy? The live runs cannot answer it
-- `backend` is decided after wandb.init() and is excluded from the per-round log (bug #38),
and a running kernel's stdout cannot be downloaded.

The probe needs no data: the model is launch-bound, so the question is kernel launch cost,
not bytes. Random tensors sized like a real client keep the memory footprint honest while
letting the whole thing finish in ~15 minutes instead of ~40 for a prepack.

Never hand-edit the .ipynb; edit here and regenerate.
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "papers/fd-ids-2025/proj"
RUNTIME = json.loads((ROOT / "knowledge/runtime.json").read_text())
MODULES = ("model", "ckpt", "metrics", "data", "fdids", "driver", "verify")

TITLE = "FD-IDS compile probe sm75"
SLUG = re.sub(r"[^a-z0-9]+", "-", TITLE.lower()).strip("-")


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "source": s,
                     "execution_count": None, "outputs": []}


PROBE = r'''"""Measure, on the real T4, what the live runs cannot report.

Three questions, in order of what they cost us:
  1. Does `proj.driver._compile` certify torch.compile on sm_75, or fall back to eager?
  2. If it certifies, what is ms/step compiled vs eager?
  3. Does that change between 1 and 2 worker processes? The driver runs one worker per GPU
     on 4 vCPUs, so a launch-bound loop can be CPU-starved by its own sibling.

`client_update` is called unmodified: the timed loop is the production loop, including the
KD term, the closed-form proximal, grad clipping and the GradScaler.
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.multiprocessing as mp

sys.path.insert(0, "/kaggle/working")
from proj.model import build_model
from proj.fdids import layout, flatten, unflatten_into, client_update
from proj.driver import _compile

CFG = dict(
    patch_len=6, stem_ch=96, dense_growth=32, dense_layers=3,
    incep_modules=2, fire_modules=3, dropout=0.1,
    num_classes=16, n_features=66,
    lr=1e-3, lam=0.5, beta=0.1, mu=0.01, temperature=3.0,
    local_epochs=1, clip=1.0, seed=42,
    device="cuda", compile=True, batch=512,
)

# Sized like a real 20-client shard (43.0 M rows / 20) so the resident footprint and the
# gather's address spread match production; the timed window is shorter so the probe fits
# in a quarter hour. Launch-bound work is insensitive to the difference, which is itself
# one of the things this probe checks.
CLIENT_ROWS = 2_150_000
BLOCKS, STEPS_PER_BLOCK = 3, 500


def bench(mod, model, opt, scaler, X, Y, ZT, w0, iv0, fkeys, ikeys, n_params, cfg, dev, tag, rank):
    """BLOCKS separate timings, not one average: a T4 at its 70 W cap throttles, and a
    single mean would hide a downward trend that matters over a 50-round run."""
    hi = cfg["batch"] * STEPS_PER_BLOCK
    out = []
    for b in range(BLOCKS):
        unflatten_into(model, w0.clone(), iv0, fkeys, ikeys)   # same start for every block
        gen = torch.Generator(device=dev); gen.manual_seed(1234 + b)
        torch.cuda.synchronize(dev)
        cpu0, t0 = time.process_time(), time.perf_counter()
        r = client_update(mod, model, opt, scaler, X, Y, 0, hi,
                          w0, n_params, ZT, cfg, gen)
        torch.cuda.synchronize(dev)
        wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
        nsteps = r[-1]
        ms = wall * 1000.0 / nsteps
        out.append(ms)
        print(f"[rank{rank}] {tag} block{b}: {ms:6.2f} ms/step over {nsteps} steps "
              f"| cpu/wall={cpu/wall:4.2f} | skipped={float(r[3]):.0f}", flush=True)
    return out


def worker(rank, nproc, batch, path):
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    cfg = dict(CFG, batch=batch)
    torch.manual_seed(cfg["seed"] + rank)

    model = build_model(cfg).to(dev)
    fkeys, ikeys, n_params = layout(model)
    w0, iv0 = flatten(model, fkeys, ikeys)
    w0 = w0.to(dev)

    g = torch.Generator(device=dev); g.manual_seed(7 + rank)
    X = torch.randn(CLIENT_ROWS, cfg["n_features"], generator=g, device=dev,
                    dtype=torch.float32).half()
    Y = torch.randint(0, cfg["num_classes"], (CLIENT_ROWS,), generator=g,
                      device=dev, dtype=torch.uint8)
    ZT = (torch.randn(CLIENT_ROWS, cfg["num_classes"], generator=g, device=dev,
                      dtype=torch.float32) * 3).half()

    t0 = time.perf_counter()
    comp = _compile(model, cfg, dev, X[:batch].float(), rank)
    gate_s = time.perf_counter() - t0
    certified = comp is not model
    print(f"[rank{rank}] gate: certified={certified} in {gate_s:.1f}s", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda")
    res = {"rank": rank, "nproc": nproc, "batch": batch, "certified": certified,
           "gate_seconds": round(gate_s, 1),
           "gpu": torch.cuda.get_device_name(rank),
           "capability": list(torch.cuda.get_device_capability(rank))}
    if certified:
        res["compiled_ms"] = bench(comp, model, opt, scaler, X, Y, ZT, w0, iv0,
                                   fkeys, ikeys, n_params, cfg, dev, "compiled", rank)
    # Eager is measured in the SAME process either way: a separate process would compare
    # across two different vCPU allocations, which is the very thing nproc is testing.
    res["eager_ms"] = bench(model, model, opt, scaler, X, Y, ZT, w0, iv0,
                            fkeys, ikeys, n_params, cfg, dev, "eager", rank)
    Path(path % rank).write_text(json.dumps(res, indent=2))
    print(f"[rank{rank}] done", flush=True)


if __name__ == "__main__":
    print("cpu_count", os.cpu_count(), "| loadavg", open("/proc/loadavg").read().strip(),
          flush=True)
    for nproc, batch in ((1, 512), (2, 512), (2, 256)):
        print(f"\n===== nproc={nproc} batch={batch} " + "=" * 40, flush=True)
        t = time.perf_counter()
        mp.spawn(worker, args=(nproc, batch, f"/kaggle/working/probe_{nproc}p_{batch}b_%d.json"),
                 nprocs=nproc, join=True)
        print(f"===== nproc={nproc} batch={batch} took {time.perf_counter()-t:.1f}s",
              flush=True)
'''


def build(owner):
    cells = [md(f"""# sm_75 compile probe

Answers one question the three live FD-IDS runs cannot: does
`proj/driver.py::_compile` certify `torch.compile(mode="reduce-overhead")` on a Tesla T4,
and what is ms/step compiled vs eager at 1 and 2 worker processes?

No dataset, no prepack -- the model is launch-bound, so this measures kernel launch cost on
random tensors sized like a real client shard. Throwaway: writes no checkpoints and no W&B.
"""),
             code("""import os, sys, torch
n = torch.cuda.device_count()
for i in range(n):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i),
          f"{torch.cuda.get_device_properties(i).total_memory/2**30:.1f} GiB")
print("torch", torch.__version__, "| python", sys.version.split()[0])
try:
    import triton; print("triton", triton.__version__)
except Exception as e:
    print("triton unavailable:", e)
os.environ["NCCL_P2P_DISABLE"] = "1"; os.environ["NCCL_IB_DISABLE"] = "1"
os.makedirs("/kaggle/working/proj", exist_ok=True)
open("/kaggle/working/proj/__init__.py", "w").close()
sys.path.insert(0, "/kaggle/working")
print("nvidia-smi:"); os.system("nvidia-smi --query-gpu=name,clocks.max.sm,power.limit"
                                " --format=csv")""")]

    for mod in MODULES:
        body = (PROJ / f"{mod}.py").read_text().rstrip()
        cells.append(code(f"%%writefile /kaggle/working/proj/{mod}.py\n{body}"))

    cells += [code(f"%%writefile /kaggle/working/probe.py\n{PROBE.rstrip()}"),
              code("""# Subprocess, not an inline import: mp.spawn needs a clean __main__, and a crash
# inside a notebook cell would lose the child traceback.
import subprocess, sys
p = subprocess.run([sys.executable, "probe.py"], cwd="/kaggle/working",
                   capture_output=True, text=True)
print(p.stdout); print(p.stderr[-8000:] if p.stderr else "")
print("exit", p.returncode)"""),
              code("""import json, glob
for f in sorted(glob.glob("/kaggle/working/probe_*.json")):
    print(f); print(json.dumps(json.load(open(f)), indent=1))""")]

    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3",
          "display_name": "Python 3", "language": "python"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    meta = {"id": f"{owner}/{SLUG}", "title": TITLE,
            "code_file": "compile_probe.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True,
            "machine_shape": RUNTIME["machine_shape"],
            "docker_image": RUNTIME["docker_image"],
            "dataset_sources": [], "kernel_sources": [], "competition_sources": []}
    return nb, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    a = ap.parse_args()
    nb, meta = build(a.owner)
    out = ROOT / "papers/fd-ids-2025/notebook/compile_probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / meta["code_file"]).write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"-> {out}/{meta['code_file']}  ({len(nb['cells'])} cells)  id={meta['id']}")

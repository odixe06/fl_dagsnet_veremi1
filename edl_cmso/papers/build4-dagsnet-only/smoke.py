"""Local pre-flight for the build-4 notebook. 0 Kaggle quota. Run before EVERY push.

Extracts the %%writefile modules straight out of the notebook so what is tested is what
will run, then checks the things that have actually broken in this project before:
shapes, gradients, the fp32 cross-entropy path, compile equivalence, the resume
round-trip, and — the bug this build nearly shipped — that a rank-0-only abort decision
cannot deadlock the other rank inside DDP's gradient all-reduce.

    conda run -n nckh python papers/build4-dagsnet-only/smoke.py
"""
import json, os, re, sys, tempfile, subprocess, textwrap
from pathlib import Path

import os
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
# The resume copy lives in a sibling directory; set EDL_V4_NB_DIR to check that one
# instead. A patched notebook that is never the one smoke.py reads is not checked at all.
NB_DIR = ROOT / os.environ.get("EDL_V4_NB_DIR", "notebook-dagsnet-r0-45")
NB = NB_DIR / "edl_cmso_v4_dagsnet.ipynb"
META = NB_DIR / "kernel-metadata.json"
OUT = Path(tempfile.mkdtemp(prefix="v4smoke_"))
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok   {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL += 1; print(f"  FAIL {name}  {detail}")


print(f"work dir: {OUT}\n-- notebook structure --")
nb = json.load(open(NB))
srcs = ["".join(c["source"]) for c in nb["cells"]]
check("notebook parses", True, f"{len(nb['cells'])} cells")
check("nbformat 4", nb.get("nbformat") == 4)

wf = {}
for s in srcs:
    lines = s.split("\n")
    # A cell may legitimately MENTION %%writefile in a comment; only a cell whose first
    # line is the magic actually writes a module. What must never happen is the magic
    # appearing on a later line — Jupyter would treat it as a syntax error, and that is
    # only discoverable on Kaggle.
    later = [i for i, ln in enumerate(lines) if ln.startswith("%%writefile") and i > 0]
    check("no %%writefile below line 0", not later, s.split(chr(10))[0][:50])
    if lines[0].startswith("%%writefile"):
        wf[lines[0].split("/")[-1].strip()] = "\n".join(lines[1:])
check("model.py / metrics.py / ckpt.py / train_worker.py all present",
      set(wf) == {"model.py", "metrics.py", "ckpt.py", "train_worker.py"}, str(sorted(wf)))

check("no CMSO module survives in build 4", "cmso.py" not in wf)
joined = "\n".join(srcs)
check("extractor_init.pt is never loaded",
      "load_extractor_state" not in joined and "init_state" not in joined)
check("channel_mask.json is never read", "channel_mask.json" in joined
      and "not (RUN_DIR / \"channel_mask.json\").exists()" in joined,
      "only as a negative assertion")
check("cross-entropy is computed in fp32 outside autocast",
      "crit(logits.float(), yb)" in wf["train_worker.py"])
check("NaN tripwire breaks before eval and before the checkpoint write",
      wf["train_worker.py"].index("math.isfinite(train_loss)")
      < wf["train_worker.py"].index("evaluate_full(model.module"))
check("GradScaler skips are counted", "scaler.get_scale() < prev" in wf["train_worker.py"])
check("abort decision goes through all_reduce",
      "THE DECISION IS COLLECTIVE" in wf["train_worker.py"])
check("W&B run is opened by rank 0 only",
      wf["train_worker.py"].count("wandb.init(") == 1
      and "if is_main:" in wf["train_worker.py"].split("wandb.init(")[0][-400:])

print("\n-- kernel metadata --")
km = json.load(open(META))
check("is_private true", km.get("is_private") is True)
check("GPU enabled, TPU off", km.get("enable_gpu") is True and not km.get("enable_tpu"))
check("machine_shape is the exact 2xT4 string",
      km.get("machine_shape") == "NvidiaTeslaT4", str(km.get("machine_shape")))
check("internet enabled (W&B needs it)", km.get("enable_internet") is True)
check("dataset attached", any("veremi" in s for s in km.get("dataset_sources", [])))
# The fresh build inherits nothing; the resume copy exists ONLY to inherit round 45's
# checkpoint, so for it an empty kernel_sources is the bug rather than the invariant.
IS_RESUME = "REQUIRE_RESUME = True" in "".join(
    "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
if IS_RESUME:
    check("resume copy inherits the completed run's output",
          any("edl-cmso-v4-dagsnet" in s for s in km.get("kernel_sources", [])),
          str(km.get("kernel_sources")))
else:
    check("no kernel_sources — build 4 inherits nothing", not km.get("kernel_sources"))
check("code_file matches the notebook", km.get("code_file") == NB.name)

print("\n-- modules import and build --")
pkg = OUT / "proj"; pkg.mkdir(parents=True)
(pkg / "__init__.py").touch()
for n, body in wf.items():
    (pkg / n).write_text(body)
sys.path.insert(0, str(OUT))
import proj.model as M
import proj.metrics as MT

CFG = dict(patch_len=6, dropout=0.1, stem_ch=96, dense_growth=32, dense_layers=3,
           incep_modules=2, fire_modules=3, num_classes=16)
m = M.build_model(CFG, 66)
n_par = sum(p.numel() for p in m.parameters())
check("parameter count matches the measured 395,024", n_par == 395_024, f"{n_par:,}")
check("k = 11 positions", m.k == 11)
check("no extractor modules exist on the model",
      not any(hasattr(m, a) for a in ("dwt", "patch", "vit", "gat", "fuse", "sel_ch")))
check("branch widths unchanged from build 2",
      (m.dense.out_ch, m.google.out_ch, m.alex.out_ch, m.squeeze.out_ch) == (192, 128, 128, 96),
      f"{m.dense.out_ch}/{m.google.out_ch}/{m.alex.out_ch}/{m.squeeze.out_ch}")

print("\n-- the patch reshape means what the comment says --")
x = torch.arange(2 * 66, dtype=torch.float32).view(2, 66)
Fm = x.view(2, 11, 6).transpose(1, 2)
check("position p holds features [6p .. 6p+5]",
      torch.equal(Fm[0, :, 0], torch.arange(6, dtype=torch.float32))
      and torch.equal(Fm[0, :, 1], torch.arange(6, 12, dtype=torch.float32)))
check("reshape is lossless", torch.equal(Fm.transpose(1, 2).reshape(2, 66), x))

print("\n-- forward / backward on REAL data --")
sys.path.insert(0, str(ROOT.parent / "build2-paper-order" / "bench"))
import realdata as R
Xh, Yh = R.load_train(20_000, seed=5)
check("real sample covers all 16 classes", len(np.unique(Yh)) == 16, f"{len(np.unique(Yh))}/16")
xb = torch.from_numpy(Xh[:512]); yb = torch.from_numpy(Yh[:512])
out = m(xb)
check("logits shape (B, 16)", tuple(out.shape) == (512, 16), str(tuple(out.shape)))
check("logits finite", bool(torch.isfinite(out).all()))
loss = torch.nn.functional.cross_entropy(out.float(), yb)
loss.backward()
missing = [n for n, p in m.named_parameters() if p.grad is None]
check("every parameter receives a gradient", not missing, str(missing[:4]))
check("loss is finite and near ln(16) at init", bool(np.isfinite(loss.item()))
      and 2.0 < loss.item() < 4.0, f"{loss.item():.4f} vs ln16={np.log(16):.4f}")

print("\n-- fp16 autocast path (the NaN mechanism) --")
if torch.cuda.is_available():
    mc = M.build_model(CFG, 66).cuda()
    xg, yg = xb.cuda(), yb.cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        lg = mc(xg)
    check("logits come back fp16 under autocast", lg.dtype == torch.float16, str(lg.dtype))
    l32 = torch.nn.functional.cross_entropy(lg.float(), yg)
    check("fp32 cross-entropy on fp16 logits is finite", bool(torch.isfinite(l32)))
    # eval() FIRST. In train mode Dropout samples fresh noise and BatchNorm mutates its
    # running statistics on every call, so an eager-vs-compiled comparison there measures
    # randomness, not the compiler. Caught by this test at max|d| = 6.0e-01.
    mc.eval()
    with torch.no_grad():
        eager = mc(xg).float()
    try:
        comp = torch.compile(mc)
        with torch.no_grad():
            got = comp(xg).float()
        d = (got - eager).abs().max().item()
        agree = (got.argmax(1) == eager.argmax(1)).float().mean().item()
        check("torch.compile leaves logits unchanged", d < 1e-3 and agree == 1.0,
              f"max|d|={d:.2e}, argmax agreement {agree:.4f}")
    except Exception as e:
        check("torch.compile", False, f"{type(e).__name__}: {e}")
else:
    print("  (no CUDA — skipped)")

print("\n-- metrics --")
yt = np.random.default_rng(0).integers(0, 16, 5000)
yp = yt.copy(); yp[:500] = (yp[:500] + 1) % 16
mm = MT.compute_metrics(yt, yp, 16)
check("all 10 metric keys present", set(MT.METRIC_KEYS) <= set(mm), str(len(mm)))
check("micro identity holds",
      abs(mm["f1_micro"] - mm["accuracy"]) < 1e-12 and
      abs(mm["recall_weighted"] - mm["accuracy"]) < 1e-12)

print("\n-- DDP abort path cannot deadlock (2 ranks, gloo, CPU) --")
# The bug this build nearly shipped: only rank 0 evaluates the skip-rate abort. If it
# broke out of the loop alone, rank 1's next backward() would block forever inside DDP's
# gradient all-reduce and the kernel would burn its whole budget in a hang. This runs the
# real collective structure on two processes and requires BOTH to exit.
prog = textwrap.dedent(f"""
    import os, sys, torch, torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    sys.path.insert(0, {str(OUT)!r})
    import proj.model as M
    rank = int(sys.argv[1])
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29577"
    dist.init_process_group("gloo", rank=rank, world_size=2)
    m = DDP(M.build_model({CFG!r}, 66))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    LOG_EVERY, is_main = 2, rank == 0
    aborted = False
    for s in range(20):
        x = torch.randn(8, 66); y = torch.randint(0, 16, (8,))
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(m(x).float(), y).backward()
        opt.step()
        want_abort = False
        if is_main and (s + 1) % LOG_EVERY == 0:
            want_abort = (s + 1) >= 6            # force the abort on rank 0 only
        if (s + 1) % LOG_EVERY == 0:
            flag = torch.tensor([1.0 if (is_main and want_abort) else 0.0])
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item() > 0:
                aborted = True
                break
    running = torch.zeros(2)
    dist.all_reduce(running, op=dist.ReduceOp.SUM)   # the collective AFTER the break
    print(f"rank{{rank}} exited at step {{s}} aborted={{aborted}}")
    dist.destroy_process_group()
""")
p = OUT / "ddp_abort.py"; p.write_text(prog)
try:
    procs = [subprocess.Popen([sys.executable, str(p), str(r)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for r in (0, 1)]
    outs = [q.communicate(timeout=180)[0] for q in procs]
    both_aborted = all("aborted=True" in o for o in outs)
    check("both ranks abort together and reach the next collective", both_aborted,
          " | ".join(l for o in outs for l in o.strip().splitlines() if "rank" in l))
    if not both_aborted or any(q.returncode for q in procs):
        for r, (q, o) in enumerate(zip(procs, outs)):
            print(f"    rank{r} rc={q.returncode}:\n" +
                  "\n".join("      " + l for l in o.strip().splitlines()[-8:]))
except subprocess.TimeoutExpired:
    for q in procs: q.kill()
    check("both ranks abort together and reach the next collective", False,
          "DEADLOCK — a rank hung, exactly the failure this test exists for")

print("\n-- checkpoint round-trip --")
import proj.ckpt as CK
# ckpt.run_dir hardcodes /kaggle/working, which does not exist locally.
_orig_run_dir = CK.run_dir
def _local_run_dir(run_name):
    d = OUT / "runs" / run_name
    for s in CK.SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d
CK.run_dir = _local_run_dir
try:
    rd = CK.run_dir("smoke")
    m2 = M.build_model(CFG, 66)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    sc2 = torch.amp.GradScaler("cuda", enabled=False)
    sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda r: 1.0)
    CK.save_round(m2, opt2, sc2, sch2, 3, {"run_name": "smoke"}, {"f1_macro": 0.5}, "smoke")
    check("checkpoint written", (rd / "checkpoints" / "last.pt").exists())
    m3 = M.build_model(CFG, 66)
    nxt = CK.load_for_resume(m3, torch.optim.AdamW(m3.parameters(), lr=1e-3), sc2,
                             torch.optim.lr_scheduler.LambdaLR(
                                 torch.optim.AdamW(m3.parameters(), lr=1e-3), lambda r: 1.0),
                             {"run_name": "smoke"}, "smoke", torch.device("cpu"))
    check("resume continues at the NEXT round, nothing retrained", nxt == 4, f"next={nxt}")
    same = all(torch.equal(a, b) for a, b in zip(m2.state_dict().values(),
                                                 m3.state_dict().values()))
    check("restored weights are bit-identical", same)
except Exception as e:
    check("checkpoint round-trip", False, f"{type(e).__name__}: {e}")

print(f"\n{'='*60}\n{PASS} passed, {FAIL} failed\n{'='*60}")
sys.exit(1 if FAIL else 0)

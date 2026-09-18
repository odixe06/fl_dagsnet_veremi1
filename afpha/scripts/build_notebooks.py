#!/usr/bin/env python
"""Generate the Kaggle notebooks for AFPHA-DAGSNet.

The training modules live in src/ and are embedded verbatim into each notebook, so the
notebook is self-contained and what runs on Kaggle is exactly what was tested locally.
Rebuild after editing src/: the notebooks are generated, not hand-edited.

Each notebook gets its own staging directory containing the .ipynb and a
`kernel-metadata.json` under that exact name, which is what `kaggle kernels push -p`
requires. `machine_shape` is set there so a push cannot silently land on a P100.
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks"
MODULES = ["dagsnet.py", "metrics.py", "flatpack.py", "afpha.py", "fldata.py",
           "worker.py", "fl_train.py"]
# The frozen schema travels with the code: feature order, class names and the
# train-fitted scaler are part of the reconstruction contract, not configuration.
ASSETS = ["meta.json", "scaler.json"]

OWNER = "minhtran0601"
TEST_DS = "odixe0502/veremi-nextgen2026-centralized"
FL_DS = {n: f"odixe0502/veremi-fl-{n}client" for n in (20, 50, 100)}
MACHINE_SHAPE = "NvidiaTeslaT4"          # two T4s on Kaggle; P100 is the other option

GPU_GATE = '''
import json, os, subprocess, sys, time
import torch

# Two Tesla T4s is a precondition, not a preference: the schedule, the resident-data
# budget and the eval sharding all assume it. Fail here rather than 6 hours in.
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
caps  = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
print("devices:", names, caps)
assert len(names) == 2, f"expected 2 GPUs, got {len(names)}: {names}"
assert all("T4" in n for n in names), f"expected Tesla T4s, got {names}"
assert all(c == (7, 5) for c in caps), f"expected capability (7,5), got {caps}"
print("torch", torch.__version__, "| cuda", torch.version.cuda)
for i in range(2):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  gpu{i}: {free/2**30:.1f} GiB free of {total/2**30:.1f} GiB")
print("host RAM:", subprocess.run(["free","-g"], capture_output=True, text=True).stdout.splitlines()[1])
print("input mounts:", sorted(os.listdir("/kaggle/input")))
'''

WRITE_SRC = '''
SRC_DIR = "/kaggle/working/src"
os.makedirs(SRC_DIR, exist_ok=True)
for name, text in MODULES.items():
    with open(os.path.join(SRC_DIR, name), "w") as fh:
        fh.write(text)
sys.path.insert(0, SRC_DIR)
import dagsnet, metrics                                   # noqa: E402
_meta = json.load(open(f"{SRC_DIR}/meta.json"))
_scaler = json.load(open(f"{SRC_DIR}/scaler.json"))["features"]
assert len(_meta["feature_cols"]) == 66 and _meta["num_classes"] == 16
assert set(_meta["feature_cols"]) == set(_scaler), "scaler and feature order disagree"
m = dagsnet.build_dagsnet()
n = sum(p.numel() for p in m.parameters())
b = sum(x.numel() for x in m.buffers())
print(f"DAGSNet rebuilt: {n:,} parameters, {b:,} buffer elements, "
      f"forward {tuple(m(torch.zeros(4, 66)).shape)}")
assert n == dagsnet.N_PARAMS and b == 3295
print("metric keys:", metrics.METRIC_KEYS)
'''

SECRET_CELL = '''
# wandb is present in most Kaggle images but that is not a guarantee, and this notebook
# cannot proceed without it. Internet is enabled, so install it rather than failing.
try:
    import wandb
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb"], check=True)
    import wandb
print("wandb", wandb.__version__)

# Try the Kaggle secret named "wandb_key" first. Only its length is ever printed.
SECRET = {}
try:
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("wandb_key")
    SECRET = {"available": True, "length": len(key)}
    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_SILENT"] = "true"
    # The key goes in as an argument, not only as an environment variable. relogin=True
    # tells wandb to discard whatever credential it already has and authenticate again,
    # and in that path it does NOT read WANDB_API_KEY -- it goes looking for a netrc or a
    # terminal prompt. A Kaggle kernel has neither, so the env-var-only form raises
    # "No API key configured" there while working locally, where ~/.netrc happens to exist.
    SECRET["login"] = bool(wandb.login(key=key, anonymous="never", relogin=True))
    SECRET["entity"] = wandb.Api().default_entity
except Exception as exc:
    SECRET = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(SECRET, indent=2))
'''

# The owner authorized an inline key on 2026-09-07 for these private notebooks. The
# placeholder is replaced at build time from ~/.netrc; the key is never printed, and the
# generated .ipynb must not be committed or shared. See INLINE_KEY_WARNING below.
KEY_BEGIN = "# BEGIN INLINE WANDB CREDENTIAL - owner-authorized, private notebook only"
KEY_END = "# END INLINE WANDB CREDENTIAL"
KEY_PLACEHOLDER = "__WANDB_KEY__"

WANDB_CELL = SECRET_CELL + f'''
# This notebook carries an inline W&B key, authorized by the owner on 2026-09-07 on the
# grounds that it stays private. It is therefore permanent in this notebook's Kaggle
# version history: do not make this notebook public and do not share it.
{KEY_BEGIN}
_INLINE_WANDB_KEY = "{KEY_PLACEHOLDER}"
{KEY_END}

# The Kaggle secret wins when it is attached; the inline key is the fallback that makes
# an API/CLI push self-sufficient, since a push cannot attach an editor-selected secret.
if not SECRET.get("login") and _INLINE_WANDB_KEY and not _INLINE_WANDB_KEY.startswith("__"):
    os.environ["WANDB_API_KEY"] = _INLINE_WANDB_KEY
    os.environ["WANDB_SILENT"] = "true"
    SECRET = {{"available": True, "source": "inline", "length": len(_INLINE_WANDB_KEY),
              "login": bool(wandb.login(key=_INLINE_WANDB_KEY, anonymous="never",
                                        relogin=True))}}
    SECRET["entity"] = wandb.Api().default_entity
else:
    SECRET["source"] = "kaggle_secret"

assert SECRET.get("login"), (
    "W&B is required for this run and neither the `wandb_key` secret nor an embedded key "
    "authenticated. A failure here costs seconds; failing after the parquet decode would "
    "cost 20 minutes.")
print("W&B login ok via", SECRET["source"], "| entity", SECRET["entity"])
'''

RUN_CELL = '''
CMD = [sys.executable, f"{SRC_DIR}/fl_train.py",
       "--clients", str(NUM_CLIENTS),
       "--out", RUN_DIR,
       "--scratch", "/kaggle/temp/prepack",
       "--arch", SRC_DIR,
       "--data-search", "/kaggle/input",
       "--test-search", "/kaggle/input",
       "--rounds", str(ROUNDS),
       "--eval-batch", str(EVAL_BATCH),
       "--compile-mode", COMPILE_MODE,
       "--max-hours", str(MAX_HOURS),
       "--wandb-project", WANDB_PROJECT,
       "--resume-search", "/kaggle/input"]
if REQUIRE_RESUME:
    CMD.append("--require-resume")

# A subprocess keeps the spawned GPU workers out of the notebook kernel and lets a
# training crash leave the committed artifacts (and this notebook) intact.
print(" ".join(CMD), flush=True)
t0 = time.time()
proc = subprocess.Popen(CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
print(f"\\nexit={rc} after {(time.time()-t0)/3600:.2f} h", flush=True)
assert rc == 0, f"training exited {rc}"
'''

REPORT_CELL = '''
import pandas as pd
from metrics import METRIC_KEYS

hist = (pd.read_csv(f"{RUN_DIR}/metrics/history.csv")
          .drop_duplicates(subset="round", keep="last").sort_values("round"))
print(f"{len(hist)} completed rounds")
display(hist[["round", *METRIC_KEYS]].round(6))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for k in ("accuracy", "f1_macro", "f1_weighted"):
    ax[0].plot(hist["round"], hist[k], label=k)
ax[0].set_xlabel("round"); ax[0].set_ylabel("score"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title(f"AFPHA-DAGSNet, {NUM_CLIENTS} clients, global test")
ax[1].plot(hist["round"], hist["t_round_s"] / 60)
ax[1].set_xlabel("round"); ax[1].set_ylabel("minutes"); ax[1].grid(alpha=.3)
ax[1].set_title("round wall time")
plt.tight_layout(); plt.show()

last = int(hist["round"].max())
print(json.dumps(json.load(open(f"{RUN_DIR}/metrics/round_{last:03d}.json")), indent=2))
print("\\nartifact sizes:")
!du -sh {RUN_DIR}/* | sort -h
'''

CAVEATS = """## Caveats that must travel with every number above

* **AFPHA is not fully specified in the paper.** Section 4.6 describes it in prose as
  FedAvg + FedProx + HFL. The proximal coefficient, the adaptive rule, the clustering,
  the participation rate and the optimizer schedule are this project's choices
  (`papers/khan-2025-afpha/rebuild.md`, Proposal A), not the authors' published
  hyperparameters. This is not an exact reproduction of AFPHA.
* **The hierarchy is algebraically FedAvg.** One sample-weighted cluster average followed
  by one sample-weighted server average equals sample-weighted FedAvg over all clients.
  The two levels describe the system, not an optimization difference.
* **These are not the paper's numbers.** The paper reports binary CAN and CIC-IDS results;
  this is 16-class VeReMi with a different metric set. Do not compare them directly.
* **A client is a receiver unit** `(attack_type, scenario, receiver_id)`, not a physical
  vehicle, and the scaler was fitted on all training data before partitioning. This is a
  simulation of the FL algorithm on one machine, not a private distributed system.
* **Global-test scores are not personalized performance**, and the reported result is the
  final round, not the best round selected on test.
* Features are stored at fp16 precision for GPU residency; `running_var` averaging is a
  buffer-merge convention, not a pooled variance.
* Re-running an interrupted round does not reproduce it bit for bit: cuDNN atomics and the
  AMP scale trajectory differ. Recovery restores a valid round, not the lost one.
* The split is temporal, not vehicle-disjoint. Sybil, benign-donor, rate-group and
  `timeDelayAttack` caveats from the dataset documentation still apply.
"""


NB_METADATA = {
    # Kaggle executes notebooks with papermill, which raises
    # "No kernel name found in notebook and no override provided" unless kernelspec is
    # present. nbformat.v4.new_notebook() leaves metadata empty, so set it explicitly.
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python", "version": "3.12"},
}


def new_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata.update(NB_METADATA)
    return nb


def code(src):
    return nbf.v4.new_code_cell(src.strip("\n"))


def md(src):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def module_cell():
    payload = {name: (SRC / name).read_text() for name in MODULES}
    payload.update({name: (ROOT / "architecture" / name).read_text() for name in ASSETS})
    return code("MODULES = " + json.dumps(payload, indent=1))


# Clean-stop deadline per scenario. Measured on T4x2 (calibration 2026-09-07): a
# 20-client round is 319.7 s steady state, so 50 rounds + prepack project to 4.44 h, and
# the 100-client scenario to 8.07 h. calibration.json suggests 4.9/4.9/8.6 -- roughly one
# round of margin, which a 6% slowdown erases. Quota is not the binding constraint here
# (29.6 h remain against 17.0 h expected), so the deadlines below take ~35% headroom
# instead: a needless resume session costs a push and another prepack, while the extra
# headroom costs nothing unless it is actually used. Worst case if all three run to their
# deadline is 21.5 h, still inside the remaining quota, and each stays under Kaggle's 12 h
# session cap. The driver stops before the round that would cross the deadline, so a
# committed round is never lost either way.
MAX_HOURS = {20: 6.0, 50: 6.0, 100: 9.5}


def training_notebook(num_clients):
    batch = 512 if num_clients in (20, 50) else 256
    max_hours = MAX_HOURS[num_clients]
    nb = new_notebook()
    nb.cells = [
        md(f"""
# AFPHA-DAGSNet on VeReMi -- {num_clients} clients

50 communication rounds, one full local epoch per client per round, batch {batch} per
client, 100% participation, fixed clusters of 5. Fresh DAGSNet (395,024 parameters,
PyTorch default initialization). After each aggregation the global model is evaluated on
the **entire** 10,761,343-row test set and all 10 metrics are recorded.

Specification: `papers/khan-2025-afpha/rebuild.md` Proposal A, approved 2026-09-06.
Read the caveats at the bottom before quoting any number.
"""),
        md("## 1. Hardware gate"),
        code(GPU_GATE),
        md("""
## 2. Configuration

`MAX_HOURS` is a clean-stop deadline: the driver stops after the last round that fits,
budgeting on the **worst** round seen so far, and the next session resumes from the
completion marker. Set it below the session limit *and* below remaining weekly quota,
with room to commit artifacts. Take the value from `calibration.json`.

To continue in a later session, attach this notebook's own output through
`kernel_sources` and set `REQUIRE_RESUME = True`; the driver imports the completed
rounds from `/kaggle/input` and refuses to start over silently.
"""),
        code(f"""
NUM_CLIENTS    = {num_clients}
BATCH          = {batch}          # per client, fixed by the specification
ROUNDS         = 50
EVAL_BATCH     = 16384
COMPILE_MODE   = "reduce-overhead"   # falls back to default, then eager, if validation fails
MAX_HOURS      = {max_hours}          # clean-stop deadline; see MAX_HOURS above
REQUIRE_RESUME = False               # True on a continuation session
RUN_DIR        = "/kaggle/working/runs/afpha-dagsnet-{num_clients}client"
WANDB_PROJECT  = "afpha-dagsnet-veremi"
os.makedirs(RUN_DIR, exist_ok=True)
print(json.dumps({{"clients": NUM_CLIENTS, "batch": BATCH, "rounds": ROUNDS,
                  "run_dir": RUN_DIR, "require_resume": REQUIRE_RESUME}}, indent=2))
"""),
        md("## 3. Training modules (generated from `src/`, do not edit here)"),
        module_cell(),
        code(WRITE_SRC),
        md("## 4. Weights & Biases"),
        code(WANDB_CELL),
        md(f"""
## 5. Train

The driver spawns one persistent process per GPU. Each holds the whole {num_clients}-client
training matrix resident in VRAM as fp16 and its own disjoint half of the test set, so no
data crosses PCIe during a round. Clients are handed out longest-first from a shared queue;
aggregation runs in the parent in fixed cluster order, never in completion order.

A client whose update is non-finite, or that skipped more of its AMP steps than the
warm-up allowance, aborts the round rather than being averaged in or silently dropped.
"""),
        code(RUN_CELL),
        md("## 6. Results -- all 10 metrics, every round"),
        code(REPORT_CELL),
        md(CAVEATS),
    ]
    return nb


CALIB_BENCH = '''
# Micro-benchmark: informational, and it proves Inductor works on sm75 before the real
# round pays for it. AMP=False is measured for the record only -- production is always
# AMP=True, so it is never a candidate configuration.
import time, torch
from dagsnet import build_dagsnet
from afpha import ADAM_BETAS, ADAM_EPS, GRAD_CLIP

def bench(batch, mode, amp, steps=300, warmup=120, pool=200_000):
    dev = "cuda:0"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    X = torch.randn(pool, 66, device=dev, dtype=torch.float16)
    Y = torch.randint(0, 16, (pool,), device=dev)
    model = build_dagsnet().to(dev)
    net = model if mode == "eager" else torch.compile(
        model, mode=None if mode == "default" else mode)
    params = list(model.parameters())
    w_g = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=1e-3, betas=ADAM_BETAS, eps=ADAM_EPS, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = torch.nn.CrossEntropyLoss()
    g = torch.Generator(device=dev); g.manual_seed(1)
    ce = torch.zeros((), device=dev)

    def one():
        idx = torch.randint(0, pool, (batch,), device=dev, generator=g)
        xb = X[idx]
        if amp:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(net(xb), Y[idx])
            scaler.scale(loss).backward(); scaler.unscale_(opt)
        else:
            loss = crit(net(xb.float()), Y[idx]); loss.backward()
        torch._foreach_add_([p.grad for p in params],
                            torch._foreach_sub(params, w_g), alpha=0.01)
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        if amp:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        ce.add_(loss.detach())

    try:
        torch._dynamo.reset()
        for _ in range(warmup): one()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(steps): one()
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / steps
        peak = torch.cuda.max_memory_allocated(dev) / 2**30
    except Exception as exc:
        return {"batch": batch, "mode": mode, "amp": amp,
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        del model, net, opt
        torch._dynamo.reset(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    return {"batch": batch, "mode": mode, "amp": amp, "candidate": bool(amp),
            "ms_per_step": round(dt * 1e3, 3), "samples_per_s": round(batch / dt),
            "peak_GiB": round(peak, 3)}

STEP = []
for batch in (512, 256):
    for mode, amp in (("eager", True), ("default", True),
                      ("reduce-overhead", True), ("reduce-overhead", False)):
        r = bench(batch, mode, amp)
        STEP.append(r); print(json.dumps(r), flush=True)

CANDIDATES = [r for r in STEP if r.get("candidate") and "ms_per_step" in r]
assert CANDIDATES, "no AMP configuration survived the micro-benchmark"
'''

CALIB_RUN = '''
# The real thing: the production driver, the real 20-client partition, both GPUs, the
# full test set. Everything the synthetic loop above cannot measure -- parquet decode,
# worker startup, compile validation, resident-VRAM fit, eval sharding, commit I/O --
# is in these numbers.
CAL_DIR = "/kaggle/working/calib"
CMD = [sys.executable, f"{SRC_DIR}/fl_train.py",
       "--clients", "20", "--out", CAL_DIR, "--scratch", "/kaggle/temp/prepack",
       "--arch", SRC_DIR, "--data-search", "/kaggle/input", "--test-search", "/kaggle/input",
       "--rounds", str(CAL_ROUNDS), "--eval-batch", "16384",
       "--compile-mode", "reduce-overhead", "--max-hours", str(CAL_MAX_HOURS),
       "--no-wandb"]
print(" ".join(CMD), flush=True)
t0 = time.time()
proc = subprocess.Popen(CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
CAL_LOG = []
for line in proc.stdout:
    CAL_LOG.append(line.rstrip())
    print(line, end="", flush=True)
rc = proc.wait()
CAL_WALL = time.time() - t0
print(f"\\nexit={rc} after {CAL_WALL/60:.1f} min", flush=True)
assert rc == 0, f"calibration run exited {rc}"
'''

CALIB_PROJECT = '''
# Derive the production numbers from the real run, then project the other two scenarios.
import csv, math, glob
import fldata

cal_cfg = json.load(open(f"{CAL_DIR}/config.json"))
rounds = list(csv.DictReader(open(f"{CAL_DIR}/metrics/history.csv")))
prepack = json.load(open("/kaggle/temp/prepack/prepack.json"))
workers = [json.loads(l.split("workers ready: ")[1]) for l in CAL_LOG if "workers ready: " in l]
compile_modes = sorted({w["compile_mode"] for w in workers[0].values()}) if workers else []
resident = max((w["resident_GiB"] for w in workers[0].values()), default=None) if workers else None

# Round 1 pays compile; steady state is the median of the rest when there is one.
steady = rounds[1:] or rounds
def med(vals):
    v = sorted(vals); return v[len(v) // 2]
t_train = med(float(r["t_train_s"]) for r in steady)
t_eval = med(float(r["t_eval_s"]) for r in steady)
t_round = med(float(r["t_round_s"]) for r in steady)
# history's round_s is train+aggregate+eval; commit I/O is only in the heartbeat, and
# leaving it out of the budget is how a session gets hard-killed mid-write.
beats = [json.loads(l) for l in open(f"{CAL_DIR}/heartbeat.jsonl")]
done_beats = [b for b in beats if b.get("completed")][1:] or [b for b in beats if b.get("completed")]
t_commit = med(b["commit_s"] for b in done_beats)
t_full = t_round + t_commit
steps_20 = int(cal_cfg["steps_per_round"])
# two GPUs share the round's steps
MS_PER_STEP_512 = t_train * 1000.0 * 2 / steps_20

ratio_256 = None
c512 = [r for r in CANDIDATES if r["batch"] == 512 and r["mode"] == "reduce-overhead"]
c256 = [r for r in CANDIDATES if r["batch"] == 256 and r["mode"] == "reduce-overhead"]
if c512 and c256:
    ratio_256 = c256[0]["ms_per_step"] / c512[0]["ms_per_step"]

PROJ = {}
for n in (20, 50, 100):
    try:
        root = fldata.find_root(f"{n}_client/client_stats.json", ["/kaggle/input"]) / f"{n}_client"
    except FileNotFoundError as exc:
        PROJ[n] = {"error": str(exc)[:160]}
        continue
    stats = json.load(open(root / "client_stats.json"))
    rws = [c["rows"] for c in stats["clients"]]
    batch = 512 if n in (20, 50) else 256
    steps = sum(math.ceil(r / batch) for r in rws)
    ms = MS_PER_STEP_512 * (ratio_256 if batch == 256 and ratio_256 else 1.0)
    train_h = steps * ms / 1000 / 2 * 50 / 3600
    other_h = (t_full - t_train) * 50 / 3600           # eval + aggregate + commit
    PROJ[n] = {"clients": len(rws), "rows": int(sum(rws)), "steps_per_round": steps,
               "ms_per_step": round(ms, 3),
               "train_h": round(train_h, 2), "other_h": round(other_h, 2),
               "total_h": round(train_h + other_h, 2),
               "resident_GiB": round(sum(rws) * 66 * 2 / 2**30, 2),
               "suggested_max_hours": round(train_h + other_h + 0.5, 1)}
    print(json.dumps({n: PROJ[n]}, indent=2), flush=True)

total = sum(v["total_h"] for v in PROJ.values() if "total_h" in v)
print(f"\\nprojected GPU hours for all scenarios: {total:.2f}")
print("MEASURED on this session, not extrapolated from another GPU:")
print(json.dumps({"compile_modes": compile_modes, "resident_GiB_after_load": resident,
                  "train_decode_s": prepack.get("train_decode_s"),
                  "test_decode_s": prepack.get("test_decode_s"),
                  "median_train_s": round(t_train, 1), "median_eval_s": round(t_eval, 1),
                  "median_commit_s": round(t_commit, 1),
                  "median_full_round_s": round(t_full, 1),
                  "ms_per_step_batch512": round(MS_PER_STEP_512, 3)}, indent=2))
'''

CALIB_WRITE = '''
CAL = {"devices": names, "capabilities": [list(c) for c in caps],
       "torch": torch.__version__, "cuda": torch.version.cuda,
       "wandb_secret": SECRET,
       "micro_bench": STEP,
       "real_run": {"rounds": len(rounds), "config": cal_cfg,
                    "history": rounds, "wall_s": round(CAL_WALL, 1),
                    "prepack": prepack, "compile_modes": compile_modes,
                    "resident_GiB_after_load": resident,
                    "ms_per_step_batch512": round(MS_PER_STEP_512, 3),
                    "median_train_s": round(t_train, 1), "median_eval_s": round(t_eval, 1),
                    "median_commit_s": round(t_commit, 1),
                    "median_full_round_s": round(t_full, 1),
                    "ratio_batch256_over_512": ratio_256},
       "projection": PROJ}
with open("/kaggle/working/calibration.json", "w") as fh:
    json.dump(CAL, fh, indent=2)
print(json.dumps({k: v for k, v in CAL.items() if k != "real_run"}, indent=2)[:4000])
print("\\nwrote /kaggle/working/calibration.json")
'''


def calibration_notebook():
    nb = new_notebook()
    nb.cells = [
        md("""
# AFPHA-DAGSNet -- T4 calibration

Measures on the real hardware what a laptop GPU could only estimate. The important part
is **section 5: the production driver runs for real** on the 20-client partition with
both GPUs and the full test set, so parquet decode, worker startup, compile validation,
resident-VRAM fit, eval sharding and commit I/O are measured rather than assumed.
Section 4 is a synthetic micro-benchmark kept only to compare backends.

Writes `calibration.json` with a suggested `MAX_HOURS` per scenario.
**Expect roughly 30-50 minutes.** No 50-round training happens here.
"""),
        md("## 1. Hardware gate"),
        code(GPU_GATE),
        code('''
CAL_ROUNDS    = 3      # round 1 pays compile; the rest give the steady-state cost
CAL_MAX_HOURS = 1.5
RUN_DIR = "/kaggle/working"
'''),
        md("""
## 2. Can this session read the `wandb_key` secret?

This decides how the three training runs get launched. A Kaggle API/CLI push has been
observed to create a version with **no** secret attached, which would mean the owner has
to launch each training notebook from the editor with *Save Version > Save & Run All*.
This cell answers the question for the price of one calibration run instead of one
wasted training run. **It never raises** -- a failure here is a result, not an error.
"""),
        code(SECRET_CELL + '''
print()
print("launch path:",
      "secret visible -> either path launches training"
      if SECRET.get("login") else
      "no secret -> the training notebooks fall back to their inline key, which also "
      "makes an API/CLI push self-sufficient")
'''),
        md("## 3. Training modules"),
        module_cell(),
        code(WRITE_SRC),
        md("## 4. Backend micro-benchmark (synthetic; proves Inductor works on sm75)"),
        code(CALIB_BENCH),
        md("""
## 5. The production driver, for real

Three rounds of the actual 20-client run. This is the number that budgets the launch.
"""),
        code(CALIB_RUN),
        md("## 6. Measured cost, and the projection for all three scenarios"),
        code(CALIB_PROJECT),
        md("## 7. Write `calibration.json`"),
        code(CALIB_WRITE),
        md("""
These are measurements of this session's hardware, not a guarantee about the next one.
Re-read live quota before launching a training run and set `MAX_HOURS` from the smaller
of remaining quota, the session limit, and `suggested_max_hours` above.

The 50- and 100-client projections scale the measured 20-client step cost; only the
20-client row is a direct measurement. The batch-256 ratio comes from the synthetic
micro-benchmark, so the 100-client projection carries that extra assumption.
"""),
    ]
    return nb


def metadata(slug, title, sources, kernel_sources=()):
    return {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": MACHINE_SHAPE,
        "dataset_sources": list(sources),
        "competition_sources": [],
        "kernel_sources": list(kernel_sources),
        "model_sources": [],
    }


def read_netrc_key(path=None):
    """W&B key from ~/.netrc. Never returned to a log, only into the notebook file."""
    import netrc
    auth = netrc.netrc(path or Path.home() / ".netrc").authenticators("api.wandb.ai")
    if not auth or not auth[2]:
        raise SystemExit("no api.wandb.ai credential in ~/.netrc; run `wandb login` first")
    return auth[2]


def inject_key(path, key):
    """Replace the placeholder in a written notebook. Returns True if it was there."""
    text = path.read_text()
    if KEY_PLACEHOLDER not in text:
        return False
    path.write_text(text.replace(KEY_PLACEHOLDER, key))
    return True


INLINE_KEY_WARNING = """
!! These notebooks now contain a live W&B API key.
   - Keep them private on Kaggle; the key stays in its version history permanently.
   - Do not commit notebooks/ to a shared or public repository.
   - Rotating the key means rebuilding and re-pushing every notebook.
   Rebuild without it using --no-embed-wandb-key.
"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed-wandb-key", action="store_true",
                    help="leave the placeholder in place and rely on the Kaggle secret")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    # Kaggle derives the kernel slug from the TITLE and ignores a metadata id that
    # disagrees with it, so the title here is chosen to slugify to exactly the id.
    # Otherwise the notebook lands at a URL the status/output calls cannot address.
    plan = [
        ("afpha-dagsnet-calibration", "AFPHA DAGSNet calibration", calibration_notebook(),
         [FL_DS[20], FL_DS[50], FL_DS[100], TEST_DS]),
    ]
    for n in (20, 50, 100):
        plan.append((f"afpha-dagsnet-train-{n}client",
                     f"AFPHA DAGSNet train {n}client",
                     training_notebook(n), [FL_DS[n], TEST_DS]))

    key = None if args.no_embed_wandb_key else read_netrc_key()
    for slug, title, nb, sources in plan:
        stage = OUT / slug
        stage.mkdir(exist_ok=True)
        path = stage / f"{slug}.ipynb"
        nbf.write(nb, path)
        meta = metadata(slug, title, sources)
        assert meta["is_private"] is True, "refusing to embed a credential in a public notebook"
        (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        injected = False if args.no_embed_wandb_key else inject_key(path, key)
        print(f"wrote {stage.relative_to(ROOT)}/  "
              f"({path.stat().st_size/1024:.0f} KiB, {len(sources)} dataset sources"
              f"{', inline W&B key' if injected else ''})")
    if not args.no_embed_wandb_key:
        print(INLINE_KEY_WARNING)
    print(f"push one with:  conda run -n nckh kaggle kernels push -p notebooks/<slug>")


if __name__ == "__main__":
    main()

# Notebook template

Cell order, the resume contract, and the DDP worker. Adapt the contents; keep the order and the contract.

This is a scaffold, not the validated project notebook. Preserve the current notebook's
stability guards, W&B initialization before preparation, and explicit resume preflight.
Validate the exact target file rather than a smoke script's default notebook (see `SKILL.md`).
The results cell must render every completed round and all 10 `METRIC_KEYS`, without pandas
row/column truncation. This is also the schema required when generating `report.md`.

## Cell order

| # | Type | Contents |
|---|---|---|
| 1 | md | Title, paper citation, one-paragraph method summary, **Paper vs. this notebook** deviation table |
| 2 | code | Environment probe — torch version, GPU count/name, VRAM, CPU count, RAM, free disk |
| 3 | code | **`CFG`** — the single knob cell (rounds, epochs, batch, lr, seed) |
| 4 | code | Paths + `RUN_DIR` scaffold + **resume probe** (prints what is already done) |
| 5 | md | Dataset description and the audit findings |
| 6 | code | Load + audit train/test ([`dataset-audit.md`](dataset-audit.md)) |
| 7 | code | Preprocess once → arrays on disk → freeze `meta.json` |
| 8 | md | Architecture, with the paper's equations in LaTeX |
| 9 | code | `%%writefile proj/model.py` |
| 10 | md | Loss, with formula |
| 11 | code | `%%writefile proj/data.py` |
| 12 | md | The 10 metrics, formulas first ([`metrics.md`](metrics.md)) |
| 13 | code | `%%writefile proj/metrics.py` |
| 14 | code | `%%writefile proj/ckpt.py` — the resume contract |
| 15 | code | `%%writefile proj/train_worker.py` — the DDP worker |
| 16 | code | **Launch** `mp.spawn` |
| 17 | code | Results: `history.csv` table, curves, final confusion matrix + per-class report |
| 18 | md | Reporting caveats; compare with the paper only when task and metrics are commensurable |

`%%writefile` keeps the code visible in the notebook *and* importable by spawned child processes — which functions defined in notebook cells are not. That is why the model does not live in a bare cell.

Create the package before any `%%writefile`:

```python
import os, sys, pathlib
PKG = pathlib.Path("/kaggle/working/proj"); PKG.mkdir(parents=True, exist_ok=True)
(PKG / "__init__.py").touch()
sys.path.insert(0, "/kaggle/working")
```

## Cell 2 — environment probe

```python
import os, sys, math, json, time, shutil, platform, subprocess
import numpy as np, torch

print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.version.cuda)
n = torch.cuda.device_count(); print("GPUs:", n)
gpu_names = []
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    gpu_names.append(p.name)
    print(f"  cuda:{i} {p.name} {p.total_memory/2**30:.1f} GB sm_{p.major}{p.minor}")
print("cpu count:", os.cpu_count())
print("RAM GB:", round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30, 1))
print("free on /kaggle/working:", round(shutil.disk_usage('/kaggle/working').free / 2**30, 1), "GB")
assert n == 2, f"Expected 2x T4 — got {n}. Set Accelerator to 'GPU T4 x2' and restart."
assert all("T4" in name for name in gpu_names), f"Expected Tesla T4 devices — got {gpu_names}"
```

The assert is deliberate: a notebook that quietly runs on one GPU produces a result the user cannot compare to anything.

## Cell 3 — the knob cell

```python
from dataclasses import dataclass, asdict, field

@dataclass
class CFG:
    run_name: str = "edl_cmso_v1"

    # --- knobs the user retunes per task ------------------------------
    rounds: int            = 10     # checkpoints produced = rounds
    epochs_per_round: int  = 5
    batch_per_gpu: int     = 256    # PER GPU. global = batch_per_gpu * 2 * grad_accum
    grad_accum: int        = 1
    lr: float              = 1e-3
    weight_decay: float    = 1e-4
    # ------------------------------------------------------------------

    world_size: int        = 2
    amp: bool              = True          # fp16 — T4 has no bf16
    clip: float            = 1.0
    eval_batch: int        = 1024
    seed: int              = 42
    num_classes: int       = -1            # filled from meta.json
    resume: bool           = True
    max_hours: float       = 11.0          # leave commit time before Kaggle's 12 h cap

CFG = CFG()
print(f"global batch = {CFG.batch_per_gpu} x {CFG.world_size} x {CFG.grad_accum} "
      f"= {CFG.batch_per_gpu * CFG.world_size * CFG.grad_accum}")
```

One cell, clearly fenced, so retuning never means hunting through the notebook.

## Resume contract

**The rule: kill the run at any instant and the next run loses at most one round.**

`/kaggle/working` survives cell re-runs inside a session but is wiped when a *new* session starts — so resume resolves from working first, then from attached inputs (see [`kaggle-mcp.md`](kaggle-mcp.md#cross-session-resume-the-part-people-get-wrong)).

```python
%%writefile /kaggle/working/proj/ckpt.py
"""Weights, resume state and the completion marker — three files, one atomic round.

The per-round file holds WEIGHTS ONLY and loads with weights_only=True. Optimizer,
scaler, scheduler and RNG live in a separate resume bundle keyed by the same round,
so keeping every round costs weights and not optimizer moments, and so a reader never
has to unpickle arbitrary objects to look at a checkpoint.
"""
import os, json, shutil, hashlib, random
from pathlib import Path
import numpy as np, torch

SUBDIRS = ("weights", "resume", "complete", "metrics", "preds", "confusion", "reports", "logs")

def run_dir(run_name):
    d = Path("/kaggle/working/runs") / run_name
    for s in SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d

def fingerprint(cfg_dict, keys=("num_classes", "n_features", "batch_per_gpu", "total_rounds",
                                "run_name")):
    """Shape- and schedule-affecting config only. A changed lr may resume; a changed
    num_classes may not, and neither may total_rounds when the LR schedule spans it."""
    payload = json.dumps({k: cfg_dict[k] for k in keys}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def atomic_save(obj, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp); os.replace(tmp, path)     # replace is atomic on POSIX

# --- RNG kept as tensors and primitives so the resume bundle also loads weights_only=True.
# np.random.get_state() hands back a tuple containing an ndarray; left as-is it is the one
# object that forces every future reader onto weights_only=False.
def rng_state():
    npy = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (npy[0], torch.from_numpy(npy[1].copy()), int(npy[2]), int(npy[3]),
                      float(npy[4])),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def set_rng_state(s):
    random.setstate(tuple(s["python"]))
    n = s["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(s["torch"].cpu())
    if s.get("cuda") is not None: torch.cuda.set_rng_state_all(s["cuda"])

# --------------------------------------------------------------------------- write
def save_round(model, opt, scaler, sched, rnd, cfg_dict, metrics, run_name):
    """Rank 0 only. weights -> resume -> marker, in that order. A crash anywhere leaves
    round `rnd` unmarked, so the next session redoes it instead of half-trusting it."""
    d = run_dir(run_name)
    core = getattr(model, "module", model)                    # unwrap DDP -> portable state
    core = getattr(core, "_orig_mod", core)                    # unwrap torch.compile
    weights = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
    atomic_save({"round": int(rnd),
                 "model": weights,                             # tensors only
                 "cfg": cfg_dict,                              # rebuild recipe
                 "fingerprint": fingerprint(cfg_dict),
                 "metrics": metrics},
                d / "weights" / f"round_{rnd:03d}.pt")
    atomic_save({"round": int(rnd),
                 "optim": opt.state_dict(),
                 "scaler": scaler.state_dict() if scaler is not None else None,
                 "sched": sched.state_dict() if sched is not None else None,
                 "rng": rng_state(),
                 "fingerprint": fingerprint(cfg_dict)},
                d / "resume" / f"round_{rnd:03d}.pt")
    (d / "complete" / f"round_{rnd:03d}.done").write_text("")  # absolutely last
    return d / "weights" / f"round_{rnd:03d}.pt"

# --------------------------------------------------------------------------- rebuild
def load_weights(path, build, device="cpu", expect_params=None):
    """Rebuild the model at exactly this checkpoint, from weights alone.

    `build` is the run's own model factory, build(cfg) -> nn.Module. Every assert here
    exists because its failure is otherwise silent: strict=True catches a filtered
    running_mean/var (eval() would then normalize by 0/1 and report nothing), and the
    parameter count catches a cfg that drifted from the one that trained these weights."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build(ck["cfg"])
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v
          for k, v in ck["model"].items()}
    model.load_state_dict(sd, strict=True)
    n = sum(p.numel() for p in model.parameters())
    if expect_params is not None and n != expect_params:
        raise RuntimeError(f"rebuilt {n:,} parameters, expected {expect_params:,}")
    if ck.get("fingerprint") != fingerprint(ck["cfg"]):
        raise RuntimeError("weights file fingerprint disagrees with its own cfg")
    return model.to(device).eval(), ck    # eval(): Dropout off, BN on running stats

# --------------------------------------------------------------------------- resume
def last_complete_round(d):
    done = sorted((d / "complete").glob("round_*.done"))
    return int(done[-1].stem.split("_")[1]) if done else None

def resolve_resume(run_name):
    """working/ first, then any attached input (previous kernel output or a checkpoint
    dataset). Copies what it finds into working/ and returns the last complete round."""
    d = run_dir(run_name)
    if last_complete_round(d) is not None:
        return last_complete_round(d)
    attached = Path("/kaggle/input")
    # Mount depth varies by source and attach path; search any depth, never guess
    # between candidates.
    roots = [] if not attached.exists() else sorted(
        {p.parent for p in attached.rglob("complete/round_*.done") if run_name in p.parts})
    if len(roots) > 1:
        raise RuntimeError(f"Multiple resume trees for {run_name}: {roots}")
    if not roots:
        return None
    src = roots[0].parent
    print(f"[resume] importing from {src}")
    for sub in SUBDIRS:
        peer = src / sub
        if not peer.is_dir(): continue
        for f in peer.iterdir():
            if f.is_file():
                # copyfile, not copy2: a read-only mount's mode would carry across and
                # the first rewrite of history.csv would die with PermissionError.
                shutil.copyfile(f, d / sub / f.name); os.chmod(d / sub / f.name, 0o644)
    return last_complete_round(d)

def load_for_resume(model, opt, scaler, sched, cfg_dict, run_name, device):
    """Returns the round to start from. 0 when there is nothing to resume."""
    last = resolve_resume(run_name)
    if last is None:
        # A notebook that exists ONLY to continue another run must NOT quietly start
        # over. Measured 2026-09-04: a resume push whose glob missed the checkpoint
        # trained from round 0 for its entire 1.9 h budget and produced a worse
        # duplicate of work already done. Absence of an expected checkpoint is a
        # failure, and it must surface before the parquet pass, not after.
        if cfg_dict.get("require_resume"):
            raise SystemExit("require_resume is set and no checkpoint was found — "
                             "fix the attachment, not the round counter.")
        print("[resume] no checkpoint found — starting from round 0"); return 0
    d = run_dir(run_name)
    w = torch.load(d / "weights" / f"round_{last:03d}.pt", map_location="cpu",
                   weights_only=True)
    r = torch.load(d / "resume"  / f"round_{last:03d}.pt", map_location="cpu",
                   weights_only=True)
    if w.get("fingerprint") != fingerprint(cfg_dict):
        raise RuntimeError(
            f"Checkpoint fingerprint {w.get('fingerprint')} != current {fingerprint(cfg_dict)}. "
            "A shape-affecting config changed. Rename CFG.run_name to start a new run, "
            "or restore the original config to resume this one.")
    core = getattr(model, "module", model)
    getattr(core, "_orig_mod", core).load_state_dict(w["model"], strict=True)
    opt.load_state_dict(r["optim"])
    if scaler is not None and r.get("scaler"): scaler.load_state_dict(r["scaler"])
    if sched  is not None and r.get("sched"):  sched.load_state_dict(r["sched"])
    try: set_rng_state(r["rng"])
    except Exception as e: print("[resume] RNG restore skipped:", e)
    print(f"[resume] loaded round {last} -> continuing at round {last + 1}")
    return last + 1
```

Checkpoint budget: **every round is kept**, so before launching print `estimated = ckpt_bytes × rounds` against `shutil.disk_usage('/kaggle/working').free` and warn the user if it will not fit. Thinning `last.pt` is never the answer; reducing `rounds` or pushing older checkpoints to a Dataset is.

For rounds long enough that losing one hurts, add an in-round `heartbeat.pt` written every N steps carrying the same blob plus a step index. Off by default — it costs one write per N steps and complicates the restart path.

## Cell 15 — the DDP worker

```python
%%writefile /kaggle/working/proj/train_worker.py
import os, sys, json, time, socket
from datetime import timedelta
import numpy as np, torch, torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler

sys.path.insert(0, "/kaggle/working")
from proj import ckpt as C
from proj.model import build_model
from proj.data import build_datasets
from proj.metrics import compute_metrics, save_round_artifacts

def ddp_setup(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    # Rank 1 waits while rank 0 evaluates the full test set; 10 min is too short
    # for very large datasets. Device selection must precede NCCL initialization.
    dist.init_process_group("nccl", rank=rank, world_size=world,
                            timeout=timedelta(minutes=60))

def evaluate_full(model, test_ds, cfg, device, num_classes):
    """Rank 0 only, plain DataLoader: the ENTIRE test set, exactly once.
    A DistributedSampler would pad or drop the tail and silently change the score."""
    model.eval()
    loader = DataLoader(test_ds, batch_size=cfg["eval_batch"], shuffle=False,
                        num_workers=2, pin_memory=True, drop_last=False)
    ys, ps, probs = [], [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
                logits = model(x)
            pr = torch.softmax(logits.float(), dim=1)
            probs.append(pr.cpu().numpy()); ps.append(pr.argmax(1).cpu().numpy())
            ys.append(y.numpy())
    model.train()
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)

def main_worker(rank, world, cfg):
    ddp_setup(rank, world)
    session_started = time.time()
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0
    torch.manual_seed(cfg["seed"] + rank); np.random.seed(cfg["seed"] + rank)
    torch.backends.cudnn.benchmark = True          # fixed input shapes

    train_ds, test_ds, num_classes = build_datasets(cfg)
    cfg["num_classes"] = num_classes

    model = build_model(cfg).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank,
                gradient_as_bucket_view=True, find_unused_parameters=False)

    opt    = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = GradScaler("cuda", enabled=cfg["amp"])
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=cfg["rounds"] * cfg["epochs_per_round"])
    crit   = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    # rank 0 alone imports checkpoints from /kaggle/input — concurrent copies race.
    # After the barrier every rank loads the identical file out of working/.
    if cfg["resume"] and is_main:
        C.resolve_resume(cfg["run_name"])
    dist.barrier()
    start_round = C.load_for_resume(model, opt, scaler, sched, cfg,
                                    cfg["run_name"], device) if cfg["resume"] else 0

    workers = max(1, ((os.cpu_count() or 4) - world) // world)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,
                                 drop_last=True, seed=cfg["seed"])
    loader  = DataLoader(train_ds, batch_size=cfg["batch_per_gpu"], sampler=sampler,
                         num_workers=workers, pin_memory=True,
                         persistent_workers=workers > 0, drop_last=True)

    for rnd in range(start_round, cfg["rounds"]):
        t0 = time.time(); model.train(); running = torch.zeros(2, device=device)

        # ---- ROUND BODY -------------------------------------------------
        # Swap this block for the round shape the task uses; everything
        # around it stays identical.
        for ep in range(cfg["epochs_per_round"]):
            sampler.set_epoch(rnd * cfg["epochs_per_round"] + ep)   # new shuffle per epoch
            opt.zero_grad(set_to_none=True)
            for step, (x, y) in enumerate(loader):
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
                    loss = crit(model(x), y) / cfg["grad_accum"]
                scaler.scale(loss).backward()
                if (step + 1) % cfg["grad_accum"] == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                running += torch.tensor([loss.item() * cfg["grad_accum"], 1.0], device=device)
            sched.step()
        # -----------------------------------------------------------------

        dist.all_reduce(running, op=dist.ReduceOp.SUM)     # loss averaged across ranks
        train_loss = (running[0] / running[1]).item()

        if is_main:
            y_true, y_pred, y_prob = evaluate_full(model.module, test_ds, cfg, device, num_classes)
            m = save_round_artifacts(rnd, y_true, y_pred, y_prob, num_classes,
                                     cfg["class_names"], C.run_dir(cfg["run_name"]),
                                     extra={"train_loss": train_loss,
                                            "lr": opt.param_groups[0]["lr"],
                                            "seconds": round(time.time() - t0, 1),
                                            "peak_gb": round(torch.cuda.max_memory_allocated()/2**30, 2)})
            C.save_round(model, opt, scaler, sched, rnd, cfg, m, cfg["run_name"])
            print(f"[round {rnd:03d}] loss {train_loss:.4f} acc {m['accuracy']:.4f} "
                  f"F1_mac {m['f1_macro']:.4f} F1_wtd {m['f1_weighted']:.4f} "
                  f"{m['seconds']}s peak {m['peak_gb']}GB", flush=True)
            torch.cuda.reset_peak_memory_stats()

        dist.barrier()      # after the write: a kill during eval still leaves a valid last.pt

        # All ranks make one collective stop decision. Independent clock checks
        # can send one rank into the next round while another exits, deadlocking DDP.
        round_seconds = time.time() - t0
        elapsed = time.time() - session_started
        should_stop = rnd + 1 < cfg["rounds"] and (
            elapsed + 1.15 * round_seconds >= cfg.get("max_hours", 11.0) * 3600
        )
        stop = torch.tensor(int(should_stop), device=device)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            if is_main:
                path = C.run_dir(cfg["run_name"]) / "logs" / "stopped_early.json"
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps({"last_round": rnd, "elapsed_s": elapsed,
                                           "reason": "wall_clock_budget"}, indent=2))
                os.replace(tmp, path)
                print(f"[stop] wall-clock budget reached after round {rnd}", flush=True)
            dist.barrier()
            break

    dist.destroy_process_group()
```

## Cell 16 — launch

```python
import os, socket, json, importlib, torch.multiprocessing as mp

os.environ["NCCL_P2P_DISABLE"] = "1"        # required on Kaggle T4s
os.environ["NCCL_IB_DISABLE"]  = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
with socket.socket() as s:                  # free port — never hard-code 29500
    s.bind(("", 0)); os.environ["MASTER_PORT"] = str(s.getsockname()[1])

cfg = asdict(CFG); cfg["class_names"] = class_names; cfg["num_classes"] = len(class_names)

# Freeze the effective launch config in committed output. report.md reads this,
# not the editable source notebook, after any crash-driven retuning.
from proj import ckpt as C
config_path = C.run_dir(CFG.run_name) / "config.json"
config_tmp = config_path.with_suffix(".json.tmp")
config_tmp.write_text(json.dumps(cfg, indent=2))
os.replace(config_tmp, config_path)

import proj.train_worker as W
importlib.reload(W)                         # picks up the latest %%writefile

try:
    mp.spawn(W.main_worker, args=(CFG.world_size, cfg), nprocs=CFG.world_size, join=True)
except Exception:
    import traceback; traceback.print_exc()
    print("\nRun failed. Fix the cause above, re-run the %%writefile cell, then re-run "
          "this cell — training resumes from the last completed round.")
    raise
```

`mp.spawn` needs an **importable** target, which is exactly why `main_worker` lives in `proj/train_worker.py`. A `ProcessRaisedException` wraps the child's real traceback — read the inner one.

`!torchrun --nproc_per_node=2 -m proj.train_worker` is an equally valid launcher with cleaner log streaming; keep `mp.spawn` when the notebook needs to hold Python state across the call.

## Round shapes

Only the marked ROUND BODY changes.

**Epoch block** — as written above.

**Federated (FedAvg)** — each rank owns a set of clients, trains them locally from the current global weights, then weights are averaged:

```python
local_states = []
for cid in client_ids_for_rank(rank, world, cfg):
    load_state(model.module, global_state)
    for ep in range(cfg["local_epochs"]):
        train_one_epoch(model, client_loader(cid), opt, scaler, crit, device, cfg)
    local_states.append((n_samples[cid], {k: v.detach().clone()
                                          for k, v in model.module.state_dict().items()}))
global_state = fedavg_all_reduce(local_states, device)   # sample-weighted mean, all ranks agree
model.module.load_state_dict(global_state)
```

Aggregate with `dist.all_reduce` on the flattened weighted sum and on the total sample count, then divide — so every rank ends the round with an identical global model and the checkpoint is rank-independent. State the FedAvg formula in the markdown cell above it:
$w_{t+1}=\sum_{k}\frac{n_k}{n}w_{t+1}^{k}$

**Incremental / continual** — the round selects its own data and the model carries over:

```python
train_ds_r = build_round_dataset(rnd, cfg)      # new stage / task / pseudo-labels
sampler = DistributedSampler(train_ds_r, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
loader  = DataLoader(train_ds_r, batch_size=cfg["batch_per_gpu"], sampler=sampler,
                     num_workers=workers, pin_memory=True, drop_last=True)
for ep in range(cfg["epochs_per_round"]): ...
```

Evaluation stays on the **full, fixed** test set every round regardless of shape — that is what makes the per-round curve comparable.

## Cell 17 — results

```python
import numpy as np
import pandas as pd
from proj.metrics import METRIC_KEYS

history_path = C.run_dir(CFG.run_name) / "metrics" / "history.csv"
if not history_path.exists():
    print("No completed rounds; no test metrics to report. Inspect logs for the stop reason.")
else:
    hist = pd.read_csv(history_path)
    required = ["round", *METRIC_KEYS]
    missing = set(required) - set(hist.columns)
    assert not missing, f"Incomplete metric schema: {sorted(missing)}"
    # Only identical metric rows may be deduplicated; conflicting repeats need investigation.
    assert not hist[required].drop_duplicates().duplicated("round").any(), "Conflicting round metrics"
    hist = hist.drop_duplicates("round", keep="last").sort_values("round")
    values = hist[METRIC_KEYS].to_numpy(dtype=float)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(hist[required].round(6))
    if not hist.empty:
        ax = hist.plot(x="round", y=["accuracy", "f1_macro", "f1_weighted"],
                       marker="o", figsize=(9, 4))
        ax.set_title("Test metrics per round"); ax.grid(alpha=.3)
```

When completed rounds exist, finish with the final round's confusion matrix, the per-class `classification_report`, and the required reporting caveats. Otherwise show available stop diagnostics and skip final-round file reads. Add a paper-comparison table only when task, data, class semantics, and metric definitions are commensurable and the project context allows it. Every checkpoint stays in `runs/<run_name>/checkpoints/` — `rounds` files, one per round, as committed notebook output.


## Kaggle metadata that actually takes effect

Three things measured by pushing, not read from documentation:

1. **A notebook without `metadata.kernelspec` never runs.** Kaggle executes through
   papermill, which raises `ValueError: No kernel name found in notebook and no override
   provided` before the first cell. `nbformat.v4.new_notebook()` leaves metadata empty, so
   a generated notebook must set it:

   ```python
   nb.metadata.update({
       "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
       "language_info": {"name": "python", "version": "3.12"},
   })
   ```

   The failure costs no accelerator quota — the kernel errors before any cell executes —
   but it does burn a version and a round trip. Assert kernelspec in the notebook validator.

2. **The slug comes from the title, not from the metadata `id`.** Pushing a notebook titled
   "AFPHA DAGSNet calibration" with `id: <owner>/afpha-calibration` creates
   `<owner>/afpha-dagsnet-calibration` and prints only a soft warning. Every later status,
   output and session call then addresses a slug that does not exist. Choose titles that
   slugify to exactly the id, and check it in the generator.

3. **`kernel-metadata.json` supports `machine_shape`.** `kaggle kernels push` maps
   `--accelerator` onto it and falls back to the metadata value
   (`request.machine_shape = acc if acc else get_or_default(meta, "machine_shape")`).
   Set `"machine_shape": "NvidiaTeslaT4"` so a push cannot silently land on a P100 and
   trip a two-GPU assertion. `"NvidiaTeslaP100"` is the other value.

`kaggle kernels push -p <dir>` requires the metadata file to be named exactly
`kernel-metadata.json`, so give every notebook its own staging directory rather than
keeping several scenario-named metadata files side by side.

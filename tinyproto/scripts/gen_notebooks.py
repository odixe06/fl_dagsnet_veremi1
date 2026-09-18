#!/usr/bin/env python
"""Generate the Kaggle notebooks for TinyProto-FP from the tested modules in `src/`.

There is one source of truth for the algorithm: `src/*.py`. The notebooks embed those files
verbatim through `%%writefile`, which keeps the code readable inside the notebook AND importable
by the spawned worker processes (functions defined in notebook cells are not).

    python scripts/gen_notebooks.py [--only calib] [--out papers/tinyproto-lee-2026/notebook]

Each notebook gets its own staging directory because `kaggle kernels push -p DIR` requires the
metadata file to be named exactly `kernel-metadata.json`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODULES = ["model.py", "cps.py", "protos.py", "metrics.py", "data.py",
           "ckpt.py", "evaluate.py", "fl_worker.py", "driver.py", "sweep.py"]

DATASETS = {
    20: "odixe0502/veremi-fl-20client",
    50: "odixe0502/veremi-fl-50client",
    100: "odixe0502/veremi-fl-100client",
}
CENTRALIZED = "odixe0502/veremi-nextgen2026-centralized"


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


# ---------------------------------------------------------------------------
# shared cells
# ---------------------------------------------------------------------------

HEADER_MD = """# {title}

**TinyProto-FP** — Lee & Choi, *Communication-Efficient Heterogeneous Federated Learning with
Sparse Prototypes in Resource-Constrained Environments* (AAAI-26 submission 02846), §4.1–4.3,
Algorithm 1. Classifier at every client: **DAGSNet** (Khan et al. 2025, *Sci. Rep.*
`10.1038/s41598-025-94445-9` §4.10, Eq. 38–48), 395,024 parameters, 66 raw features in,
16 logits out.

Each client trains one local epoch on its own shard, computes class-mean activations of the
penultimate layer (Eq. 3), sparsifies them with a fixed per-class binary mask (CPS, Eq. 7–8),
scales them by its class counts (APS, Eq. 10) and uploads only the `s` non-zero entries. The
server averages those over the clients holding each class and sends them back; the next round
adds `λ·R_i` (Eq. 5, 11) to the cross-entropy. **No model weights are ever exchanged** — every
client keeps its own model for the whole run, which is why all 10 metrics are measured per
client on the full fixed test set every round.

## Paper vs. this notebook

| item | paper | here | why |
|---|---|---|---|
| dataset | CIFAR-10/100, GTSRB-43, Flowers-102, Tiny-ImageNet-200, AG News | VeReMi NextGen, 16 classes, 43,045,415 train / 10,761,343 test | project target |
| client architectures | 4 different lightweight CNNs across clients | **DAGSNet at every client** | user requirement — this build does **not** test the paper's model-heterogeneity claim |
| feature extractor | separate extraction stage | **dropped**, 66 raw z-scored features | user requirement |
| feature dim `d` | 500 | **256** (DAGSNet `head[3]`) | fixed by the architecture |
| CPS dim `s` | 50 (= 90% compression at d=500) | **50** (= 80.5% at d=256) | keeps the paper's value; its two statements conflict at d=256 |
| clients / rounds / batch | 20 / 300 / 32 | **{n_clients} / {rounds} / {batch}** | user specification |
| Dirichlet α | 0.1 | **0.5** | the partition that exists |
| optimizer | not named (reference code uses SGD), lr 0.01 | **AdamW, lr 1e-3, wd 1e-4**, no scheduler | user decision; the configuration proven on this model and data |
| ρ(·,·) | "Euclidean distance" | masked MSE over the `s` active dims, per sample | reduces to FedProto's `nn.MSELoss` when `s = d` |
| prototype timing | Eq. (3) is written for one θ_i | one inference pass **after** the local epoch, in `eval()` | a local epoch here is up to 11,506 steps, not ~59 |
| μ | 5 published values; method "in the appendix" — **the PDF has no appendix** | **grid search on a train-side validation split** | user decision; this build's realization, not the authors' method |
| seeds | 3, averaged | **1 (seed 42)** | GPU budget — **no replication** |

Full decision table with every gap named: `papers/tinyproto-lee-2026/rebuild.md`.
"""

ENV_CELL = '''import os, sys, json, time, math, shutil, platform, subprocess
import numpy as np, torch

SESSION_STARTED_AT = time.time()
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.version.cuda)
try:
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout)
except (OSError, subprocess.CalledProcessError) as e:
    print("nvidia-smi probe:", type(e).__name__)
n = torch.cuda.device_count()
names, caps = [], []
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    names.append(p.name); caps.append((p.major, p.minor))
    print(f"  cuda:{i} {p.name} {p.total_memory/2**30:.1f} GB sm_{p.major}{p.minor} SMs {p.multi_processor_count}")
print("cpu:", os.cpu_count(), "| RAM GB:",
      round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
      "| free on /kaggle/working GB:", round(shutil.disk_usage("/kaggle/working").free / 2**30, 1))
print("free on /kaggle/temp GB:", round(shutil.disk_usage("/kaggle/temp").free / 2**30, 1)
      if os.path.isdir("/kaggle/temp") else "n/a")

# An invalid machine_shape is silently coerced to a single P100 -- no error, no warning. This
# assert is the cheap failure; a run that proceeds on an unconfirmed accelerator is not.
assert n == 2, f"expected 2 GPUs, got {n}. Set machine_shape=NvidiaTeslaT4."
assert all("T4" in x for x in names), f"expected Tesla T4, got {names}"
assert all(c == (7, 5) for c in caps), f"expected sm_75, got {caps}"
print("\\nOK: 2 x Tesla T4 (sm_75) confirmed before any training.")
'''

WRITE_PKG_HEAD = '''import pathlib, sys
PKG = pathlib.Path("/kaggle/working/proj"); PKG.mkdir(parents=True, exist_ok=True)
(PKG / "__init__.py").write_text("")
sys.path.insert(0, "/kaggle/working")
print("package dir:", PKG)
'''

RESOLVE_CELL = '''from pathlib import Path
import json, numpy as np
sys.path.insert(0, "/kaggle/working")
from proj.data import find_client_root, find_test_root, footer_fingerprint, footer_rows, plan_gpus
from proj.model import CFG as MODEL_CFG, FlatPacker, build_model

SEARCH = [Path("/kaggle/input")]
CLIENT_ROOT = find_client_root(SEARCH, CFG["n_clients"])
TEST_DIR    = find_test_root(SEARCH)
print("client root:", CLIENT_ROOT)
print("test dir   :", TEST_DIR)

# Resolve sidecars from the selected datasets, never an unrelated attached run.
DATA_ROOT = TEST_DIR.parent
SCALER_PATH = DATA_ROOT / "scaler.json"
META = json.loads((DATA_ROOT / "label_mapping.json").read_text())
CLASS_NAMES = META["classes"]
FEATURE_COLS = json.loads((DATA_ROOT / "feature_schema.json").read_text())["feature_columns"]
assert META["class_to_label"] == {c: i for i, c in enumerate(CLASS_NAMES)}
assert META["label_to_class"] == {str(i): c for i, c in enumerate(CLASS_NAMES)}
assert FEATURE_COLS == EXPECTED_FEATURES and CLASS_NAMES == EXPECTED_CLASSES, "architecture data contract changed"
STATS = json.loads((CLIENT_ROOT / "client_stats.json").read_text())
rows = {}
for cid in range(CFG["n_clients"]):
    fs = sorted((CLIENT_ROOT / "train" / f"client_id={cid:03d}").glob("*.parquet"))
    assert fs, f"client {cid} has no parquet files"
    rows[cid] = footer_rows(fs)
side = {c["client_id"]: c["rows"] for c in STATS["clients"]}
bad = {c: (rows[c], side.get(c)) for c in rows if side.get(c) != rows[c]}
assert not bad, f"footer rows disagree with client_stats.json: {bad}"
TOTAL = sum(rows.values())
assert TOTAL == 43_045_415, f"train rows {TOTAL:,} != 43,045,415"
TEST_ROWS = footer_rows(sorted(TEST_DIR.glob("*.parquet")))
assert TEST_ROWS == 10_761_343, f"test rows {TEST_ROWS:,} != 10,761,343"
print(f"train {TOTAL:,} rows across {CFG['n_clients']} clients "
      f"(min {min(rows.values()):,} max {max(rows.values()):,}) | test {TEST_ROWS:,} rows")

ASSIGNMENT = plan_gpus(rows, 2)
load = [sum(rows[c] for c in b) for b in ASSIGNMENT]
print(f"GPU split: {len(ASSIGNMENT[0])}/{len(ASSIGNMENT[1])} clients, "
      f"{load[0]:,}/{load[1]:,} rows, imbalance {abs(load[0]-load[1])/TOTAL*100:.3f}%")

STEPS = sum(math.ceil(r / CFG["batch"]) for r in rows.values())
print(f"optimizer steps per round: {STEPS:,}  ({STEPS*CFG['rounds']:,} over {CFG['rounds']} rounds)")
'''

WANDB_PRELUDE = '_INLINE_WANDB_KEY = ""\n'

WANDB_CELL = '''"""Live progress. W&B streams scalars WHILE the kernel runs; Kaggle publishes its
output only when the kernel stops, so without this a 5-hour sweep is invisible until it ends.

W&B is strictly off the training path: every failure here is swallowed and training continues.
Nothing reported from W&B is a result -- checkpoints, confusion matrices and history.csv still
only exist once the kernel commits, and those are what any number gets quoted from.
"""
import os
WANDB = None
WANDB_PROJECT = "tinyproto-fp"
try:
    import wandb
    key = None
    try:            # a secret is preferred when the notebook was launched from the editor
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("wandb_key")
    except Exception:
        key = None
    key = key or (_INLINE_WANDB_KEY or None)
    if key:
        os.environ["WANDB_API_KEY"] = key            # for wandb.Api() later
        # The key MUST be passed as an argument: relogin does not consult WANDB_API_KEY, and a
        # Kaggle kernel has neither a netrc nor a terminal to fall back to.
        wandb.login(key=key, relogin=True)
        WANDB = wandb
        print("W&B: logged in; runs will stream to project", WANDB_PROJECT)
    else:
        print("W&B: no key available; continuing without live metrics")
except Exception as e:
    WANDB = None
    print("W&B disabled:", type(e).__name__, str(e)[:120])
'''


CONFIG_BUILD = '''from proj.ckpt import file_digest
cfg = dict(CFG)
cfg.update({
    "class_names": CLASS_NAMES, "feature_cols": FEATURE_COLS,
    "session_started_at": globals().get("SESSION_STARTED_AT", time.time()),
    "model_cfg": MODEL_CFG,
    "packer_manifest": FlatPacker(build_model(66, MODEL_CFG)).manifest(),
    "assignment": ASSIGNMENT, "devices": ["cuda:0", "cuda:1"],
    "data_fingerprint": footer_fingerprint(sorted((CLIENT_ROOT / "train").rglob("*.parquet"))),
    "test_fingerprint": footer_fingerprint(sorted(TEST_DIR.glob("*.parquet"))),
    # Part of the mu-selection signature: swapping the scaler changes what a prototype means,
    # so a mu chosen under a different scaler must stop validating.
    "scaler_fingerprint": file_digest(Path(SCALER_PATH)),
})

if "MU_GRID" in globals():          # sweep notebooks only
    from proj.sweep import MU_PROTOCOL
    _knob = {"grid": MU_GRID, "rounds": SWEEP_ROUNDS,
             "val_fraction": VAL_FRAC, "val_seed": VAL_SEED}
    _frozen = {k: MU_PROTOCOL[k] for k in _knob}
    assert _knob == _frozen, (
        f"knob cell {_knob} disagrees with the frozen protocol {_frozen}. Change both together "
        "and bump MU_PROTOCOL['version'], or production will reject this sweep's output.")
    print(f"mu protocol v{MU_PROTOCOL['version']} confirmed: {_knob}")
paths = {"client_root": str(CLIENT_ROOT), "test_dir": str(TEST_DIR), "scaler": str(SCALER_PATH)}

# Production refuses placeholder mu and uses only a completed sweep for THIS scenario.
if CFG.get("require_mu_selection", False):
    from proj.sweep import selected_mu
    explicit = CFG.get("mu_selection_path")
    hits = [Path(explicit)] if explicit else sorted(Path("/kaggle/input").rglob("mu_sweep.json"))
    matches = []
    for hit in hits:
        doc = json.loads(hit.read_text())
        if doc.get("scenario") == cfg["scenario"]:
            matches.append((hit, doc))
    assert len(matches) == 1, "Attach exactly one completed mu-sweep output for this scenario, or set mu_selection_path"
    mu_path, doc = matches[0]
    value, winner = selected_mu(doc, cfg)
    cfg.update(mu_kind="absolute", mu_value=value, mu_selection=doc)
    print("selected mu:", value, "from", mu_path)

# Include two prediction rules, optimizer moments and overlapping resume writes.
# A sweep keeps ONE COMPLETE RUN PER GRID CANDIDATE, so the budget is per-candidate cost times
# the number of candidates, plus the shared validation cache. Sizing this for a single candidate
# under-reports the sweep by the size of the grid.
manifest = cfg["packer_manifest"]
per_round = cfg["n_clients"] * (4*(manifest["n_params"] + manifest["n_buffers"]) + 8*manifest["n_ints"])
pred_bytes = cfg["n_clients"] * TEST_ROWS * 2 * len(cfg.get("save_preds_rounds", []))
resume_bytes = cfg["n_clients"] * manifest["n_params"] * 8 * 2
RUNS_KEPT = len(globals().get("MU_GRID", [1]))          # 1 for production, len(grid) for a sweep
per_run = per_round * cfg["rounds"] + pred_bytes + resume_bytes
cache_bytes = 0
if "MU_GRID" in globals():
    # train_index (int64 row ids for the 98% kept) plus the shared fp16 validation matrix.
    cache_bytes = int(TOTAL * (1 - VAL_FRAC) * 8) + int(TOTAL * VAL_FRAC * (len(FEATURE_COLS) * 2 + 1))
needed = per_run * RUNS_KEPT + cache_bytes + 512 * 2**20
free = shutil.disk_usage("/kaggle/working").free
print(f"estimated output budget: {needed/2**30:.2f} GiB "
      f"({RUNS_KEPT} run(s) x {per_run/2**30:.2f} GiB"
      + (f" + {cache_bytes/2**30:.2f} GiB split cache" if cache_bytes else "")
      + f" + margin); free {free/2**30:.1f} GiB")
assert needed < free * 0.9, (
    f"insufficient artifact disk space: need {needed/2**30:.2f} GiB of {free/2**30:.1f} GiB free")
'''

RESULTS_CELL = '''import pandas as pd
from proj.metrics import METRIC_KEYS, RULES
from proj import ckpt as C

OUT_ROOT = Path("/kaggle/working")
d = C.run_dir(OUT_ROOT, CFG["run_name"])
hp = d / "metrics" / "history.csv"
if not hp.exists():
    print("No completed rounds. Check logs/ for the stop reason.")
else:
    hist = pd.read_csv(hp).sort_values("round")
    assert not hist["round"].duplicated().any(), "duplicate round in history"
    print(f"Completed {C.last_complete(d)}/{CFG['rounds']} rounds")
    for rule in RULES:
        cols = ["round"] + [f"{rule}_{k}" for k in METRIC_KEYS]
        miss = set(cols) - set(hist.columns)
        assert not miss, f"incomplete metric schema for {rule}: {sorted(miss)}"
        v = hist[cols[1:]].to_numpy(float)
        assert np.isfinite(v).all() and ((v >= -1e-9) & (v <= 1 + 1e-9)).all()
        print(f"\\n=== rule: {rule}  (mean over all {CFG['n_clients']} clients, full test set) ===")
        with pd.option_context("display.max_rows", None, "display.max_columns", None,
                               "display.width", 250):
            display(hist[cols].round(6))
    best = int(hist["round"].max())
    print(f"\\nlast completed round (primary report): {best}")
    ax = hist.plot(x="round", y=["proto_f1_macro", "proto_accuracy",
                                 "clf_f1_macro", "clf_accuracy"],
                   marker="o", figsize=(10, 4))
    ax.set_title(f"{CFG['scenario']}: mean over clients, full {10_761_343:,}-row test set")
    ax.grid(alpha=.3)

    m = json.loads((d / "metrics" / f"round_{best:03d}.json").read_text())
    print(f"\\n--- per-class, round {best}, rule=proto (pooled over clients) ---")
    display(pd.DataFrame(m["per_class"]["proto"]).round(4))
    print("\\n--- spread across clients, round", best, "---")
    for rule in RULES:
        a = m["aggregate"][rule]
        print(f"  {rule:5s} f1_macro mean {a['mean_over_clients']['f1_macro']:.4f} "
              f"std {a['std_over_clients']['f1_macro']:.4f} "
              f"min {a['min_over_clients']['f1_macro']:.4f} "
              f"max {a['max_over_clients']['f1_macro']:.4f} | "
              f"pooled {a['pooled']['f1_macro']:.4f}")
    print(f"\\ncommunication: {m['communication']['params_per_round']:,} params/round "
          f"({m['communication']['compression_vs_fedproto']:.2f}x below dense FedProto)")
'''

CAVEATS_MD = """## Reporting caveats — required beside every number above

1. **Not a reproduction of the paper.** Different dataset, task, class count, client
   architecture, α, round count and batch size. The method is inherited; the numbers are not
   comparable to Table 1 and must not be placed in the same table.
2. **`f1_macro`, not `accuracy`.** The test set is imbalanced 41:1 and two classes hold 44.5%
   of it. `precision_micro = recall_micro = f1_micro = recall_weighted = accuracy` is an
   identity of single-label multi-class classification, not a duplicated column.
3. **Personalized models.** There is no global model. Every number is a mean over the clients'
   own models on the same fixed global test set, which measures how well a personalized model
   generalizes globally — not how it performs on its own local distribution.
4. **One seed, one run.** No replication; the paper averages three seeds.
5. **μ came from a grid search on a train-side validation split**, not from the authors'
   (unpublished) selection method.
6. **Features are stored fp16** on the GPU. Measured max |x| = 570.44 against an fp16 ceiling of
   65504, so no saturation — but the stored value is genuinely quantized.
7. **`scaler.json` was fit on all 43 M training rows**, i.e. across every client. A real
   federated client has no such global statistic. This leak predates this build.
8. **Dataset caveats carry over**: simulation-time split, Sybil flow leakage, `benign` drawn
   from attack-free runs, 3,338,358 ambiguous rows dropped upstream, clients are receiver units
   rather than real vehicles, and the test set contains only the `_7` scenarios.
"""


# ---------------------------------------------------------------------------
# notebook assembly
# ---------------------------------------------------------------------------

def module_cells() -> list:
    cells = [nbf.v4.new_code_cell(WRITE_PKG_HEAD)]
    for m in MODULES:
        body = (SRC / m).read_text()
        cells.append(nbf.v4.new_code_cell(f"%%writefile /kaggle/working/proj/{m}\n{body}"))
    return cells


def base_notebook(title: str, cfg_cell: str, n_clients: int, rounds: int, batch: int,
                  launch_cell: str, extra_md: str = "") -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(HEADER_MD.format(title=title, n_clients=n_clients,
                                                  rounds=rounds, batch=batch) + extra_md),
        nbf.v4.new_code_cell(ENV_CELL),
        nbf.v4.new_markdown_cell("## The knob cell\n\nEverything retunable lives here."),
        nbf.v4.new_code_cell(cfg_cell),
        nbf.v4.new_markdown_cell(
            "## Algorithm source\n\n`%%writefile` keeps the code visible here **and** importable "
            "by the spawned worker processes — functions defined in notebook cells are not. "
            "These files are byte-identical to `src/` in the repository, which is where they are "
            "tested."),
        *module_cells(),
        nbf.v4.new_markdown_cell(
            "## Resolve the data\n\nRoots are found by **sentinel**, never by a fixed mount "
            "prefix: Kaggle nests dataset mounts by kind and owner and has changed that nesting "
            "before. Row counts come from parquet footers and are cross-checked against "
            "`client_stats.json` — a silent partial mount would otherwise look like a valid run."),
        nbf.v4.new_code_cell("EXPECTED_FEATURES = " + repr(json.loads((ROOT / "knowledge/meta.json").read_text())["feature_cols"]) + "\nEXPECTED_CLASSES = " + repr(json.loads((ROOT / "knowledge/meta.json").read_text())["class_names"]) + "\n" + RESOLVE_CELL),
        nbf.v4.new_code_cell(WANDB_PRELUDE + WANDB_CELL),
        nbf.v4.new_code_cell(CONFIG_BUILD),
        nbf.v4.new_markdown_cell("## Run"),
        nbf.v4.new_code_cell(launch_cell),
        nbf.v4.new_markdown_cell("## Results — all 10 metrics, every completed round"),
        nbf.v4.new_code_cell(RESULTS_CELL),
        nbf.v4.new_markdown_cell(CAVEATS_MD),
    ]
    nb.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"},
        # The accelerator also lives in the notebook document, not only in kernel-metadata.json.
        # The bounded validation notebook carried this block and got 2 x T4; the notebooks that
        # omitted it are the ones that came up with zero GPUs. Keep both in agreement.
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                   "isInternetEnabled": True, "language": "python",
                   "sourceType": "notebook"},
    })
    return nb


TRAIN_CFG = '''CFG = {{
    "run_name": "tinyproto_fp_{n}client",
    "scenario": "{n}client",
    "n_clients": {n},

    # ---- the paper's protocol -----------------------------------------------
    "rounds": {rounds},            # communication rounds T
    "local_epochs": 1,             # §5.1
    "lam": 1.0,                    # λ in Eq. (5); §5.1 "we set λ = 1 following FedProto"
    "cps_s": 50,                   # CPS dimension s; §5.1 default
    "mask_seed": 42,

    # ---- APS scaling constant μ (Eq. 11) ------------------------------------
    # These two are PLACEHOLDERS. require_mu_selection below makes the config cell read the
    # winner out of the attached mu_sweep.json and overwrite them, but only if that document
    # matches this scenario and this data fingerprint. Do not hand-copy a number here: a
    # hand-copied mu is not checked against anything.
    "mu_kind": "{mu_kind}",
    "mu_value": {mu_value},
    "require_mu_selection": True,
    "mu_selection_path": None,  # optional exact path to the scenario-specific mu_sweep.json

    # ---- this build's choices ----------------------------------------------
    "batch": {batch},              # per client; user specification
    "eval_batch": 8192,
    "lr": 1e-3, "weight_decay": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
    "clip": 1.0, "amp": True, "compile": True,
    "eval_group": 1,               # MEASURED on 2 x T4 (calibration 2026-09-08): the vmapped
                                   # path is SLOWER than folded-sequential (0.50/0.64/0.62x at
                                   # G=2/4/8) and differs numerically (0.028% of confusion
                                   # entries). Both reasons independently rule it out. Do not
                                   # raise this without re-measuring.
    "seed": 42,
    "num_classes": 16, "n_features": 66, "feature_dim": 256,

    # ---- session management -------------------------------------------------
    "max_hours": {max_hours},      # stop before Kaggle's 12 h cap, leaving commit time
    "expected_round_s": {expected_round_s},   # calibration-derived; only used to gate the FIRST
                                   # round of a session, before this run has timed one itself.
                                   # 20-client is measured; 50/100 are projections from it.
    "rounds_this_session": None,   # set an int to force an earlier session boundary
    "require_resume": {require_resume},  # True on a push whose ONLY purpose is to continue
    "import_roots": ["/kaggle/input"],
    "save_preds_rounds": [{rounds}],
}}
for k, v in CFG.items():
    print(f"  {{k:22s}} {{v}}")
'''

LAUNCH_CELL = '''import importlib, proj.driver as D
importlib.reload(D)

OUT_ROOT = Path("/kaggle/working")

# Live progress for production. A 50-round run is split across sessions and accounts, so the
# W&B run id is the RUN NAME, not the session: resume="allow" makes every session append to one
# continuous chart, and driver.run logs with step=rnd, which stays monotonic across the split.
# Telemetry is off the training path -- any failure here leaves training untouched.
wb = None
if WANDB is not None:
    try:
        wb = WANDB.init(project=WANDB_PROJECT, id=cfg["run_name"], name=cfg["run_name"],
                        group=f"train-{cfg['scenario']}", job_type="production",
                        resume="allow", reinit=True,
                        config={"scenario": cfg["scenario"], "rounds": cfg["rounds"],
                                "batch": cfg["batch"], "mu_value": cfg["mu_value"],
                                "n_clients": cfg["n_clients"]})
        print("W&B run:", getattr(wb, "url", "(no url)"))
    except Exception as e:
        print("W&B init failed:", type(e).__name__, str(e)[:120]); wb = None

t0 = time.time()
try:
    res = D.run(cfg, paths, OUT_ROOT, wandb_run=wb)
except Exception:
    import traceback; traceback.print_exc()
    print("\\nRun failed. Fix the cause above, restart the kernel after changing source, then run every "
          "cell -- training continues from the last completed round.")
    raise
finally:
    # finish() can raise on a flaky network. The rounds are already committed; refusing to let
    # a telemetry teardown mask the real outcome of the session.
    if wb is not None:
        try:
            wb.finish()
        except Exception as e:
            print("W&B finish failed (ignored):", type(e).__name__, str(e)[:120])
print(json.dumps(res, indent=2), f"\\ntotal {time.time()-t0:.0f}s")
'''


# Round-time basis: 20-client is MEASURED (763 s steady state, eval_batch 16384, calibration
# 2026-09-08). 50 and 100 are projected from it -- eval scales with client count, train scales
# with steps at the measured per-step cost. Rounded up, and only ever used as a first-round
# admission estimate that the run's own timings replace.
EXPECTED_ROUND_S = {20: 800.0, 50: 1300.0, 100: 2500.0}


def train_notebook(n: int, rounds: int, batch: int, max_hours: float,
                   mu_kind: str, mu_value: str,
                   require_resume: bool = False) -> nbf.NotebookNode:
    title = f"TinyProto FP train {n}client"
    cfg = TRAIN_CFG.format(n=n, rounds=rounds, batch=batch, max_hours=max_hours,
                           expected_round_s=EXPECTED_ROUND_S[n],
                           mu_kind=mu_kind, mu_value=mu_value,
                           require_resume=require_resume)
    return base_notebook(title, cfg, n, rounds, batch, LAUNCH_CELL)


# ---------------------------------------------------------------------------
# calibration notebook
# ---------------------------------------------------------------------------

CALIB_CFG = '''CFG = {
    "run_name": "tinyproto_fp_calib",
    "scenario": "20client", "n_clients": 20,
    "rounds": 3,                   # calibration only: 3 rounds separates warm-up from steady state
    "local_epochs": 1, "lam": 1.0, "cps_s": 50, "mask_seed": 42,
    "mu_kind": "inv_mean_nij", "mu_value": 1.0,
    "batch": 512, "eval_batch": 8192,
    "lr": 1e-3, "weight_decay": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
    "clip": 1.0, "amp": True, "compile": True, "eval_group": 1,
    "seed": 42, "num_classes": 16, "n_features": 66, "feature_dim": 256,
    "max_hours": 11.0, "rounds_this_session": None, "require_resume": False,
    "import_roots": [], "save_preds_rounds": [],
}
BENCH_ONLY = False     # True skips the 3-round run and only produces the micro-benchmarks
BENCH_SMOKE = False    # True shrinks every benchmark and skips torch.compile.
                       # Set by tests/notebook_sim.py: Inductor's compile workers alone exceed
                       # the 7.6 GB of the development laptop and take WSL down with them.
for k, v in CFG.items(): print(f"  {k:22s} {v}")
'''

CALIB_BENCH = '''"""Micro-benchmarks that decide `max_hours`, `eval_group` and whether compile is on.

Every number in `papers/.../perf notes` from another device is a RATIO guide. These are the
measurements the production configs are actually derived from.
"""
import importlib, os, torch, time
# Kaggle gives ~4 vCPU; parallel Inductor workers buy nothing here and cost a lot of RAM.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
import proj.model as M, proj.evaluate as E, proj.data as DD
importlib.reload(M); importlib.reload(E)
from torch.amp import autocast, GradScaler

dev = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True
report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}

# ---- 1. prepack: parquet -> resident fp16, for the two clients on GPU 0 -----
t0 = time.time()
sub = ASSIGNMENT[0][:3]
X, y, spans, aud = DD.decode_clients(CLIENT_ROOT, sub, FEATURE_COLS, progress=None)
report["prepack"] = {"clients": len(sub), "rows": int(aud["rows"]),
                     "seconds": round(time.time() - t0, 1),
                     "rows_per_s": int(aud["rows"] / max(time.time() - t0, 1e-9)),
                     "abs_max": aud["abs_max"], "fp16_safe": aud["fp16_safe"]}
print("prepack:", report["prepack"])
est = TOTAL / report["prepack"]["rows_per_s"]
print(f"  -> whole train set would take ~{est:.0f}s per worker")

t0 = time.time()
import json as _json
Xt, yt, aud_t = DD.decode_test(TEST_DIR, FEATURE_COLS,
                               _json.loads(Path(SCALER_PATH).read_text()), progress=None)
report["prepack_test"] = {"rows": int(aud_t["rows"]), "seconds": round(time.time() - t0, 1),
                          "raw_mean_f_snd_spd": aud_t["raw_mean_f_snd_spd"],
                          "abs_max": aud_t["abs_max"]}
print("test prepack:", report["prepack_test"])

Xg = torch.from_numpy(X).to(dev); Yg = torch.from_numpy(y).to(dev).long()
Xtg = torch.from_numpy(Xt).to(dev); Ytg = torch.from_numpy(yt).to(dev).long()
del X, y, Xt, yt
print("resident GiB:", round((Xg.nelement() + Xtg.nelement()) * 2 / 2**30, 2))

# ---- 2. training step time: eager vs compile, AMP on/off, batch 512 and 256 --
def bench_step(model, fn, bs, amp, iters=(5 if BENCH_SMOKE else 120)):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, fused=True)
    sc = GradScaler("cuda", enabled=amp); crit = torch.nn.CrossEntropyLoss()
    idx = torch.randint(0, Xg.shape[0], (bs,), device=dev)
    xb, yb = Xg[idx].float(), Yg[idx]
    for _ in range(2 if BENCH_SMOKE else 15):
        with autocast("cuda", dtype=torch.float16, enabled=amp):
            _, lo = fn(xb)
        sc.scale(crit(lo.float(), yb)).backward(); sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sc.step(opt); sc.update(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(iters):
        with autocast("cuda", dtype=torch.float16, enabled=amp):
            _, lo = fn(xb)
        sc.scale(crit(lo.float(), yb)).backward(); sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sc.step(opt); sc.update(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1e3

report["step_ms"] = {}
print(f"\\n{'batch':>6} {'mode':>16} {'AMP':>4} {'ms/step':>9} {'samples/s':>12}")
MODES = [("eager", None)] if BENCH_SMOKE else [("eager", None),
                                               ("reduce-overhead", "reduce-overhead")]
for bs in ((256,) if BENCH_SMOKE else (512, 256)):
    for mode, maker in MODES:
        for amp in (True, False):
            torch.manual_seed(42)
            mdl = M.build_model(66, MODEL_CFG).to(dev).train()
            fn = mdl.forward_both if maker is None else torch.compile(mdl.forward_both, mode=maker)
            try:
                ms = bench_step(mdl, fn, bs, amp)
                report["step_ms"][f"{bs}|{mode}|amp={amp}"] = ms
                print(f"{bs:>6} {mode:>16} {str(amp):>4} {ms:>9.3f} {bs/ms*1000:>12,.0f}")
            except Exception as e:
                report["step_ms"][f"{bs}|{mode}|amp={amp}"] = f"FAILED {type(e).__name__}: {e}"
                print(f"{bs:>6} {mode:>16} {str(amp):>4}  FAILED  {type(e).__name__}: {str(e)[:70]}")
            del mdl, fn; torch.cuda.empty_cache()

# ---- 3. evaluation: BN-folded sequential vs vmapped, on the REAL test set ----
NC = 2 if BENCH_SMOKE else 8            # 8 stand-in clients is enough to read the curve
torch.manual_seed(0)
models = []
for i in range(NC):
    m = M.build_model(66, MODEL_CFG).to(dev).eval()
    for b in m.modules():
        if isinstance(b, torch.nn.BatchNorm1d):
            b.running_mean.normal_(0, 1); b.running_var.uniform_(0.3, 3.0)
    models.append(m)
folded = [E.fold_bn(m) for m in models]
report["fold_check"] = E.verify_fold(models[0], folded[0], Xtg[:8192].float(), amp=True)
print("\\nBN fold exactness:", report["fold_check"])

P = torch.rand(NC, 16, 256, device=dev)
R = torch.ones(NC, 16, dtype=torch.bool, device=dev)
SUB = min(200_000 if BENCH_SMOKE else 2_000_000, Xtg.shape[0])
Xs, Ys = Xtg[:SUB], Ytg[:SUB]

report["eval"] = {"unfolded": {}, "folded": {}, "vmap": {}}
EBS = (4096,) if BENCH_SMOKE else (8192, 16384, 32768)
for bs in EBS:
    for tag, ms_ in (("unfolded", models), ("folded", folded)):
        torch.cuda.synchronize(); t = time.time()
        cmp_, cmc_, _ = E.eval_sequential(ms_, P, R, Xs, Ys, batch=bs, C=16)
        torch.cuda.synchronize(); dt = time.time() - t
        report["eval"][tag][bs] = {"seconds": dt, "client_rows_per_s": SUB * NC / dt}
        print(f"  {tag:>9} bs {bs:>6}: {dt:6.2f}s  {SUB*NC/dt:>12,.0f} client-rows/s")
        if tag == "folded" and bs == EBS[-1 if BENCH_SMOKE else 1]:
            ref_p, ref_c = cmp_.clone(), cmc_.clone()
            REF_BS = bs

meta_base = E.make_meta_base(lambda: M.build_model(66, MODEL_CFG))
for G in ((2,) if BENCH_SMOKE else (2, 4, 8)):
    bs = max(2048, (16384 if BENCH_SMOKE else 131072) // G)
    try:
        ev = E.VmapEvaluator(folded, G, meta_base, half=True)
        torch.cuda.synchronize(); t = time.time()
        cmp_, cmc_, _ = ev.run(P, R, Xs, Ys, batch=bs, C=16)
        torch.cuda.synchronize(); dt = time.time() - t
        report["eval"]["vmap"][G] = {
            "batch": bs, "seconds": dt, "client_rows_per_s": SUB * NC / dt,
            "speedup_vs_folded_seq": report["eval"]["folded"][REF_BS]["seconds"] / dt,
            "cm_proto_mismatch": int((cmp_ - ref_p).abs().sum()),
            "cm_clf_mismatch": int((cmc_ - ref_c).abs().sum()),
            "mismatch_pct": 100.0 * int((cmp_ - ref_p).abs().sum()) / (SUB * NC),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        print(f"  vmap G={G:>2} bs {bs:>6}: {dt:6.2f}s  {SUB*NC/dt:>12,.0f} client-rows/s  "
              f"{report['eval']['vmap'][G]['speedup_vs_folded_seq']:.2f}x  "
              f"proto mismatch {report['eval']['vmap'][G]['cm_proto_mismatch']}")
        del ev
    except Exception as e:
        report["eval"]["vmap"][G] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  vmap G={G:>2}: FAILED {type(e).__name__}: {str(e)[:90]}")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# ---- 4. project the three scenarios ----------------------------------------
# The driver uses folded sequential eval (eval_group=1). A faster experimental
# vmap result is not a measurement of that path, even if confusion counts match.
rate = report["eval"]["folded"][REF_BS]["client_rows_per_s"]
report["projection_basis"] = {"eval_group": 1, "eval_batch": REF_BS,
                              "includes_setup_and_commit": False,
                              "kind": "microbenchmark estimate; confirm with real rounds"}
def projected_step(bs):
    compiled = report["step_ms"].get(f"{bs}|reduce-overhead|amp=True")
    if CFG["compile"] and isinstance(compiled, float):
        return compiled
    return report["step_ms"].get(f"{bs}|eager|amp=True")
step, step256 = projected_step(512), projected_step(256)
proj = {}
for n, bs, st in ((20, 512, step), (50, 512, step), (100, 256, step256)):
    if not isinstance(st, float): continue
    steps = {20: 84083, 50: 84098, 100: 168200}[n]
    train_s = steps * st / 1000 / 2
    proto_s = 43_045_415 / rate / 2      # rough proxy; confirm with actual prototype pass
    eval_s = n * 10_761_343 / rate / 2
    proj[n] = {"train_s": train_s, "proto_s": proto_s, "eval_s": eval_s,
               "round_s": train_s + proto_s + eval_s,
               "hours_50_rounds": (train_s + proto_s + eval_s) * 50 / 3600}
    print(f"\\n{n:>3} clients: train {train_s:6.0f}s + protos {proto_s:5.0f}s + eval {eval_s:6.0f}s "
          f"= {proj[n]['round_s']:6.0f}s/round -> {proj[n]['hours_50_rounds']:.1f} h for 50 rounds")
report["projection"] = proj
del Xg, Yg, Xtg, Ytg, models, folded; torch.cuda.empty_cache()
Path("/kaggle/working/calibration.json").write_text(json.dumps(report, indent=2, default=str))
print("\\nwrote /kaggle/working/calibration.json")
'''

CALIB_LAUNCH = '''if BENCH_ONLY:
    print("BENCH_ONLY: skipping the 3-round run.")
else:
    import importlib, proj.driver as D
    importlib.reload(D)
    OUT_ROOT = Path("/kaggle/working")
    t0 = time.time()
    res = D.run(cfg, paths, OUT_ROOT)
    print(json.dumps(res, indent=2), f"\\ntotal {time.time()-t0:.0f}s")
    import csv
    with open(Path(OUT_ROOT) / "runs" / CFG["run_name"] / "metrics" / "history.csv") as fh:
        rows = list(csv.DictReader(fh))
    timing_dir = Path(OUT_ROOT) / "runs" / CFG["run_name"] / "logs"
    wall_times = []
    print("\\nreal round times (including checkpoint commit; setup/compile excluded):")
    for r in rows:
        timing = json.loads((timing_dir / f"timing_{int(r['round']):03d}.json").read_text())
        wall_times.append(float(timing["round_including_commit_s"]))
        print(f"  round {r['round']}: {wall_times[-1]:7.1f}s "
              f"(train {float(r['train_seconds']):7.1f}s eval {float(r['eval_seconds']):6.1f}s) "
              f"proto f1_macro {float(r['proto_f1_macro']):.4f}")
    steady = wall_times[1:]
    if steady:
        worst = max(steady)
        print(f"\\nsteady-state round: {worst:.0f}s -> 50 rounds = {worst*50/3600:.2f} h "
              f"(round 1: {wall_times[0]:.0f}s; add measured setup/compile per session)")
'''


def calib_notebook() -> nbf.NotebookNode:
    title = "TinyProto FP calibration"
    nb = base_notebook(title, CALIB_CFG, 20, 3, 512, CALIB_LAUNCH, extra_md="""
---

**This notebook produces measurements, not results.** It fixes three numbers the production
notebooks need and that no other machine can supply: the real step time on sm_75 with and
without CUDA graphs, the real evaluation throughput (which at 100 clients × 50 rounds is the
dominant cost of the whole experiment), and the real prepack time. It then runs 3 real rounds on
the 20-client partition so `max_hours` is derived from a round the driver actually executed,
not from a synthetic loop.
""")
    # insert the benchmark cell before the launch cell
    idx = next(i for i, c in enumerate(nb.cells)
               if c.cell_type == "markdown" and c.source.startswith("## Run"))
    nb.cells.insert(idx, nbf.v4.new_markdown_cell(
        "## Micro-benchmarks\n\nThe production configuration is derived from these numbers. "
        "Ratios measured on other GPUs are guides; only these are measurements of this model on "
        "this accelerator."))
    nb.cells.insert(idx + 1, nbf.v4.new_code_cell(CALIB_BENCH))
    return nb


# ---------------------------------------------------------------------------
# mu sweep notebook
# ---------------------------------------------------------------------------

MU_CFG = '''# --- the grid -------------------------------------------------------------
# mu_kind "inv_mean_nij" makes the grid scenario-independent: the value is a MULTIPLE of
# 1/mean(n_ij), so the same grid means the same thing at 20, 50 and 100 clients.
#   k = 1.0  -> mu.chatG_j has the same magnitude as a local prototype
#   k < 1    -> a weaker pull toward the global prototype
# These four ARE the frozen mu-selection protocol (proj/sweep.py::MU_PROTOCOL). They are
# written out here so the knob cell stays readable, and checked against the frozen copy in the
# config cell below -- editing one without the other fails loudly there rather than producing a
# mu that production would later reject.
MU_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
SWEEP_ROUNDS = 4          # round 1 has no regularizer, so >= 3 is needed to see mu at all
VAL_FRAC = 0.02           # held out from EACH client, stratified by class
VAL_SEED = 42
SWEEP_RESUME = False      # True ONLY when continuing an interrupted sweep with the previous
                          # session's output attached. It asserts that state was actually
                          # found: a forgotten attachment otherwise looks like a fresh sweep
                          # and silently retrains the whole grid.

CFG = {
    "run_name": "tinyproto_fp_musweep",
    "scenario": "20client", "n_clients": 20,
    "rounds": SWEEP_ROUNDS, "local_epochs": 1, "lam": 1.0, "cps_s": 50, "mask_seed": 42,
    "mu_kind": "inv_mean_nij", "mu_value": 1.0,      # overwritten per grid point
    "batch": 512, "eval_batch": 8192,
    "lr": 1e-3, "weight_decay": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
    "clip": 1.0, "amp": True, "compile": True, "eval_group": 1,
    "seed": 42, "num_classes": 16, "n_features": 66, "feature_dim": 256,
    "max_hours": 11.0, "rounds_this_session": None, "require_resume": False,
    # A sweep round scores on the 2% validation slice, not the 10.76 M-row test set, so it is
    # far cheaper than a production round. Still generous: this only gates the first round.
    "expected_round_s": 600.0,
    "import_roots": ["/kaggle/input"], "save_preds_rounds": [],
}
for k, v in CFG.items(): print(f"  {k:22s} {v}")
print("\\ngrid:", MU_GRID)
'''

MU_LAUNCH = '''"""Grid search for μ, selected on a TRAIN-side validation split.

The project's rule forbids tuning a hyperparameter on the test set. Each client keeps 98% of its
rows for training and 2% as a local validation slice, stratified by class with a fixed seed. The
union of those slices becomes a shared validation set: every client is scored on the SAME rows,
exactly as with the global test set, so per-client metrics stay comparable. The winner is the μ
with the highest mean-over-clients `f1_macro` under the paper's own rule (Eq. 12). The three
production runs then train on **100%** of train and never see this split again.

The paper says only that μ came from "systematic parameter selection ... detailed in the
appendix". The PDF has no appendix and never uses the phrase "grid search". This procedure is
this build's choice, made at the user's direction, and must be reported as such.
"""
import importlib, pyarrow.dataset as pds
import proj.driver as D, proj.data as DD
importlib.reload(D)
from proj.data import stratified_holdout
from proj.sweep import (MU_PROTOCOL, candidate_name, choose_winner, plan_sweep,
                        read_claims, selection_signature)

# ---- resume gate FIRST -----------------------------------------------------------------
# This used to run after the holdout pass below, which streams the entire training set. A wrong
# or missing attachment then cost that whole pass before anything complained. Nothing here
# touches the data, so it is free to do first.
SEARCH_ROOTS = [Path("/kaggle/working/runs"), *[Path(p) for p in CFG["import_roots"]]]
CLAIMS = read_claims(CFG["scenario"], SEARCH_ROOTS)
PLAN = plan_sweep(MU_GRID, CFG["scenario"], SEARCH_ROOTS, claims=CLAIMS)   # raises if partial
STARTED = {k for k, v in PLAN.items() if v["committed_rounds"] > 0}
print(f"{'candidate':>10} {'run_name':>34} {'committed':>10} {'claimed':>8}  action")
for k in MU_GRID:
    v = PLAN[str(k)]
    act = ("resume" if 0 < v["committed_rounds"] < SWEEP_ROUNDS
           else "reuse (complete)" if v["committed_rounds"] >= SWEEP_ROUNDS else "start fresh")
    print(f"{k:>10} {v['run_name']:>34} {v['committed_rounds']:>10} {v['claimed_rounds']:>8}  {act}")
if SWEEP_RESUME:
    assert STARTED, ("SWEEP_RESUME is set but no candidate has a committed round. Attach the "
                     "previous sweep's output as a kernel source; do not clear this flag.")

TR_PATH = Path("/kaggle/working/train_index.npz")
VX, VY = Path("/kaggle/temp/val_X.npy"), Path("/kaggle/temp/val_y.npy")
Path("/kaggle/temp").mkdir(exist_ok=True)

if True:  # Rebuild deterministically; never reuse a partially written/stale holdout cache.
    # One streaming pass per client: split by class, keep the 2% as the shared validation set.
    tr_idx, vx, vy = {}, [], []
    scal = json.loads(Path(SCALER_PATH).read_text())
    for cid in range(CFG["n_clients"]):
        fs = sorted((CLIENT_ROOT / "train" / f"client_id={cid:03d}").glob("*.parquet"))
        ds = pds.dataset(fs, format="parquet")
        Xc, yc = [], []
        for b in ds.to_batches(columns=FEATURE_COLS + ["label"], batch_size=65_536):
            Xc.append(np.column_stack([b.column(c).to_numpy(zero_copy_only=False)
                                       for c in FEATURE_COLS]).astype(np.float16))
            yc.append(np.asarray(b.column("label").to_numpy(zero_copy_only=False), np.int8))
        Xc = np.concatenate(Xc); yc = np.concatenate(yc)
        t, v = stratified_holdout(yc, {cid: (0, len(yc))}, VAL_FRAC, seed=VAL_SEED)
        tr_idx[str(cid)] = t.astype(np.int64)
        vx.append(Xc[v]); vy.append(yc[v])
        print(f"  client {cid:03d}: train {len(t):>9,}  val {len(v):>7,}", flush=True)
        del Xc, yc
    np.savez(TR_PATH, **tr_idx)
    # train/ is ALREADY standardized, so the validation slice needs no scaler -- unlike test/.
    np.save(VX, np.concatenate(vx)); np.save(VY, np.concatenate(vy))
    del tr_idx, vx, vy

vy_all = np.load(VY)
print(f"\\nshared validation set: {len(vy_all):,} rows, "
      f"{(len(vy_all)*66*2)/2**20:.0f} MiB fp16, class counts {np.bincount(vy_all.astype(int), minlength=16).tolist()}")
paths_sweep = dict(paths, train_index=str(TR_PATH), eval_X=str(VX), eval_y=str(VY))
from proj.ckpt import file_digest
from proj.metrics import atomic_write_json
cfg["validation_fingerprint"] = {"train_index": file_digest(TR_PATH), "X": file_digest(VX),
                                 "y": file_digest(VY), "fraction": VAL_FRAC, "seed": VAL_SEED}
summary_path = Path("/kaggle/working/mu_sweep.json")


results = {}
for k in MU_GRID:
    name = candidate_name(cfg["scenario"], k)
    c = dict(cfg); c.update({"run_name": name, "mu_value": float(k),
                             "rounds": SWEEP_ROUNDS,
                             "require_resume": str(k) in STARTED})
    print(f"\\n{'='*70}\\nmu multiplier k = {k}\\n{'='*70}", flush=True)
    wb = None
    if WANDB is not None:
        try:
            wb = WANDB.init(project=WANDB_PROJECT, name=name, group=f"musweep-{cfg['scenario']}",
                            job_type="mu-sweep", reinit=True,
                            config={"k": float(k), "scenario": cfg["scenario"],
                                    "protocol_version": MU_PROTOCOL["version"],
                                    "rounds": SWEEP_ROUNDS, "batch": cfg["batch"]})
        except Exception as e:                      # telemetry must never stop the sweep
            print("W&B init failed:", type(e).__name__, str(e)[:120]); wb = None
    try:
        result = D.run(c, paths_sweep, Path("/kaggle/working"), wandb_run=wb)
        if result["last_round"] != SWEEP_ROUNDS:
            results[str(k)] = {"rounds": result["last_round"], "status": "incomplete"}
            break
        import csv
        with open(Path("/kaggle/working/runs") / name / "metrics" / "history.csv") as fh:
            rows = list(csv.DictReader(fh))
        mu_j = json.loads((Path("/kaggle/working/runs") / name / "mu.json").read_text())
        results[str(k)] = {
            "best_proto_f1_macro": max(float(r["proto_f1_macro"]) for r in rows),
            "last_proto_f1_macro": float(rows[-1]["proto_f1_macro"]),
            "last_clf_f1_macro": float(rows[-1]["clf_f1_macro"]),
            "rounds": len(rows), "mu_absolute": mu_j["resolved"][0],
            "mean_nij": mu_j["mean_nij_all"],
            "history": [{kk: r[kk] for kk in ("round", "proto_f1_macro", "clf_f1_macro",
                                              "reg_loss", "train_loss")} for r in rows]}
        print(f"  -> last {results[str(k)]['last_proto_f1_macro']:.4f}  "
              f"best {results[str(k)]['best_proto_f1_macro']:.4f}  "
              f"mu = {results[str(k)]['mu_absolute']:.4e}")
    except Exception as e:
        import traceback; traceback.print_exc()
        results[str(k)] = {"error": f"{type(e).__name__}: {e}"}
    finally:
        if wb is not None:
            try:
                wb.finish()
            except Exception:
                pass
    # Durable per-candidate progress, rewritten after every candidate. A later session reads
    # this to detect a PARTIAL attachment: a candidate recorded here but whose run directory is
    # absent means finished work went missing, which must stop the sweep rather than silently
    # restart that candidate and mix two sweeps into one grid.
    atomic_write_json(Path("/kaggle/working/mu_sweep_progress.json"), {
        "scenario": cfg["scenario"], "protocol": MU_PROTOCOL,
        "selection_signature": selection_signature(cfg),
        "candidates": {str(kk): {"run_name": candidate_name(cfg["scenario"], kk),
                                 "committed_rounds": int(
                                     len(list((Path("/kaggle/working/runs") /
                                               candidate_name(cfg["scenario"], kk) /
                                               "complete").glob("round_*.done"))))}
                       for kk in MU_GRID}})

winner = choose_winner(results, MU_GRID, SWEEP_ROUNDS)
summary = {"scenario": cfg["scenario"], "grid": MU_GRID, "rounds": SWEEP_ROUNDS,
           # The protocol travels WITH the result: production checks the number came from this
           # exact procedure, not merely from some sweep that reported success.
           "protocol": MU_PROTOCOL,
           "selection_signature": selection_signature(cfg), "results": results,
           "validation_fingerprint": cfg["validation_fingerprint"],
           "candidates": {str(kk): {"run_name": candidate_name(cfg["scenario"], kk),
                                    "committed_rounds": int(
                                        len(list((Path("/kaggle/working/runs") /
                                                  candidate_name(cfg["scenario"], kk) /
                                                  "complete").glob("round_*.done"))))}
                          for kk in MU_GRID},
           "status": "complete" if winner else "incomplete"}
if winner:
    summary["winner"] = winner
    print("Winner for this scenario only:", winner)
else:
    print("Incomplete grid: no winner. Attach these outputs and resume the sweep.")
atomic_write_json(summary_path, summary)
print("wrote", summary_path)

'''


MU_RESULTS = '''import pandas as pd
r = json.loads(Path("/kaggle/working/mu_sweep.json").read_text())
rows = [{"k": k, **{kk: v[kk] for kk in ("best_proto_f1_macro", "last_proto_f1_macro", "rounds")}}
        for k, v in r["results"].items() if isinstance(v, dict) and "best_proto_f1_macro" in v]
if rows:
    df = pd.DataFrame(rows)
    df["k"] = df["k"].astype(float)
    df = df.sort_values("k")
    display(df.round(6))
    ax = df.plot(x="k", y=["last_proto_f1_macro", "best_proto_f1_macro"], marker="o",
                 logx=True, figsize=(8, 4))
    ax.set_xlabel("mu multiplier k  (mu = k / mean(n_ij))")
    ax.set_ylabel("mean-over-clients f1_macro (validation)")
    ax.grid(alpha=.3)
if "winner" in r:
    print(f"\\nwinner k={r['winner']['k']}  mu_absolute={r['winner']['mu_absolute']:.6e}")
    print("Do NOT copy that number into the production notebook. Attach THIS kernel's output")
    print(f"as a kernel source of tinyproto-fp-train-{r['scenario']}; the config cell reads the")
    print("winner from mu_sweep.json and verifies the selection signature before training.")
else:
    print("\\nGrid incomplete: no winner. Attach this output and re-push with SWEEP_RESUME=True.")
'''


def mu_notebook(n: int = 20) -> nbf.NotebookNode:
    title = "TinyProto FP mu sweep" + (f" {n}client" if n != 20 else "")
    batch = 256 if n == 100 else 512
    config = MU_CFG.replace('"scenario": "20client", "n_clients": 20', f'"scenario": "{n}client", "n_clients": {n}')
    config = config.replace('"batch": 512', f'"batch": {batch}')
    nb = base_notebook(title, config, n, 4, batch, MU_LAUNCH, extra_md="""
---

**This notebook selects one hyperparameter; it produces no reportable metric.** μ is scored on a
2% per-client validation slice carved out of TRAIN, never on the test set. A separate sweep is required for each scenario. Attach its completed output to the matching
production notebook, which trains on 100% of train.
""")
    nb.cells[-2] = nbf.v4.new_code_cell(MU_RESULTS)
    return nb


# ---------------------------------------------------------------------------
# metadata + write
# ---------------------------------------------------------------------------

RUNTIME = json.loads((ROOT / "knowledge" / "runtime.json").read_text())


def wandb_prelude(embed: bool) -> str:
    """The inline-key line prepended to WANDB_CELL.

    Embedding is standing authorized policy for this owner's own PRIVATE notebooks, because a
    Kaggle API push cannot attach an editor-selected secret. The cost is real and is restated
    at every use: the key becomes PERMANENT in Kaggle private version history, sharing the
    notebook leaks it, and rotating means rebuilding and re-pushing every notebook carrying it.
    Removal order lives in references/wandb.md section 1.1 -- revoke at W&B FIRST.
    """
    if not embed:
        return '_INLINE_WANDB_KEY = ""    # not embedded; attach a Kaggle secret wandb_key\n'
    sys.path.insert(0, str(ROOT / "scripts"))
    from embed_wandb_key import wandb_key
    k = wandb_key()
    if not k:
        return '_INLINE_WANDB_KEY = ""    # no key in ~/.netrc at build time\n'
    return "_INLINE_WANDB_KEY = " + repr(k) + "\n"


def metadata(title: str, owner: str, sources: list[str], internet: bool = True,
             kernel_sources: list[str] | None = None) -> dict:
    slug = slugify(title)
    return {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": internet,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": sources,
        "competition_sources": [],
        "kernel_sources": kernel_sources or [],
        # Pinned to the image measured to deliver 2 x T4 (knowledge/runtime.json carries the
        # provenance). Kaggle's rolling "latest" has changed torch under this project before.
        "docker_image": RUNTIME["docker_image"],
    }


def write(nb, title: str, out_root: Path, owner: str, sources: list[str],
          kernel_sources: list[str] | None = None) -> Path:
    slug = slugify(title)
    d = out_root / slug
    d.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, d / f"{slug}.ipynb")
    (d / "kernel-metadata.json").write_text(
        json.dumps(metadata(title, owner, sources, kernel_sources=kernel_sources), indent=2))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "papers" / "tinyproto-lee-2026" / "notebook"))
    ap.add_argument("--owner", default="odixe0502")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--mu-kind", default="inv_mean_nij")
    ap.add_argument("--mu-value", default="1.0")
    ap.add_argument("--only", default="")
    ap.add_argument("--mu-dataset", default="",
                    help="OWNER/SLUG of a dataset carrying this scenario's mu_sweep.json, for "
                         "when the running account cannot read the sweep owner's private kernel "
                         "output. May contain {n}, expanded to 20/50/100. The file must be named "
                         "exactly mu_sweep.json: the config cell globs that exact name, which is "
                         "what keeps mu_sweep_progress.json (same scenario key) from matching too.")
    ap.add_argument("--session", type=int, default=0,
                    help="Session number of a multi-session run. Gives this push its OWN "
                         "notebook slug (…-sN) so session N can attach session N-1 without a "
                         "notebook depending on itself. run_name is unchanged, so the "
                         "fingerprint still matches and import_previous still resumes.")
    ap.add_argument("--resume-from", default="",
                    help="OWNER/SLUG of the PREVIOUS session's train notebook. Adds it as a "
                         "kernel source and sets require_resume, so the push refuses to start "
                         "from round 1 if the import did not land. SAME-ACCOUNT ONLY -- see "
                         "--resume-dataset.")
    ap.add_argument("--resume-dataset", default="",
                    help="OWNER/SLUG of a DATASET carrying the previous session's run tree, "
                         "used instead of --resume-from when the previous session ran under a "
                         "DIFFERENT account. Measured 2026-09-09: a private notebook output is "
                         "not readable across accounts -- Kaggle drops the kernel source at push "
                         "time with a one-line warning, reports success, and mounts nothing, so "
                         "the session starts with an empty /kaggle/input. Sets require_resume the "
                         "same way; find_import_source globs config.json under /kaggle/input and "
                         "does not care which mount it came from.")
    ap.add_argument("--no-embed-wandb-key", action="store_true",
                    help="build notebooks with no W&B key at all (see wandb.md 1.1)")
    a = ap.parse_args()
    for flag, val in (("--mu-dataset", a.mu_dataset), ("--resume-from", a.resume_from),
                      ("--resume-dataset", a.resume_dataset)):
        if val and val.count("/") != 1:
            ap.error(f"{flag} must be OWNER/SLUG, got {val!r}")
    # Two resume routes would give find_import_source two candidates with the same fingerprint,
    # which it refuses outright rather than guessing between.
    if a.resume_from and a.resume_dataset:
        ap.error("--resume-from and --resume-dataset are two routes to the same thing; pass one")
    # One resume source belongs to exactly one scenario. Without this guard the same previous
    # output is attached to all three, and find_import_source either imports the wrong run or
    # refuses on a fingerprint mismatch after the session has already been burnt.
    if (a.resume_from or a.resume_dataset) and a.only not in ("train20", "train50", "train100"):
        ap.error("a resume source applies to ONE scenario; pass --only train20|train50|train100")
    out = Path(a.out)
    made = []
    global WANDB_PRELUDE
    embed = not a.no_embed_wandb_key
    # Part of the authorization, not an optional extra: only PRIVATE notebooks may carry
    # the key. metadata() sets is_private True for every notebook it builds; assert it
    # here so a future change to that default cannot quietly publish a key.
    if embed:
        assert metadata("probe", a.owner, ["x"])["is_private"] is True, \
            "refusing to embed a W&B key: notebooks are not private"
    WANDB_PRELUDE = wandb_prelude(embed)
    if embed and "_INLINE_WANDB_KEY = \"\"" not in WANDB_PRELUDE:
        print("W&B key embedded. It is now PERMANENT in Kaggle private version history\n"
              "  for every version pushed from these files. Sharing a notebook leaks it;\n"
              "  rotating means rebuilding and re-pushing all of them. To remove it,\n"
              "  follow references/wandb.md 1.1 -- revoke at W&B FIRST.\n"
              "  Do not commit or share the generated .ipynb files.\n")

    if not a.only or a.only == "calib":
        made.append(write(calib_notebook(), "TinyProto FP calibration", out, a.owner,
                          [DATASETS[20], CENTRALIZED]))
    for n in (20, 50, 100):
        if not a.only or a.only in ("mu", f"mu{n}"):
            title = "TinyProto FP mu sweep" + (f" {n}client" if n != 20 else "")
            made.append(write(mu_notebook(n), title, out, a.owner, [DATASETS[n], CENTRALIZED]))
    for n, batch in ((20, 512), (50, 512), (100, 256)):
        if a.only and a.only != f"train{n}":
            continue
        nb = train_notebook(n, a.rounds, batch, a.max_hours, a.mu_kind, a.mu_value,
                            require_resume=bool(a.resume_from or a.resume_dataset))
        sources = [DATASETS[n], CENTRALIZED]
        # The config cell reads mu under a signature check and refuses to start without it.
        # It can come from the sweep's own kernel output (same account) or from a shared
        # dataset (cross-account, where a private kernel output is not readable).
        if a.mu_dataset:
            sources.append(a.mu_dataset.replace("{n}", str(n)))
            kernel_sources = []
        else:
            sweep_slug = slugify("TinyProto FP mu sweep" + (f" {n}client" if n != 20 else ""))
            kernel_sources = [f"{a.owner}/{sweep_slug}"]
        if a.resume_from:
            kernel_sources.append(a.resume_from)
        if a.resume_dataset:
            sources.append(a.resume_dataset)
        suffix = f" s{a.session}" if a.session else ""
        made.append(write(nb, f"TinyProto FP train {n}client{suffix}", out, a.owner,
                          sources, kernel_sources=kernel_sources))

    for d in made:
        meta = json.loads((d / "kernel-metadata.json").read_text())
        print(f"  {d}  ->  {meta['id']}  ({meta['code_file']})")
    print(f"\n{len(made)} notebooks written to {out}")


if __name__ == "__main__":
    main()

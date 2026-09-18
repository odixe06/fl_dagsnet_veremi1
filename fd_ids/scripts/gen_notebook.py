#!/usr/bin/env python
"""Generate one FD-IDS training notebook per client scenario.

The notebook body is assembled from papers/fd-ids-2025/proj/*.py, which are the same files
the local tests exercise. Editing a module and regenerating is the only supported path;
never hand-edit the .ipynb.
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "papers/fd-ids-2025/proj"
RUNTIME = json.loads((ROOT / "knowledge/runtime.json").read_text())
META = json.loads((ROOT / "knowledge/meta.json").read_text())
MODULES = ("model", "ckpt", "metrics", "data", "fdids", "driver", "verify")

# batch from knowledge/DATASET.md 4; every other hyperparameter from the paper's Table 3.
SCENARIOS = {20: dict(batch=512, dataset="veremi-fl-20client"),
             50: dict(batch=512, dataset="veremi-fl-50client"),
             100: dict(batch=256, dataset="veremi-fl-100client")}
PAPER = dict(lr=1e-3, lam=0.5, beta=0.1, mu=0.01, temperature=3.0, rounds=50)


def kaggle_slug(title):
    """Kaggle's own rule: lowercase, every run of non-alphanumerics becomes one hyphen."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "source": s,
                     "execution_count": None, "outputs": []}


def build(K, owner, wandb_project, require_resume=False, kernel_sources=(),
          probe=False, max_hours=11.0, run_tag="", dataset_sources=()):
    sc = SCENARIOS[K]
    # A probe measures the T4 and is thrown away: its own run_name keeps its checkpoints,
    # its W&B run and its fingerprint separate from the real run of the same scenario.
    # run_tag does the same for a relaunch. W&B is opened with resume="allow" on a fixed id,
    # so re-running a name whose history already reaches step 3 makes every log(step=1..3)
    # non-monotonic and W&B DROPS it -- the run looks alive and reports nothing. run_name is
    # also in ckpt.FINGERPRINT_KEYS, so a tagged run additionally cannot resume from, or be
    # spliced onto, the checkpoints of the untagged one. Both are what a relaunch wants.
    run_name = f"fdids_{K}c_probe" if probe else f"fdids_{K}c{run_tag}"
    rounds = 2 if probe else PAPER["rounds"]
    # Kaggle derives the kernel slug from the TITLE and ignores the `id` field when the two
    # disagree: pushing id "fdids-veremi-100c" with title "FD-IDS VeReMi 100 clients"
    # created khanhmay0304/fd-ids-veremi-100-clients and left the declared id unusable for
    # `kernels status`, for pulling output and for kernel_sources on a continuation.
    title = f"FD-IDS VeReMi {K} clients" + (" probe" if probe else "")
    slug = kaggle_slug(title)
    cells = [md(f"""# FD-IDS on VeReMi NextGen — {K} clients

FedProx + round-wise knowledge distillation (Zhang et al., *Sensors* 2025, 25, 4309),
Algorithm 1 and Eq. (2)-(6), with **DAGSNet** (395,024 params) as the classifier instead of
the paper's 5-layer DNN, on VeReMi NextGen (16 classes, 66 features) instead of
Edge-IIoT/N-BaIoT.

| from the paper (Table 3) | from `knowledge/` |
|---|---|
| Adam, lr 1e-3, mu 0.01, lambda 0.5, beta 0.1, T 3 | DAGSNet architecture, 66-feature order, class order |
| CrossEntropy hard loss, ReLU, round-wise KD | batch {sc['batch']}, {rounds} rounds x 1 local epoch, seed 42 |

**Deviations, all deliberate:** classifier, dataset, client count ({K} not 9), Dirichlet
alpha = 0.5 fixed by the partition (paper sweeps theta = 1 and 0.1), 50 rounds x 1 epoch
(paper: 40 x 2). Per-round output is **weights only**; `proj/ckpt.py::load_weights` rebuilds
the model at any round, and the last cell re-derives every published metric from the
confusion matrices on disk."""),

    code("""import os, subprocess, sys, time, torch
T0 = time.monotonic()     # session clock: the 12 h cap charges for spawn and compile too.
                          # monotonic, not time(): a clock step must not move the deadline.
n = torch.cuda.device_count()
assert n == 2, f"expected 2 GPUs, got {n}. machine_shape must be NvidiaTeslaT4."
for i in range(n):
    cap = torch.cuda.get_device_capability(i)
    assert cap == (7, 5), f"GPU {i} is {cap}, expected (7,5) Tesla T4"
    print(i, torch.cuda.get_device_name(i), cap,
          f"{torch.cuda.get_device_properties(i).total_memory/2**30:.1f} GiB")
print("torch", torch.__version__, "| python", sys.version.split()[0])
# NCCL is unused (FL clients never form a process group) but P2P probing can still hang.
os.environ["NCCL_P2P_DISABLE"] = "1"; os.environ["NCCL_IB_DISABLE"] = "1"
os.makedirs("/kaggle/working/proj", exist_ok=True)
open("/kaggle/working/proj/__init__.py", "w").close()
sys.path.insert(0, "/kaggle/working")"""),

    code(f'''CFG = dict(
    # --- architecture: knowledge/ARCHITECTURE.md, frozen (395,024 params)
    patch_len=6, stem_ch=96, dense_growth=32, dense_layers=3,
    incep_modules=2, fire_modules=3, dropout=0.1,
    num_classes=16, n_features=66,
    # --- FD-IDS: paper Table 3
    lr={PAPER["lr"]}, lam={PAPER["lam"]}, beta={PAPER["beta"]},
    mu={PAPER["mu"]}, temperature={PAPER["temperature"]},
    rounds={rounds}, local_epochs=1, clip=1.0, seed=42,
    # --- compute: knowledge/DATASET.md 4
    n_clients={K}, batch={sc["batch"]}, eval_batch=16384,
    device="cuda", world_size=2, compile=True,
    # Each client starts a fresh GradScaler at 2**16 and spends a few steps calibrating.
    # That is normal; skipping far more than that is a training problem. Fixed before the
    # first measurement so it cannot be widened afterwards to make a run pass.
    max_skips_per_client=16,
    finalize_reserve_seconds=600,   # never start a round that leaves no time to commit it
    run_name="{run_name}",
    cache="/kaggle/temp/veremi_cache",
    max_seconds={max_hours} * 3600,   # 12 h hard cap; leave room to finalize artifacts
    require_resume={require_resume},
)
# data_id is filled in below, once the feature order and scaler are known. It is part of
# proj/ckpt.py FINGERPRINT_KEYS, so a run cannot resume across a changed preprocessing.
for k, v in CFG.items(): print(f"{{k:>16}} = {{v}}")''')]

    for mod in MODULES:
        body = (PROJ / f"{mod}.py").read_text().rstrip()
        cells.append(code(f"%%writefile /kaggle/working/proj/{mod}.py\n{body}"))

    cells += [code(f'''import wandb
from kaggle_secrets import UserSecretsClient

try:
    _wandb_key = UserSecretsClient().get_secret("wandb_key")
except Exception as e:
    raise SystemExit(f"W&B secret unavailable: {{e}}")
wandb.login(key=_wandb_key, relogin=True, verify=True)
del _wandb_key
# W&B fails fast before dataset decode or model preparation. The key is never persisted.
run = wandb.init(project="{wandb_project}", name="{run_name}", config=CFG,
                 resume="allow", id="{run_name}")
print("W&B:", run.url)''')]

    cells += [
    code(f'''# Cheap identity work, then the resume gate, then the decode. A continuation push must
# die at the gate, not after a two-minute parquet pass.
import hashlib, json, time, numpy as np
from pathlib import Path
from proj import ckpt as C
from proj.data import (find_root, load_clients, load_test, assert_fp16_safe,
                       cache_ok)

FL_ROOT = find_root("train/client_id=000")
CEN_ROOT = find_root("upload/test")
TEST_ROOT = CEN_ROOT / "upload/test"
SCALER = json.loads((CEN_ROOT / "upload/scaler.json").read_text())["features"]
assert len(SCALER) == 66, f"scaler has {{len(SCALER)}} entries, expected 66"
FEATS = {META["feature_cols"]!r}
CLASS_NAMES = {META["class_names"]!r}

# What the fingerprint could not otherwise see. A permuted feature order, a re-fitted
# scaler or a different partition all keep every shape identical, so without this a run
# resumes cleanly on top of incompatible weights and reports it as one experiment.
CFG["data_id"] = hashlib.sha256(json.dumps({{
    "features": FEATS, "classes": CLASS_NAMES, "n_clients": CFG["n_clients"],
    "scaler": [[SCALER[c]["mean"], SCALER[c]["std_used"]] for c in FEATS],
}}, sort_keys=True).encode()).hexdigest()[:16]
print("FL root    :", FL_ROOT)
print("test root  :", TEST_ROOT)
print("data_id    :", CFG["data_id"])
print("fingerprint:", C.fingerprint(CFG))

last = C.resolve_resume(CFG["run_name"], CFG)
if CFG["require_resume"] and last is None:
    raise SystemExit("require_resume set but no VERIFIED checkpoint found — fix the "
                     "attachment. A marker without its artifacts does not count.")
print("resume from round", last)

cache = Path(CFG["cache"]); cache.mkdir(parents=True, exist_ok=True)
MF = cache / "manifest.json"
FILES = ("train_X.f16.npy", "train_y.u8.npy", "test_X.f16.npy", "test_y.u8.npy",
         "spans.json")
want = {{"data_id": CFG["data_id"], "n_clients": CFG["n_clients"],
        "fl_root": str(FL_ROOT), "test_root": str(TEST_ROOT)}}


t0 = time.time()
if cache_ok(cache, want, CFG["n_clients"]):
    print("prepack cache reusable (manifest matches and every file checks out)")
else:
    # The presence of train_X is not evidence the cache is complete or current: a crash
    # after the first file made every later session skip the decode entirely.
    for f in FILES: (cache / f).unlink(missing_ok=True)
    MF.unlink(missing_ok=True)
    X, Y, spans = load_clients(FL_ROOT, FEATS, CFG["n_clients"])
    print(f"train {{X.shape}} max|x|={{assert_fp16_safe(X,'train'):.1f}}")
    np.save(cache / "train_X.f16.npy", X); np.save(cache / "train_y.u8.npy", Y)
    json.dump({{str(k): v for k, v in spans.items()}}, open(cache / "spans.json", "w"))
    del X, Y
    TX, TY = load_test(TEST_ROOT, FEATS, SCALER)
    print(f"test  {{TX.shape}} max|x|={{assert_fp16_safe(TX,'test'):.1f}}")
    np.save(cache / "test_X.f16.npy", TX); np.save(cache / "test_y.u8.npy", TY)
    del TX, TY
    MF.write_text(json.dumps(want))                       # cache marker: absolutely last
    assert cache_ok(cache, want, CFG["n_clients"]), "the cache just written does not validate"

spans = {{int(k): tuple(v) for k, v in json.load(open(cache / "spans.json")).items()}}
CFG["n_test"] = len(np.load(cache / "test_y.u8.npy", mmap_mode="r"))
n_train = sum(h - l for l, h in spans.values())
assert n_train == 43_045_415, f"train rows {{n_train}} != 43,045,415"
assert CFG["n_test"] == 10_761_343, f"test rows {{CFG['n_test']}}"
assert len(spans) == CFG["n_clients"], f"{{len(spans)}} spans for {{CFG['n_clients']}} clients"

# content_id reads the labels that are actually cached, hit or miss. data_id can only see
# the feature order and the scaler, so a re-partitioned or rewritten dataset keeps it
# identical; the row counts and the class histogram do not. Reads 54 MB of uint8, ~1 s.
_ytr = np.load(cache / "train_y.u8.npy", mmap_mode="r")
_yte = np.load(cache / "test_y.u8.npy", mmap_mode="r")
assert len(_ytr) == n_train, f"train X/y disagree: {{len(_ytr)}} labels for {{n_train}} rows"
CFG["content_id"] = hashlib.sha256(json.dumps({{
    "clients": [[c, spans[c][0], spans[c][1]] for c in sorted(spans)],
    "train_hist": np.bincount(np.asarray(_ytr), minlength=16).tolist(),
    "test_hist": np.bincount(np.asarray(_yte), minlength=16).tolist(),
}}, sort_keys=True).encode()).hexdigest()[:16]
del _ytr, _yte
print("content_id :", CFG["content_id"])
print(f"prepack {{time.time()-t0:.1f}}s | {{n_train:,}} train / {{CFG['n_test']:,}} test rows")'''),

    code('''from proj.driver import run as train, write_manifest
# Raises if this run's checkpoints were trained on different data. The resume gate above
# ran before the decode and could only compare data_id; content_id is the post-decode one.
write_manifest(CFG, CLASS_NAMES, spans,
               y_true_src=Path(CFG["cache"]) / "test_y.u8.npy",
               extra={"fl_root": str(FL_ROOT), "test_root": str(TEST_ROOT),
                      "feature_cols": FEATS, "n_test": CFG["n_test"],
                      "data_id": CFG["data_id"], "content_id": CFG["content_id"],
                      "scaler": {c: [SCALER[c]["mean"], SCALER[c]["std_used"]]
                                 for c in FEATS}})
hist = train(CFG, spans, CLASS_NAMES, wandb_run=run, t_origin=T0)
print(f"\\ncompleted {len(hist)} rounds this session")'''),

    code('''# Every published number, re-derived from the artifacts on disk. Never from memory.
import csv
from proj.verify import verify_run
from proj.model import build_model
from proj.metrics import METRIC_KEYS

d = C.run_dir(CFG["run_name"])
# y_true from the RUN, not from /kaggle/temp: the cache is gone with the session, and the
# check has to be the same one someone can repeat after downloading the output alone.
ok, lines = verify_run(d, cfg=CFG, build=build_model, expect_params=395_024,
                       y_true_path=d / "reports" / "y_true.u8.npy", full=True)
print("\\n".join(lines))

last = C.last_complete_round(d, C.fingerprint(CFG)) or 0
print(f"\\nrounds verified : {last} / {CFG['rounds']}")
if last < CFG["rounds"]:
    print(f"  INCOMPLETE — attach this notebook's output to the next push and regenerate "
          f"with --require-resume to continue at round {last + 1}")
rows = [r for r in csv.DictReader(open(d / "history.csv")) if int(r["round"]) <= last]
if rows:
    fin = rows[-1]
    # The headline is the LAST round, fixed before the run. best-f1 is chosen on the test
    # set after seeing it, so it is a description of the curve and not a second result.
    print(f"\\nresult at round {fin['round']}:")
    for k in METRIC_KEYS: print(f"  {k:<20} {float(fin[k]):.6f}")
    b = max(rows, key=lambda r: float(r["f1_macro"]))
    print(f"\\n[descriptive only] best f1_macro {float(b['f1_macro']):.6f} "
          f"at round {b['round']} — picked on test, not a reported result")

# Calibration, from THIS session's rounds only. Using the CSV would mix in every imported
# round: 50 historical rounds and one new one gives a negative overhead and a projection of
# about a minute. `hist` is what this session actually ran.
sec = [float(r["seconds"]) for r in hist]
overhead = (time.monotonic() - T0) - sum(sec)
print(f"\\nbackend  : {CFG.get('backend', '?')}")
print(f"session  : {len(hist)} round(s) here | startup+prepack+compile {overhead/60:.1f} min"
      f" | verify and W&B are outside this figure")
if len(sec) < 2:
    print("timing   : need 2 completed rounds to separate startup from steady state; "
          f"got {len(sec)}. No projection.")
else:
    # Round 1 pays for cudagraph capture and a cold page cache; drop it.
    steady = sum(sec[1:]) / len(sec[1:])
    vt = max(float(r.get("vram_train_gb", 0) or 0) for r in hist)
    ve = max(float(r.get("vram_eval_gb", 0) or 0) for r in hist)
    tkd = sum(float(r.get("teacher_sec", 0) or 0) for r in hist) / len(hist)
    print(f"timing   : rounds {[round(x) for x in sec[-3:]]}s | steady {steady:.0f}s/round"
          f" | teacher {tkd:.0f}s/round (summed over clients, 2 GPUs in parallel)")
    print(f"VRAM     : train {vt:.2f} GiB/GPU | eval {ve:.2f} GiB/GPU (of 16)")
    print(f"projected: {CFG['rounds']} rounds = "
          f"{(steady*CFG['rounds'] + overhead)/3600:.2f} h "
          f"({(steady*CFG['rounds'])/3600:.2f} h of rounds + {overhead/3600:.2f} h startup)")
if run is not None: run.finish()
assert ok, "artifact verification FAILED — see the FAIL lines above"''')]

    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                   "isInternetEnabled": True, "language": "python",
                   "sourceType": "notebook"}}}
    for c in nb["cells"]:
        c["source"] = [l + "\n" for l in c["source"].split("\n")]
    meta = {"id": f"{owner}/{slug}", "title": title,
            "code_file": f"{run_name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True,
            "machine_shape": RUNTIME["machine_shape"],
            "docker_image": RUNTIME["docker_image"],
            # The two data mounts first, then any checkpoint dataset. A previous session's
            # run tree re-uploaded as a dataset OWNED BY THE ACCOUNT THAT RUNS is the only
            # resume route across accounts: kernel_sources silently mounts nothing when the
            # source kernel belongs to someone else (measured 2026-09-09, multi-account.md).
            "dataset_sources": [f"odixe0502/{sc['dataset']}",
                                "odixe0502/veremi-nextgen2026-centralized",
                                *dataset_sources],
            "kernel_sources": list(kernel_sources), "competition_sources": []}
    return nb, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--wandb-project", default="fd-ids-veremi")
    ap.add_argument("--clients", type=int, nargs="*", default=[20, 50, 100])
    ap.add_argument("--run-tag", default="",
                    help="suffix for run_name: new W&B run id, new checkpoint dir, new "
                         "fingerprint. Use it for a relaunch of a scenario that has already "
                         "logged rounds under the untagged name.")
    ap.add_argument("--probe", action="store_true",
                    help="2-round calibration run: separate run_name, slug and W&B id")
    ap.add_argument("--max-hours", type=float, default=11.0,
                    help="driver stops before a round would pass this many session hours")
    ap.add_argument("--require-resume", action="store_true",
                    help="continuation push: die unless a verified checkpoint is attached")
    ap.add_argument("--kernel-source", action="append", default=[],
                    help="owner/slug of the previous run whose output carries the "
                         "checkpoint; repeatable")
    ap.add_argument("--dataset-source", action="append", default=[],
                    help="owner/slug of a checkpoint dataset holding the previous "
                         "session's run tree; the cross-account resume route; repeatable")
    a = ap.parse_args()
    if a.require_resume and not (a.kernel_source or a.dataset_source):
        ap.error("--require-resume without --kernel-source or --dataset-source would fail "
                 "at the gate every time")
    for K in a.clients:
        nb, meta = build(K, a.owner, a.wandb_project, a.require_resume, a.kernel_source,
                         a.probe, a.max_hours, a.run_tag, a.dataset_source)
        name = meta["code_file"]
        out = ROOT / "papers/fd-ids-2025/notebook" / (f"{K}c_probe" if a.probe else f"{K}c")
        out.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(json.dumps(nb, indent=1))
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"{K:>4}c -> {out}/{name}  ({len(nb['cells'])} cells)")

#!/usr/bin/env python
"""Generate one pFedES training notebook per client scenario.

The notebook body is assembled from papers/pfedes-yi-2025/proj/*.py, which are the same
files the local tests exercise. Editing a module and regenerating is the only supported
path; never hand-edit the .ipynb.

    python scripts/gen_notebook.py --owner minhtriethihi                 # 20c 50c 100c
    python scripts/gen_notebook.py --owner khanhmay0304 --clients 20 --probe --max-hours 2
    python scripts/gen_notebook.py --owner X --clients 100 --run-tag _v2 --require-resume \\
        --dataset-source X/pfedes-100c-v2-ckpt --max-hours 9
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "papers/pfedes-yi-2025/proj"
RUNTIME = json.loads((ROOT / "knowledge/runtime.json").read_text())
META = json.loads((ROOT / "knowledge/meta.json").read_text())
MODULES = ("model", "ckpt", "metrics", "data", "pfedes", "evaluate", "driver", "verify")

# batch from knowledge/DATASET.md §4. Participation is C = 100 % in EVERY scenario
# (owner's decision 2026-09-12, a deliberate departure from the paper's Tables 1-2, which
# use C = 100 %/20 %/10 % for N = 10/50/100): every client trains and is re-evaluated in
# every round, so the mean over clients is a mean over models that have all been trained
# (with C < 100 % and seed 42, client 70 of the 100-client run was never drawn in 50 rounds)
# and the eval carry-forward cache is never exercised.
SCENARIOS = {20: dict(batch=512, participation=1.0, dataset="veremi-fl-20client"),
             50: dict(batch=512, participation=1.0, dataset="veremi-fl-50client"),
             100: dict(batch=256, participation=1.0, dataset="veremi-fl-100client")}
# Owner's decisions of 2026-09-11 (CONTEXT.md §1): AdamW / wd 1e-4, mu = 0.5,
# E = E_fe = 1, 50 rounds. The paper gives no mu or E_fe value (appendix missing) and
# uses SGD 0.01; both deviations are reported with every number.
# 2026-09-13: the learning rate follows a per-round cosine schedule lr -> lr_min over the
# 50 rounds (proj/pfedes.py::lr_at, the afpha rebuild's formula), the owner's single LR
# policy across the sibling rebuilds. Chosen after the constant-rate runs of 12-09 peaked
# on the global test after ~2 local epochs and drifted down from there (CONTEXT.md §12).
PAPER = dict(lr=1e-3, lr_schedule="cosine", lr_min=1e-5, weight_decay=1e-4, mu=0.5,
             rounds=50, local_epochs=1, proxy_epochs=1)


def kaggle_slug(title):
    """Kaggle's own rule: lowercase, every run of non-alphanumerics becomes one hyphen."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "source": s,
                     "execution_count": None, "outputs": []}


CALIBRATION = '''# ---- PROBE ONLY: measure what sm_86 cannot tell us about the T4 before the rounds run.
# Train: one pFedES client step (both phases) eager vs compiled. Eval: one folded model over
# the FULL test set, eager vs compiled, at two batch sizes. Everything is freed afterwards
# so the two workers start with the whole GPU.
import gc, time, torch
from proj.model import build_model, build_proxy
from proj.pfedes import layout, flatten, unflatten_into, client_update
from proj.evaluate import fold_bn, load_folded, eval_model
from proj import driver as D

CAL = {}
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
torch.backends.cudnn.benchmark = True
for _n in ('recompile_limit', 'cache_size_limit'):
    if hasattr(torch._dynamo.config, _n): setattr(torch._dynamo.config, _n, 64)
cache = Path(CFG["cache"])
TX = D._resident(cache / "test_X.f16.npy", dev); TY = D._resident(cache / "test_y.u8.npy", dev)
lo, hi = spans[min(spans, key=lambda c: spans[c][1] - spans[c][0])]   # smallest client
hi = min(hi, lo + 400 * CFG["batch"])                                  # ~400 steps per phase
X = torch.from_numpy(np.load(cache / "train_X.f16.npy", mmap_mode="r")[lo:hi].copy()).to(dev)
Y = torch.from_numpy(np.load(cache / "train_y.u8.npy", mmap_mode="r")[lo:hi].copy()).to(dev)

def one_client(Fc, Fe, Gc, Ge):
    optF = torch.optim.AdamW(Fe.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"], fused=True)
    optG = torch.optim.AdamW(Ge.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"], fused=True)
    scF = torch.amp.GradScaler("cuda"); scG = torch.amp.GradScaler("cuda")
    scF.scale(torch.zeros(1, device=dev)); scG.scale(torch.zeros(1, device=dev))
    g = torch.Generator(device=dev); g.manual_seed(1)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    a1, n1, a2, n2 = client_update(Fc, Fe, Gc, Ge, optF, optG, scF, scG, X, Y, 0, hi - lo, CFG, g)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (n1 + n2) * 1000, n1 + n2, int(a1["skips"]) + int(a2["skips"])

torch.manual_seed(CFG["seed"])
Fe = build_model(CFG).to(dev).train(); Ge = build_proxy(CFG).to(dev).train()
fk, ik, _ = layout(Fe); gk, gik, _ = layout(Ge)
f0, i0 = flatten(Fe, fk, ik); g0, gi0 = flatten(Ge, gk, gik)
ms, n, sk = one_client(Fe, Fe, Ge, Ge)
CAL["train_eager_ms_per_phase_step"] = ms
print(f"train eager   : {ms:.2f} ms per phase-step ({n} steps, {sk} skipped) at batch {CFG['batch']}")
unflatten_into(Fe, f0, i0, fk, ik); unflatten_into(Ge, g0, gi0, gk, gik)
t0 = time.perf_counter()
Fc, Gc = D._compile_train(Fe, Ge, CFG, dev, X[:CFG["batch"]].float(), Y[:CFG["batch"]].long())
CAL["compile_seconds"] = time.perf_counter() - t0
CAL["train_backend"] = "compiled" if Fc is not Fe else "eager"
if Fc is not Fe:
    unflatten_into(Fe, f0, i0, fk, ik); unflatten_into(Ge, g0, gi0, gk, gik)
    one_client(Fc, Fe, Gc, Ge)                                  # warm the graphs
    unflatten_into(Fe, f0, i0, fk, ik); unflatten_into(Ge, g0, gi0, gk, gik)
    ms, n, sk = one_client(Fc, Fe, Gc, Ge)
    CAL["train_compiled_ms_per_phase_step"] = ms
    print(f"train compiled: {ms:.2f} ms per phase-step ({n} steps, {sk} skipped) | "
          f"{CAL['train_eager_ms_per_phase_step']/ms:.2f}x | compile+gate {CAL['compile_seconds']:.0f}s")
del Fc, Gc

Te = fold_bn(build_model(CFG).to(dev)); load_folded(Te, Fe)
for eb in (8192, 16384):
    c = dict(CFG, eval_batch=eb)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    cm, nf, _ = eval_model(Te, Te, TX, TY, c); torch.cuda.synchronize()
    r = TX.shape[0] / (time.perf_counter() - t0); CAL[f"eval_eager_folded_{eb}"] = r
    print(f"eval eager-folded  batch {eb:>5}: {r:,.0f} rows/s")
    if CFG["compile"]:
        Tc = D._compile_eval(Te, c, dev, TX[:eb].float())
        if Tc is not Te:
            eval_model(Tc, Te, TX[:4 * eb], TY[:4 * eb], c)          # warm
            torch.cuda.synchronize(); t0 = time.perf_counter()
            cm2, nf2, _ = eval_model(Tc, Te, TX, TY, c); torch.cuda.synchronize()
            r = TX.shape[0] / (time.perf_counter() - t0); CAL[f"eval_compiled_folded_{eb}"] = r
            print(f"eval compiled-folded batch {eb:>5}: {r:,.0f} rows/s | "
                  f"|dCM|={int((cm2 - cm).abs().sum())} cells of {TX.shape[0]}")
            torch._dynamo.reset()
        del Tc
del Te, Fe, Ge, X, Y, TX, TY, f0, g0
gc.collect(); torch.cuda.empty_cache(); torch._dynamo.reset()
CAL["vram_after_free_gb"] = torch.cuda.memory_allocated(dev) / 2**30
print("calibration:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in CAL.items()}))
(C.run_dir(CFG["run_name"]) / "reports" / "calibration.json").write_text(json.dumps(CAL, indent=1))
if run is not None:
    run.summary.update({f"cal_{k}": v for k, v in CAL.items()})'''


def build(K, owner, wandb_project, require_resume=False, kernel_sources=(),
          probe=False, max_hours=11.0, run_tag="", dataset_sources=(), eval_batch=16384,
          session=0):
    sc = SCENARIOS[K]
    # A probe measures the T4 and is thrown away: its own run_name keeps its checkpoints,
    # its W&B run and its fingerprint separate from the real run of the same scenario.
    # run_tag does the same for a relaunch: W&B is opened with resume="allow" on a fixed
    # id, and run_name is in ckpt.FINGERPRINT_KEYS.
    run_name = f"pfedes_{K}c_probe" if probe else f"pfedes_{K}c{run_tag}"
    rounds = 2 if probe else PAPER["rounds"]
    # Kaggle derives the kernel slug from the TITLE and ignores `id` when the two disagree.
    # A continuation session gets its own slug (" s2" -> "-s2") and attaches the previous
    # session's kernel; run_name is unchanged so the fingerprint and the import still match.
    title = f"pFedES VeReMi {K} clients" + (" probe" if probe else "") \
        + (f" {run_tag.strip('_').replace('_', ' ')}" if run_tag else "") \
        + (f" s{session}" if session else "")
    slug = kaggle_slug(title)
    K_sel = round(sc["participation"] * K)
    full_part = K_sel == K
    # With C = 100 % nobody is ever left out, so the carry-forward cache below never fires
    # and Eq. (11) normalises over the whole population -- say so instead of describing a
    # selection that does not happen.
    eval_note = ("Every client trains in every round, so every confusion matrix is "
                 "recomputed from scratch each round: the carry-forward cache for "
                 "unselected clients is never used and `cache_mismatch` stays 0."
                 if full_part else
                 "A client that was not selected keeps its weights, so its confusion "
                 "matrix is carried forward exactly and re-checked by a full "
                 "re-evaluation every 10 rounds and at the last round.")
    agg_note = (f"Eq. (11) aggregation of θ over all K = {K_sel} clients (C = 100 %)"
                if full_part else
                f"Eq. (11) aggregation of θ over the K = {K_sel} selected clients "
                f"(C = {sc['participation']:.0%})")
    dev_note = ("Eq. (11) normalises over the participating set, which at C = 100 % is "
                "every client; C is 100 % in all three scenarios by the owner's decision, "
                "not the paper's 100 %/20 %/10 %"
                if full_part else
                "Eq. (11) normalises over the selected set")
    cells = [md(f"""# pFedES on VeReMi NextGen — {K} clients

Proxy homogeneous feature-extractor sharing (Yi et al., *pFedES*), Eq. (4)–(11), with
**DAGSNet** (395,024 params) as every client's personalized classifier F_k and a
DAGSNet-type proxy extractor G (407,874 params, output dim 66) shared through the server,
on VeReMi NextGen (16 classes, 66 features).

| from the paper | from `knowledge/` and the owner's decisions |
|---|---|
| iterative training: freeze G → train F_k on [G(x); x] with μ-weighted CE; freeze F_k → train G | DAGSNet architecture, 66-feature order, class order, seed 42 |
| {agg_note} | batch {sc['batch']}, {rounds} rounds × E = E_fe = 1, AdamW / wd {PAPER['weight_decay']}, lr {PAPER['lr']} → {PAPER['lr_min']} cosine per round, μ = {PAPER['mu']} |

**Evaluation:** every round, every client's F_k(x) on the full 10,761,343-row test set;
10 metrics per client, mean/std/min/max over all {K} clients. {eval_note}

**Deviations, all deliberate:** proxy is DAGSNet-type not the paper's 2-conv CNN; AdamW
not SGD, with a per-round cosine learning-rate schedule ({PAPER['lr']} at round 1 →
{PAPER['lr_min']} at round {rounds}, constant within a round, shared by η_ω and η_θ) instead of
the paper's constant 0.01; μ and E_fe are the owner's values (the paper's appendix is
unavailable); Step 1 runs its two forwards as one concatenated batch (BatchNorm sees both
halves); {dev_note}. Per-round output is **weights only** (G + every F_k as
state_dict tensors); `proj/ckpt.py::load_weights` rebuilds them at any round, and the last
cell re-derives every published metric from the confusion matrices on disk."""),

    code("""import os, subprocess, sys, time, torch
T0 = time.monotonic()     # session clock: the 12 h cap charges for spawn and compile too.
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
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.makedirs("/kaggle/working/proj", exist_ok=True)
open("/kaggle/working/proj/__init__.py", "w").close()
sys.path.insert(0, "/kaggle/working")"""),

    code(f'''CFG = dict(
    # --- architecture: knowledge/ARCHITECTURE.md, frozen (F_k 395,024 / G 407,874 params)
    patch_len=6, stem_ch=96, dense_growth=32, dense_layers=3,
    incep_modules=2, fire_modules=3, dropout=0.1,
    num_classes=16, n_features=66,
    # --- pFedES: owner's decisions 2026-09-11 (paper: SGD 0.01; mu, E_fe unspecified);
    #     LR schedule 2026-09-13: cosine per round lr -> lr_min (proj/pfedes.py::lr_at)
    lr={PAPER["lr"]}, lr_schedule="{PAPER["lr_schedule"]}", lr_min={PAPER["lr_min"]},
    weight_decay={PAPER["weight_decay"]}, mu={PAPER["mu"]},
    local_epochs={PAPER["local_epochs"]}, proxy_epochs={PAPER["proxy_epochs"]},
    participation={sc["participation"]},        # owner 2026-09-12: C = 100 % in all scenarios
    rounds={rounds}, clip=1.0, seed=42,
    # --- compute: knowledge/DATASET.md 4
    n_clients={K}, batch={sc["batch"]}, eval_batch={eval_batch},
    device="cuda", world_size=2, compile=True,
    # Each client starts fresh GradScalers at 2**16 and spends a few steps calibrating.
    # Fixed before the first measurement so it cannot be widened afterwards.
    max_skips_per_client=16,
    # Clients not selected in a round carry their confusion matrix; everyone is re-evaluated
    # every eval_all_every rounds and at the last round, and the cache is checked then.
    eval_all_every={1000 if probe else 10},
    preds_rounds={[] if probe else [rounds]},   # (N, 10.76 M) uint8 per listed round
    finalize_reserve_seconds=900,   # never start a round that leaves no time to commit it
    run_name="{run_name}",
    cache="/kaggle/temp/veremi_cache",
    max_seconds={max_hours} * 3600,   # 12 h hard cap; leave room to finalize artifacts
    require_resume={require_resume},
)
# data_id is filled in below, once the feature order and scaler are known. It is part of
# proj/ckpt.py FINGERPRINT_KEYS, so a run cannot resume across a changed preprocessing.
for k, v in CFG.items(): print(f"{{k:>20}} = {{v}}")''')]

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

# What the fingerprint could not otherwise see: a permuted feature order, a re-fitted
# scaler or a different partition keep every shape identical.
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

# content_id reads the labels that are actually cached, hit or miss: the row counts and
# the class histogram change when the partition or the file contents change.
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
print(f"prepack {{time.time()-t0:.1f}}s | {{n_train:,}} train / {{CFG['n_test']:,}} test rows")''')]

    if probe:
        cells.append(code(CALIBRATION))

    cells += [
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
from proj.model import build_model, build_proxy, N_PARAMS_MODEL, N_PARAMS_PROXY
from proj.metrics import METRIC_KEYS

d = C.run_dir(CFG["run_name"])
# y_true from the RUN, not from /kaggle/temp: the cache is gone with the session, and the
# check has to be the same one someone can repeat after downloading the output alone.
ok, lines = verify_run(d, cfg=CFG, build_model=build_model, build_proxy=build_proxy,
                       expect_model=N_PARAMS_MODEL, expect_proxy=N_PARAMS_PROXY,
                       y_true_path=d / "reports" / "y_true.u8.npy", full=True)
print("\\n".join(lines))

last = C.last_complete_round(d, C.fingerprint(CFG)) or 0
print(f"\\nrounds verified : {last} / {CFG['rounds']}")
if last < CFG["rounds"]:
    print(f"  INCOMPLETE — attach this notebook's output (or a checkpoint dataset of it) to "
          f"the next push and regenerate with --require-resume to continue at round {last + 1}")
rows = [r for r in csv.DictReader(open(d / "history.csv")) if int(r["round"]) <= last]
if rows:
    fin = rows[-1]
    # The headline is the LAST round, fixed before the run. best-f1 is chosen on the test
    # set after seeing it, so it is a description of the curve and not a second result.
    print(f"\\nresult at round {fin['round']} (mean over {CFG['n_clients']} clients; "
          f"std / min / max of f1_macro {float(fin['f1_macro_std']):.4f} / "
          f"{float(fin['f1_macro_min']):.4f} / {float(fin['f1_macro_max']):.4f}):")
    for k in METRIC_KEYS: print(f"  {k:<20} {float(fin[k]):.6f}")
    b = max(rows, key=lambda r: float(r["f1_macro"]))
    print(f"\\n[descriptive only] best mean f1_macro {float(b['f1_macro']):.6f} "
          f"at round {b['round']} — picked on test, not a reported result")
    mm = sum(int(r.get("cache_mismatch", 0) or 0) for r in rows)
    print(f"eval cache re-check deviations over all rounds: {mm} client-rounds "
          f"(0 once client c is pinned to worker c % 2; the verifier above bounds any "
          f"cross-GPU deviation at proj.verify.CACHE_TOL_ROWS rows)")

# Calibration, from THIS session's rounds only (the CSV would mix in imported rounds).
sec = [float(r["seconds"]) for r in hist]
overhead = (time.monotonic() - T0) - sum(sec)
print(f"\\nbackend  : train {CFG.get('backend', '?')} | eval {CFG.get('backend_eval', '?')}")
print(f"session  : {len(hist)} round(s) here | startup+prepack+compile {overhead/60:.1f} min"
      f" | verify and W&B are outside this figure")
if len(sec) < 2:
    print("timing   : need 2 completed rounds to separate startup from steady state; "
          f"got {len(sec)}. No projection.")
else:
    steady = sum(sec[1:]) / len(sec[1:])
    vt = max(float(r.get("vram_train_gb", 0) or 0) for r in hist)
    ve = max(float(r.get("vram_eval_gb", 0) or 0) for r in hist)
    tr = sum(float(r["train_sec"]) for r in hist[1:]) / len(hist[1:])
    ev = sum(float(r["eval_sec"]) for r in hist[1:]) / len(hist[1:])
    print(f"timing   : rounds {[round(x) for x in sec[-3:]]}s | steady {steady:.0f}s/round "
          f"= train {tr:.0f}s + eval {ev:.0f}s (evaluated {hist[-1]['evaluated']} clients) + commit")
    print(f"VRAM     : train {vt:.2f} GiB/GPU | eval {ve:.2f} GiB/GPU (of 16)")
    print(f"projected: {CFG['rounds']} rounds = "
          f"{(steady*CFG['rounds'] + overhead)/3600:.2f} h "
          f"({(steady*CFG['rounds'])/3600:.2f} h of rounds + {overhead/3600:.2f} h startup) "
          f"— full-eval rounds cost more than the steady figure")
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
            # resume route across accounts (kernel_sources silently mounts nothing then).
            "dataset_sources": [f"odixe0502/{sc['dataset']}",
                                "odixe0502/veremi-nextgen2026-centralized",
                                *dataset_sources],
            "kernel_sources": list(kernel_sources), "competition_sources": []}
    return nb, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--wandb-project", default="pfedes-veremi")
    ap.add_argument("--clients", type=int, nargs="*", default=[20, 50, 100])
    ap.add_argument("--run-tag", default="",
                    help="suffix for run_name: new W&B run id, new checkpoint dir, new "
                         "fingerprint. Use it for a relaunch of a scenario that has already "
                         "logged rounds under the untagged name.")
    ap.add_argument("--probe", action="store_true",
                    help="2-round calibration run with a T4 micro-benchmark cell: separate "
                         "run_name, slug and W&B id")
    ap.add_argument("--max-hours", type=float, default=11.0,
                    help="driver stops before a round would pass this many session hours")
    ap.add_argument("--eval-batch", type=int, default=16384)
    ap.add_argument("--session", type=int, default=0,
                    help="continuation session number N >= 2: own slug '-sN' and output "
                         "dir '<K>c_sN'; pair with --require-resume --kernel-source")
    ap.add_argument("--require-resume", action="store_true",
                    help="continuation push: die unless a verified checkpoint is attached")
    ap.add_argument("--kernel-source", action="append", default=[],
                    help="owner/slug of the previous run whose output carries the "
                         "checkpoint (same account only); repeatable")
    ap.add_argument("--dataset-source", action="append", default=[],
                    help="owner/slug of a checkpoint dataset holding the previous "
                         "session's run tree; the cross-account resume route; repeatable")
    a = ap.parse_args()
    if a.require_resume and not (a.kernel_source or a.dataset_source):
        ap.error("--require-resume without --kernel-source or --dataset-source would fail "
                 "at the gate every time")
    if a.session and (a.session < 2 or not a.require_resume or a.probe):
        ap.error("--session N needs N >= 2 and --require-resume, and is not for a probe")
    for K in a.clients:
        nb, meta = build(K, a.owner, a.wandb_project, a.require_resume, a.kernel_source,
                         a.probe, a.max_hours, a.run_tag, a.dataset_source, a.eval_batch,
                         a.session)
        name = meta["code_file"]
        out = ROOT / "papers/pfedes-yi-2025/notebook" / (
            f"{K}c_probe" if a.probe else f"{K}c_s{a.session}" if a.session else f"{K}c")
        out.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(json.dumps(nb, indent=1))
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"{K:>4}c -> {out}/{name}  ({len(nb['cells'])} cells)")

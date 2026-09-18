#!/usr/bin/env python
"""Static gate on the generated pFedES notebooks. Cheap checks that each cost a Kaggle version."""
import ast, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "papers/pfedes-yi-2025/proj"
RUNTIME = json.loads((ROOT / "knowledge/runtime.json").read_text())
sys.path.insert(0, str(ROOT / "papers/pfedes-yi-2025"))
from proj.ckpt import FINGERPRINT_KEYS

MODULES = ("model", "ckpt", "metrics", "data", "pfedes", "evaluate", "driver", "verify")
fails = []


def cfg_keywords(cell):
    """The CFG dict as {name: value}, parsed rather than grepped. A string search cannot
    tell CFG['batch']=512 from a batch=512 that appears in a comment or a docstring."""
    tree = ast.parse(cell)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CFG"
                and isinstance(node.value, ast.Call)):
            out = {}
            for kw in node.value.keywords:
                try: out[kw.arg] = ast.literal_eval(kw.value)
                except Exception: out[kw.arg] = "<expr>"
            return out
    return {}


def undefined_names(cells):
    """Names a cell reads that NO earlier cell ever binds.

    Cells share one namespace and run in order, so `T0` used in the last cell is only safe
    because the first cell assigns it. A NameError here surfaces after prepack, hours into
    a session and after the quota is spent. Binding is collected loosely (anywhere in the
    cell, including comprehension and function-parameter names) so this reports only the
    genuinely unresolved case, never a within-cell ordering nit."""
    import builtins
    bound = set(dir(builtins)) | {"__name__", "__file__"}
    missing = []
    for i, cell in enumerate(cells):
        tree = ast.parse(cell)
        binds, reads = set(), set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                (binds if isinstance(n.ctx, (ast.Store, ast.Del)) else reads).add(n.id)
            elif isinstance(n, ast.alias):
                binds.add((n.asname or n.name).split(".")[0])
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                binds.add(n.name)
            elif isinstance(n, ast.arg):
                binds.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                binds.add(n.name)
        for r in sorted(reads - binds - bound):
            missing.append(f"cell {i} reads {r!r}, never bound by an earlier cell")
        bound |= binds
    return missing


# CFG is parsed once from its dict literal, so anything that writes to it afterwards is
# invisible to every value check below. `CFG['lr'] = 0.25` two cells later passed the whole
# gate. Only these keys are filled in later by design.
CFG_LATE_KEYS = {"data_id", "n_test", "content_id", "backend", "backend_eval", "startup_seconds"}


def cfg_overrides(cells):
    """Assignments into CFG outside its declaration, and any rebinding of the name."""
    out = []
    for i, cell in enumerate(cells):
        tree = ast.parse(cell)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                            and t.value.id == "CFG"):
                        try: key = ast.literal_eval(t.slice)
                        except Exception: key = "<expr>"
                        if key not in CFG_LATE_KEYS:
                            out.append(f"cell {i} assigns CFG[{key!r}] after the declaration")
                    elif isinstance(t, ast.Name) and t.id == "CFG" \
                            and not isinstance(node.value, ast.Call):
                        out.append(f"cell {i} rebinds CFG")
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "update" and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == "CFG"):
                out.append(f"cell {i} calls CFG.update()")
    return out


def marker_written_last(cells):
    """The cache marker must be the last write in its block, checked as statements and not
    as string positions: the first textual occurrence of 'test_y.u8.npy' is its entry in the
    FILES tuple, so an index comparison was satisfied by a filename in a list."""
    for cell in cells:
        tree = ast.parse(cell)
        mf = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "write_text" and isinstance(n.func.value, ast.Name)
              and n.func.value.id == "MF"]
        saves = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "save" and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == "np"]
        if mf:
            return bool(saves) and min(mf) > max(saves)
    return False


NB = ROOT / "papers/pfedes-yi-2025/notebook"
# Probe notebooks are validated exactly like the real ones. They run on the same T4 and
# cost the same quota; the only sanctioned difference is the round count.
targets = [(K, p, f"{K}c_probe" if p else f"{K}c") for K in (20, 50, 100) for p in (False, True)
           if (NB / (f"{K}c_probe" if p else f"{K}c")).is_dir()]
# continuation sessions (--session N): production settings, own slug, must attach a source
targets += [(K, False, d.name) for K in (20, 50, 100)
            for d in sorted(NB.glob(f"{K}c_s[0-9]*")) if d.is_dir()]

for K, probe, dirname in targets:
    d = NB / dirname
    meta = json.loads((d / "kernel-metadata.json").read_text())
    # The notebook filename comes from kernel-metadata, which is what Kaggle actually reads.
    # Deriving it here instead made the validator crash on a relaunch generated with
    # --run-tag: it looked for fdids_20c.ipynb while the pushed file was fdids_20c_v2.ipynb.
    # The run_name shape is still checked below, so nothing is loosened by trusting it.
    name = meta["code_file"][:-len(".ipynb")]
    nb = json.loads((d / f"{name}.ipynb").read_text())
    tag = dirname.replace("_", "-")

    def chk(cond, msg):
        if not cond: fails.append(f"{tag}: {msg}")

    # -- notebook document metadata: omitting this block gave ZERO GPUs in this repo
    kg = nb["metadata"].get("kaggle", {})
    chk(kg.get("accelerator") == "nvidiaTeslaT4", "nb.metadata.kaggle.accelerator")
    chk(kg.get("isGpuEnabled") is True, "nb.metadata.kaggle.isGpuEnabled")
    chk(nb["metadata"]["language_info"]["version"] == "3.12", "language_info is the image's")

    # -- kernel-metadata
    chk(meta["is_private"] is True, "is_private must be explicit True")
    chk(meta["machine_shape"] == "NvidiaTeslaT4",
        f"machine_shape {meta['machine_shape']!r} is silently coerced to one P100")
    chk(meta["docker_image"] == RUNTIME["docker_image"], "docker_image not the pinned digest")
    chk(meta["kernel_type"] == "notebook", "kernel_type")
    chk(meta["code_file"] == f"{name}.ipynb", "code_file mismatch")
    chk("/" in meta["id"], "slug must be OWNER/NAME, a bare name is rejected")
    # Kaggle ignores `id` when the title resolves to a different slug, and silently creates
    # the kernel at the title's slug instead -- discovered by losing a push to it.
    chk(meta["id"].split("/", 1)[1]
        == re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-"),
        f"id {meta['id']!r} does not match the slug Kaggle derives from title "
        f"{meta['title']!r}; Kaggle will use the title's slug")
    chk(meta["dataset_sources"][:2] == [f"odixe0502/veremi-fl-{K}client",
                                        "odixe0502/veremi-nextgen2026-centralized"],
        f"dataset_sources {meta['dataset_sources']} does not start with the {K}-client "
        "partition plus the centralized test set")
    chk(meta["enable_gpu"] is True and meta["enable_internet"] is True,
        "enable_gpu and enable_internet must both be true")

    src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    # -- every code cell must parse (strip the %%writefile line magics)
    for i, s in enumerate(src):
        body = "\n".join(l for l in s.split("\n") if not l.startswith("%%"))
        try: ast.parse(body)
        except SyntaxError as e: fails.append(f"{tag}: cell {i} SyntaxError: {e}")
    joined = "\n".join(src)
    # Ordering checks run over the EXECUTABLE cells only. A %%writefile cell is a string
    # being written to disk, not code that runs here, so `def resolve_resume` inside the
    # embedded ckpt.py would otherwise satisfy an ordering claim about the notebook's flow.
    exe = [s for s in src if not s.startswith("%%")]
    flow = "\n".join(exe)

    # -- every embedded module is byte-for-byte the file the local tests exercise
    embedded = {}
    for s in src:
        if s.startswith("%%writefile"):
            first, _, body = s.partition("\n")
            embedded[Path(first.split()[-1]).stem] = body.rstrip()
    chk(set(embedded) == set(MODULES), f"embedded modules {sorted(embedded)} != {list(MODULES)}")
    for m, body in embedded.items():
        want = (PROJ / f"{m}.py").read_text().rstrip()
        chk(body == want, f"embedded proj/{m}.py is STALE — regenerate the notebook")

    # -- the hardware assert must exist and run before training
    chk("device_count()" in src[0] and "== 2" in src[0], "GPU assert not in the first cell")
    chk("(7, 5)" in src[0], "sm_75 capability assert missing")
    chk("T0 = time.monotonic()" in src[0],
        "session clock T0 must be a monotonic reading taken in the first cell")
    chk(flow.index("device_count()") < flow.index("from proj.driver import"),
        "GPU assert must precede training")
    # -- torch must never be reinstalled on a Kaggle GPU image
    chk("pip install torch" not in joined, "reinstalling torch breaks CUDA on this image")

    # -- effective CFG, parsed. These are the values the run actually uses.
    # find the CFG cell by content: the markdown cell is not in `src`, so its index moved
    # once already and an index here would silently validate the wrong cell.
    cfg = next((c for c in (cfg_keywords(x) for x in src if x.startswith("CFG = dict(")) if c),
               {})
    for key, val in (("lr", 0.001), ("lr_schedule", "cosine"), ("lr_min", 1e-5),
                     ("weight_decay", 0.0001), ("mu", 0.5),
                     ("rounds", 2 if probe else 50),
                     ("local_epochs", 1), ("proxy_epochs", 1),
                     ("participation", 1.0),   # owner 2026-09-12: C = 100 % everywhere
                     ("eval_all_every", 1000 if probe else 10),
                     ("preds_rounds", [] if probe else [50]),
                     ("clip", 1.0), ("seed", 42), ("num_classes", 16), ("n_features", 66),
                     ("world_size", 2), ("n_clients", K), ("compile", True),
                     ("batch", 512 if K != 100 else 256)):
        chk(cfg.get(key) == val, f"CFG {key} = {cfg.get(key)!r}, expected {val!r}")
    # -- the fingerprint must be able to see every scientific parameter, or it protects
    #    nothing. data_id is the one key filled in later, once the scaler is known.
    missing = [k for k in FINGERPRINT_KEYS if k not in cfg and k != "data_id"]
    chk(not missing, f"CFG lacks fingerprint keys {missing}")
    chk('CFG["data_id"]' in flow, "data_id is never assigned")
    chk(flow.index('CFG["data_id"]') < flow.index("C.resolve_resume"),
        "data_id must be set before the fingerprint is used to resolve a checkpoint")

    # -- resume gate must precede the parquet decode
    chk(flow.index("require_resume") < flow.index("load_clients("),
        "require_resume gate must fail before the decode, not after")
    # -- what is attached must match what the notebook claims to be. A fresh run with a
    #    checkpoint dataset attached would import and continue somebody's rounds under a
    #    "from scratch" label; a continuation with nothing attached dies at the gate after
    #    the queue wait. Both are metadata-only mistakes the generator cannot see.
    extra_ds = meta["dataset_sources"][2:]
    if cfg.get("require_resume") is True:
        chk(bool(extra_ds or meta["kernel_sources"]),
            "require_resume=True but no kernel_source or checkpoint dataset is attached")
    else:
        chk(not extra_ds and not meta["kernel_sources"],
            f"a fresh run must not attach checkpoints: extra dataset_sources {extra_ds}, "
            f"kernel_sources {meta['kernel_sources']}")
    if "_s" in dirname:
        chk(cfg.get("require_resume") is True and meta["title"].endswith(f" s{dirname.split('_s')[1]}"),
            "a continuation dir must carry require_resume=True and the matching ' sN' title")
    chk(marker_written_last(exe),
        "the prepack cache marker must be written after every np.save of a cache file")
    for m in cfg_overrides(exe):
        fails.append(f"{tag}: {m}")

    # -- checkpoints are weights-only, and the run is verified from its own artifacts
    chk("save_round_weights" in joined, "per-round weights writer missing")
    chk("weights_only=True" in joined, "checkpoints must load weights_only=True")
    chk("build_proxy=build_proxy" in flow and "expect_proxy=N_PARAMS_PROXY" in flow,
        "the verifier must rebuild G and every F_k, not only F")
    chk("verify_run(" in flow, "summary must re-derive the metrics from the artifacts")
    chk("t_origin=T0" in flow, "the budget deadline must start at session start")
    # A continuation's history.csv holds every imported round. Projecting from it gives a
    # negative overhead and a 50-round estimate of about a minute.
    chk("for r in hist]" in flow,
        "the timing projection must be computed from this session's rounds (hist)")
    chk("time.monotonic() - T0" in flow, "session elapsed must use the monotonic clock")
    chk("y_true.u8.npy" in flow and 'd / "reports"' in flow,
        "the verifier must read y_true from the run, not from the session-local cache")
    # cells share a namespace and run in order: a name no earlier cell binds is a NameError
    # that only appears hours in, after the parquet decode and after the quota is spent
    for m in undefined_names(exe):
        fails.append(f"{tag}: {m}")
    chk(cfg.get("run_name") == name, f"CFG run_name {cfg.get('run_name')!r} != {name!r}")
    # a probe must not be able to overwrite the real run's checkpoints or W&B history
    chk(probe == name.endswith("_probe"), "probe run_name must be distinct")
    chk(re.fullmatch(rf"pfedes_{K}c(_probe|_[a-z0-9]+)?", name) is not None,
        f"run_name {name!r} is not pfedes_{K}c with an optional tag")
    # the probe's micro-benchmark cell must run AFTER the prepack and BEFORE the driver,
    # and only a probe may carry it
    has_cal = "PROBE ONLY" in flow
    chk(has_cal == probe, "calibration cell present iff --probe")
    if has_cal:
        chk(flow.index("PROBE ONLY") > flow.index("content_id") and
            flow.index("PROBE ONLY") < flow.index("from proj.driver import"),
            "calibration cell must sit between prepack and training")

for K in (20, 50, 100):
    if not (NB / f"{K}c").is_dir():
        fails.append(f"the {K}-client notebook is missing entirely")

print("\n".join(f"  FAIL {f}" for f in fails) if fails else
      "  all checks passed for " + ", ".join(t[2].replace("_", "-") for t in targets))
sys.exit(1 if fails else 0)

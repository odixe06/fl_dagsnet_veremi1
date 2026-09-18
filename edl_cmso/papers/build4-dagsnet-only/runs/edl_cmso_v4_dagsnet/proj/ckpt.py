"""The resume contract: kill the run at any instant, lose at most one round."""
import os, json, glob, shutil, hashlib, random
from pathlib import Path
import numpy as np, torch

SUBDIRS = ("checkpoints", "metrics", "preds", "confusion", "reports", "logs")


def run_dir(run_name):
    d = Path("/kaggle/working/runs") / run_name
    for s in SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d


def resolve_resume(run_name):
    """working/ first, then any attached input (a previous kernel's committed output).
    /kaggle/working is wiped when a NEW session starts, so cross-session resume depends
    on the dead run's output being attached as a data source."""
    d = run_dir(run_name)
    local = d / "checkpoints" / "last.pt"
    if local.exists():
        return local
    pats = [f"/kaggle/input/*/runs/{run_name}/checkpoints",
            f"/kaggle/input/*/checkpoints",
            f"/kaggle/input/*/*/runs/{run_name}/checkpoints"]
    for pat in pats:
        for src in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(src, "last.pt")):
                print(f"[resume] importing checkpoints from {src}", flush=True)
                for f in glob.glob(os.path.join(src, "*")):
                    shutil.copy2(f, d / "checkpoints")
                for sub in ("metrics", "preds", "confusion", "reports"):
                    peer = Path(src).parent / sub
                    if peer.is_dir():
                        for f in glob.glob(str(peer / "*")):
                            shutil.copy2(f, d / sub)
                # channel_mask.json and extractor_init.pt are the two files that make a
                # cross-session resume of build 2 valid: without the frozen §4.8 state the
                # resumed network would not be the one the mask was selected against.
                for side in ("channel_mask.json", "extractor_init.pt"):
                    fm = Path(src).parent / side
                    if fm.exists(): shutil.copy2(fm, d / side)
                return d / "checkpoints" / "last.pt"
    return None


def fingerprint(cfg, keys=("num_classes", "n_features", "n_selected", "batch_per_gpu",
                           "world_size", "run_name", "d_model", "patch_len")):
    """Shape-affecting config only. A changed lr may resume; a changed n_selected may not.
    n_features is in here for build 2: it is the extractor's input width, fixed at 66, and
    a checkpoint taken at another width describes a different §4.8 stack entirely."""
    payload = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def atomic_save(obj, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp); os.replace(tmp, path)          # os.replace is atomic on POSIX


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu() if hasattr(s["torch"], "cpu") else s["torch"])
    if s.get("cuda") is not None: torch.cuda.set_rng_state_all(s["cuda"])


def save_round(model, opt, scaler, sched, rnd, cfg, metrics, run_name):
    d = run_dir(run_name) / "checkpoints"
    core = getattr(model, "module", model)               # unwrap DDP -> portable state
    blob = {"round": int(rnd), "model": core.state_dict(), "optim": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "sched": sched.state_dict() if sched is not None else None,
            "rng": rng_state(), "cfg": cfg,
            "fingerprint": fingerprint(cfg), "metrics": metrics}
    atomic_save(blob, d / f"ckpt_round_{rnd:03d}.pt")
    atomic_save(blob, d / "last.pt")
    return d / f"ckpt_round_{rnd:03d}.pt"


def load_for_resume(model, opt, scaler, sched, cfg, run_name, device):
    """Returns the round to start from; 0 when there is nothing to resume."""
    d = run_dir(run_name)
    last = d / "checkpoints" / "last.pt"
    if not last.exists():
        print("[resume] no checkpoint found — starting from round 0", flush=True); return 0
    ck = torch.load(last, map_location="cpu", weights_only=False)
    if ck.get("fingerprint") != fingerprint(cfg):
        raise RuntimeError(
            f"Checkpoint fingerprint {ck.get('fingerprint')} != current {fingerprint(cfg)}. "
            "A shape-affecting config changed. Rename CFG.run_name to start a new run, "
            "or restore the original config to resume this one.")
    getattr(model, "module", model).load_state_dict(ck["model"])
    opt.load_state_dict(ck["optim"])
    if scaler is not None and ck.get("scaler"): scaler.load_state_dict(ck["scaler"])
    if sched  is not None and ck.get("sched"):  sched.load_state_dict(ck["sched"])
    try: set_rng_state(ck["rng"])
    except Exception as e: print("[resume] RNG restore skipped:", e, flush=True)
    start = int(ck["round"]) + 1
    print(f"[resume] loaded round {ck['round']} -> continuing at round {start}", flush=True)
    return start

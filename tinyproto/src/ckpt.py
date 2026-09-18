"""Crash-safe per-round commit and cross-session resume.

The rule: kill the process at any instant and the next run loses at most one round.

Write order inside a round -- the marker is a promise and is published last, and the resume
state is keyed by round so it can never disagree with the marker
(`references/verifying-artifacts.md` §1, `perf-federated.md` §8b):

    weights/round_NNN.pt
    protos/round_NNN.pt
    confusion/round_NNN.npz
    metrics/round_NNN.json
    client_log/round_NNN.csv
    metrics/history.csv            (rebuilt from every per-round JSON, atomically)
    resume/round_NNN.pt            (round-keyed, BEFORE the marker)
    complete/round_NNN.done        (absolutely last)

Only weights are kept for every round. The resume blob carries per-client Adam state and is
~316 MB at 100 clients, so exactly one is retained: the previous one is deleted only after the
new marker lands.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch

SUBDIRS = ("weights", "protos", "confusion", "metrics", "client_log",
           "preds", "reports", "resume", "complete", "logs")

# Scientific settings and worker ownership are immutable across resume.
FINGERPRINT_KEYS = ("scenario", "n_clients", "num_classes", "n_features", "feature_dim",
                    "cps_s", "mask_seed", "batch", "rounds", "local_epochs", "seed",
                    "run_name", "data_fingerprint", "test_fingerprint", "model_cfg",
                    "feature_cols", "class_names", "packer_manifest", "assignment",
                    "lr", "weight_decay", "betas", "eps", "clip", "lam", "amp",
                    "mu_kind", "mu_value", "scaler", "artifact_version",
                    "validation_fingerprint", "eval_group", "eval_batch", "compile")


def run_dir(root: Path, run_name: str) -> Path:
    d = Path(root) / "runs" / run_name
    for s in SUBDIRS:
        (d / s).mkdir(parents=True, exist_ok=True)
    return d


def fingerprint(cfg: dict) -> str:
    missing = [k for k in FINGERPRINT_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"config is missing fingerprint fields: {missing}")
    payload = json.dumps({k: cfg[k] for k in FINGERPRINT_KEYS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def atomic_save(obj, path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_savez(path, **arrays) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def rng_state() -> dict:
    ns = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (ns[0], ns[1].tolist(), int(ns[2]), int(ns[3]), float(ns[4])),
            "torch": torch.get_rng_state(),
            # Parent owns no GPU model. Do not initialize CUDA just to checkpoint its RNG;
            # workers save only their assigned device, not every GPU in the machine.
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    ns = s["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    t = s["torch"]
    torch.set_rng_state(t.cpu() if hasattr(t, "cpu") else t)
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(s["cuda"].cpu())


_ROUND_RE = re.compile(r"^round_(\d+)")


def round_of(path: Path) -> int:
    """Round number from a file named round_NNN[.anything]. Worker shards are
    `round_007.w1.pt`, so `stem.split("_")[1]` yields '007.w1' and int() explodes."""
    m = _ROUND_RE.match(path.name)
    if not m:
        raise ValueError(f"not a round-keyed artifact: {path.name}")
    return int(m.group(1))


def completed_rounds(d: Path) -> list[int]:
    return sorted(round_of(p) for p in (d / "complete").glob("round_*.done"))


def last_complete(d: Path) -> int:
    """0 when nothing is committed. Rounds are 1-based."""
    done = completed_rounds(d)
    return done[-1] if done else 0


def commit_round(d: Path, rnd: int, *, weights: dict, protos: dict, confusion: dict,
                 metrics: dict, client_rows: list[dict], client_columns: list[str],
                 resume: dict, history_rows: list[dict], history_columns: list[str],
                 preds: dict | None = None) -> None:
    """Publish one round. Everything before the marker; the marker last; nothing after it
    except deleting the now-superseded resume blob."""
    from .metrics import atomic_write_csv, atomic_write_json

    atomic_save(weights, d / "weights" / f"round_{rnd:03d}.pt")
    atomic_save(protos, d / "protos" / f"round_{rnd:03d}.pt")
    atomic_savez(d / "confusion" / f"round_{rnd:03d}.npz", **confusion)
    atomic_write_json(d / "metrics" / f"round_{rnd:03d}.json", metrics)
    atomic_write_csv(d / "client_log" / f"round_{rnd:03d}.csv", client_rows, client_columns)
    if preds:
        for name, arr in preds.items():
            # np.save appends ".npy" unless the NAME already ends with it, so write through a
            # file handle -- a path-based save to "x.npy.tmp" silently lands on "x.npy.tmp.npy".
            tmp = d / "preds" / f"{name}.npy.tmp"
            with tmp.open("wb") as fh:
                np.save(fh, arr)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, d / "preds" / f"{name}.npy")
    atomic_write_csv(d / "metrics" / "history.csv", history_rows, history_columns)
    atomic_save(resume, d / "resume" / f"round_{rnd:03d}.pt")

    marker = d / "complete" / f"round_{rnd:03d}.done"
    tmp = marker.with_suffix(".done.tmp")
    files = [d / sub / f"round_{rnd:03d}{ext}" for sub, ext in
             (("weights", ".pt"), ("protos", ".pt"), ("confusion", ".npz"),
              ("metrics", ".json"), ("client_log", ".csv"), ("resume", ".pt"))]
    files += sorted((d / "resume").glob(f"round_{rnd:03d}.w*.pt"))
    files += [d / "preds" / f"{name}.npy" for name in (preds or {})]
    files += [d / "masks.npy"]
    hashes = {str(p.relative_to(d)): file_digest(p) for p in files}
    with tmp.open("w") as fh:
        json.dump({"round": rnd, "sha256": hashes}, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, marker)

    for old in (d / "resume").glob("round_*.pt"):          # only after the marker
        if round_of(old) != rnd:
            old.unlink(missing_ok=True)


def rebuild_history(d: Path, history_columns: list[str], row_fn) -> list[dict]:
    """history.csv is derived; the per-round JSON is the source. Rebuilding it on every write
    repairs damage an earlier crash already did instead of carrying a truncated file forever."""
    rows = []
    for rnd in completed_rounds(d):
        p = d / "metrics" / f"round_{rnd:03d}.json"
        if p.exists():
            rows.append(row_fn(json.loads(p.read_text())))
    return rows


def repair_history(d: Path) -> None:
    """Recover derived CSV even when a completed run needs no further training."""
    from .metrics import atomic_write_csv, history_columns, history_row
    done = completed_rounds(d)
    if done:
        docs = [json.loads((d / "metrics" / f"round_{r:03d}.json").read_text()) for r in done]
        columns = history_columns(list(docs[-1]["extra"]))
        atomic_write_csv(d / "metrics" / "history.csv",
                         [history_row(j["round"], j["aggregate"], j["extra"]) for j in docs], columns)


def load_resume(d: Path, cfg: dict) -> tuple[int, dict | None]:
    """(start_round, resume_blob). start_round is 1 when there is nothing committed."""
    last = last_complete(d)
    if last == 0:
        return 1, None
    blob_path = d / "resume" / f"round_{last:03d}.pt"
    if not blob_path.exists():
        raise RuntimeError(
            f"round {last} is marked complete but {blob_path.name} is missing. The commit was "
            "not atomic, or the resume state was pruned by hand. Refusing to guess.")
    blob = torch.load(blob_path, map_location="cpu", weights_only=True)
    if blob.get("fingerprint") != fingerprint(cfg):
        raise RuntimeError(
            f"resume fingerprint {blob.get('fingerprint')} != current {fingerprint(cfg)}.\n"
            "A run-defining setting changed. Rename run_name to start a new run, or restore "
            "the original config to continue this one.")
    if int(blob["round"]) != last:
        raise RuntimeError(f"resume blob says round {blob['round']}, marker says {last}")
    ok, _, why = _completeness_ok(d)
    if not ok:
        raise RuntimeError(f"incomplete resume bundle: {why}")
    validate_integrity(d)
    return last + 1, blob


# ---------------------------------------------------------------------------
# cross-session import
# ---------------------------------------------------------------------------

def validate_integrity(d: Path) -> None:
    last = last_complete(d)
    for rnd in completed_rounds(d):
        marker = json.loads((d / "complete" / f"round_{rnd:03d}.done").read_text())
        hashes = marker.get("sha256")
        if not hashes:
            raise RuntimeError("legacy marker without integrity manifest; start a new run")
        for rel, expected in hashes.items():
            if rel.startswith("resume/") and rnd != last:
                continue
            p = d / rel
            if not p.is_file() or file_digest(p) != expected:
                raise RuntimeError(f"artifact integrity failed: {rel}")

def _completeness_ok(src: Path) -> tuple[bool, int, str]:
    done = completed_rounds(src)
    if not done:
        return False, 0, "no completed rounds"
    if done != list(range(1, done[-1] + 1)):
        return False, done[-1], f"rounds are not contiguous 1..{done[-1]}: {done}"
    for r in done:
        for rel in (f"weights/round_{r:03d}.pt", f"metrics/round_{r:03d}.json",
                    f"confusion/round_{r:03d}.npz", f"protos/round_{r:03d}.pt",
                    f"client_log/round_{r:03d}.csv"):
            if not (src / rel).exists():
                return False, done[-1], f"round {r} marked complete but {rel} is missing"
    last = done[-1]
    if not (src / "resume" / f"round_{last:03d}.pt").exists():
        return False, last, f"no resume state for the last complete round {last}"
    shards = sorted((src / "resume").glob(f"round_{last:03d}.w*.pt"))
    cfgp = src / "config.json"
    if cfgp.exists():
        try:
            want = len(json.loads(cfgp.read_text())["assignment"])
        except Exception:
            want = len(shards)
        expected = {f"round_{last:03d}.w{r}.pt" for r in range(want)}
        if {p.name for p in shards} != expected:
            return False, last, (f"round {last} has {len(shards)} worker optimizer shards, "
                                 f"config.json declares {want} workers")
    elif not shards:
        return False, last, f"round {last} has no worker optimizer shard"
    return True, last, "ok"


def find_import_source(search_roots: list[Path], run_name: str, fp: str) -> Path | None:
    """Locate a previous session's output for this exact run. Multiple matches are a stop
    condition, never a reason to take the one with the most rounds (verifying-artifacts.md §4)."""
    cands = []
    for base in search_roots:
        if not base.exists():
            continue
        for cfgp in base.rglob("config.json"):
            try:
                c = json.loads(cfgp.read_text())
            except Exception:
                continue
            if c.get("run_name") != run_name:
                continue
            src = cfgp.parent
            if c.get("fingerprint") and c["fingerprint"] != fp:
                continue
            ok, n, why = _completeness_ok(src)
            cands.append((src, ok, n, why))
    usable = [c for c in cands if c[1]]
    if not usable:
        for src, ok, n, why in cands:
            print(f"[import] rejecting {src}: {why}")
        return None
    if len(usable) > 1:
        raise RuntimeError(
            f"{len(usable)} candidate resume sources share fingerprint {fp}:\n" +
            "\n".join(f"  {s} ({n} rounds)" for s, _, n, _ in usable) +
            "\nRefusing to guess which run to continue.")
    return usable[0][0]


def import_previous(src: Path, dst: Path) -> int:
    """Copy a previous session's committed output into the working run directory.

    Markers are copied LAST, and only after everything they certify has arrived; copying in
    path order publishes `complete/` early (it sorts before `weights/`) and an interrupted copy
    then advertises rounds whose weights never landed. `copyfile` + explicit chmod, not `copy2`:
    a read-only `/kaggle/input` mount otherwise carries its mode across and the first rewrite of
    history.csv dies with PermissionError after the import reported success.
    """
    ok, n, why = _completeness_ok(src)
    if not ok:
        raise RuntimeError(f"refusing to import an incomplete source {src}: {why}")
    validate_integrity(src)
    from .metrics import atomic_write_json
    atomic_write_json(dst / "import_pending.json", {"source": str(src)})

    def copy_one(s: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        shutil.copyfile(s, part)
        os.chmod(part, 0o644)
        os.replace(part, target)

    for sub in ("weights", "protos", "confusion", "metrics", "client_log", "preds",
                "reports", "resume", "logs"):
        sd = src / sub
        if not sd.is_dir():
            continue
        for f in sorted(sd.rglob("*")):
            if f.is_file():
                if f.name.endswith((".tmp", ".part")):
                    continue
                if f.name.startswith("round_"):
                    r = round_of(f)
                    if r > n or (sub == "resume" and r != n):
                        continue
                copy_one(f, dst / sub / f.relative_to(sd))
    for f in sorted(src.glob("*.json")):
        if f.name != "import_pending.json":
            copy_one(f, dst / f.name)
    copy_one(src / "masks.npy", dst / "masks.npy")
    for f in sorted((src / "complete").glob("*.done")):     # markers last
        copy_one(f, dst / "complete" / f.name)
    repair_history(dst)
    (dst / "import_pending.json").unlink()
    return n

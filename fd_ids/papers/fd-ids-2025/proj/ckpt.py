"""Weights, resume state and the completion marker — three files, one atomic round.

The per-round file holds WEIGHTS ONLY and loads with weights_only=True. RNG and the round
counter live in a separate resume bundle keyed by the same round, so keeping every round
costs weights and not optimizer moments, and so a reader never has to unpickle arbitrary
objects to look at a checkpoint.

FD-IDS Algorithm 1 line 14 re-initialises every client from w_G^t each round, so client Adam
moments are deliberately discarded at the round boundary and there is no global optimizer at
all -- aggregation is a stateless weighted mean. The persistent state is therefore the round,
the global weights, and nothing else.
"""
import csv, hashlib, json, os, random, shutil
from pathlib import Path
import numpy as np, torch

SUBDIRS = ("weights", "resume", "complete", "metrics", "preds", "confusion", "reports", "logs")

# What a later session imports. `preds` and `logs` are not needed to CONTINUE, but a run
# split across three sessions must still be verifiable from its final output alone, and
# leaving them behind means the last session's output cannot prove anything about the first.
# 630 MB/scenario copied once per continuation is the cheaper side of that trade.
RESUME_SUBDIRS = ("weights", "resume", "metrics", "confusion", "preds", "logs")

# Every input that changes what the numbers mean. `rounds` is deliberately absent: the LR is
# constant and no schedule spans the run, so the weights at round r do not depend on how many
# rounds were planned -- that is what makes a continuation push legal. Paths, world_size,
# compile, eval_batch, max_seconds and require_resume are operational and also absent.
FINGERPRINT_KEYS = (
    "patch_len", "stem_ch", "dense_growth", "dense_layers", "incep_modules",
    "fire_modules", "dropout", "num_classes", "n_features",
    "lr", "lam", "beta", "mu", "temperature", "clip",
    "n_clients", "batch", "local_epochs", "seed",
    "data_id", "run_name",
)


def run_dir(run_name):
    d = Path("/kaggle/working/runs") / run_name
    for s in SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d


def fingerprint(cfg):
    """A missing key is a KeyError, never a default. Silently hashing `None` for a key that
    was renamed is exactly how a fingerprint stops protecting anything."""
    missing = [k for k in FINGERPRINT_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"fingerprint needs {missing} in CFG; add them, do not default them")
    payload = json.dumps({k: cfg[k] for k in FINGERPRINT_KEYS}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def file_sha(path):
    """Hash of the file as written. Not a re-serialisation: torch.save embeds a zip whose
    bytes are not reproducible, so only the bytes on disk are a stable identity."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
def save_round_weights(model, rnd, cfg, metrics, run_name):
    """weights -> resume. The caller writes the marker, and only after every other artifact
    of the round is on disk."""
    d = run_dir(run_name)
    core = getattr(model, "module", model)                    # unwrap DDP -> portable state
    core = getattr(core, "_orig_mod", core)                   # unwrap torch.compile
    # The hash of the weights this round was trained FROM. Two runs of the same config have
    # the same fingerprint, so without this an import can keep round 1 of run A and take
    # round 2 of run B and call the result one training history.
    prev = d / "weights" / f"round_{rnd - 1:03d}.pt"
    prev_sha = file_sha(prev) if rnd > 1 and prev.is_file() else None
    atomic_save({"round": int(rnd),
                 "model": {k: v.detach().cpu().clone() for k, v in core.state_dict().items()},
                 "cfg": {k: v for k, v in cfg.items()},       # rebuild recipe
                 "fingerprint": fingerprint(cfg),
                 "prev_sha": prev_sha,
                 "metrics": metrics},
                d / "weights" / f"round_{rnd:03d}.pt")
    # The RNG here is the DRIVER's, and the driver does not train. It is recorded for
    # forensics only: worker training RNG is re-derived from (seed, round, client) at the
    # start of every client, so continuation does not depend on restoring this.
    atomic_save({"round": int(rnd), "rng": rng_state(),
                 "note": "no optimizer state: FD-IDS clients restart from w_G each round; "
                         "worker RNG is derived from (seed, round, client)",
                 "fingerprint": fingerprint(cfg)},
                d / "resume" / f"round_{rnd:03d}.pt")
    return d / "weights" / f"round_{rnd:03d}.pt"


def mark_complete(d, rnd):
    (d / "complete" / f"round_{rnd:03d}.done").write_text("")


# --------------------------------------------------------------------------- rebuild
def load_weights(path, build, device="cpu", expect_params=None):
    """Rebuild the model at exactly this checkpoint, from weights alone.

    `build` is the run's own model factory, build(cfg) -> nn.Module. Every assert here
    exists because its failure is otherwise silent: strict=True catches a filtered
    running_mean/var (eval() would then normalize by 0/1 and report nothing), the parameter
    count catches a cfg that drifted from the one that trained these weights, and the finite
    check catches a diverged tensor that argmax would happily turn into a plausible label."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build(ck["cfg"])
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v
          for k, v in ck["model"].items()}
    bad = [k for k, v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        raise RuntimeError(f"non-finite values in checkpoint tensors {bad[:3]}")
    model.load_state_dict(sd, strict=True)
    n = sum(p.numel() for p in model.parameters())
    if expect_params is not None and n != expect_params:
        raise RuntimeError(f"rebuilt {n:,} parameters, expected {expect_params:,}")
    if ck.get("fingerprint") != fingerprint(ck["cfg"]):
        raise RuntimeError("weights file fingerprint disagrees with its own cfg")
    return model.to(device).eval(), ck    # eval(): Dropout off, BN on running stats


# --------------------------------------------------------------------------- verify
def round_ok(d, rnd, fp=None, chain=True):
    """A round counts only if every artifact of that round is present, READABLE, and links
    to the round before it.

    The marker alone proves nothing. An interrupted copy writes files in some order, so a
    marker that arrived before the metrics it vouches for is worse than no marker: it makes
    the next session skip a round it never actually has. Size is not readability either --
    a file of the right length full of garbage passed every earlier version of this check."""
    w = d / "weights" / f"round_{rnd:03d}.pt"
    r = d / "resume" / f"round_{rnd:03d}.pt"
    m = d / "metrics" / f"round_{rnd:03d}.json"
    c = d / "confusion" / f"round_{rnd:03d}.npy"
    for path in (w, r, m, c):
        if not path.is_file() or path.stat().st_size == 0:
            return False
    try:
        ck = torch.load(w, map_location="cpu", weights_only=True)
        rs = torch.load(r, map_location="cpu", weights_only=True)   # not just "it exists"
        row = json.loads(m.read_text())
        np.load(c)
    except Exception:
        return False                      # truncated or corrupt reads as a failed round
    if int(ck.get("round", -1)) != rnd or int(row.get("round", -1)) != rnd:
        return False
    if int(rs.get("round", -1)) != rnd:
        return False
    if fp is not None and (ck.get("fingerprint") != fp or rs.get("fingerprint") != fp):
        return False
    if chain:
        # Round r is only meaningful as the product of round r-1. Same config, same
        # fingerprint, different training history -> different bytes -> chain breaks here.
        prev = d / "weights" / f"round_{rnd - 1:03d}.pt"
        want = file_sha(prev) if rnd > 1 and prev.is_file() else None
        if ck.get("prev_sha") != want:
            return False
    return True


def _markers(d):
    return sorted(int(p.stem.split("_")[1]) for p in (d / "complete").glob("round_*.done"))


def last_complete_round(d, fp=None):
    """Largest r such that rounds 1..r are ALL complete. A gap ends the run: round r's
    weights are only meaningful as the product of every round before it, so the largest
    marker is not the answer when one in the middle is missing."""
    last = 0
    for r in range(1, (_markers(d)[-1] if _markers(d) else 0) + 1):
        if not (d / "complete" / f"round_{r:03d}.done").is_file() or not round_ok(d, r, fp):
            break
        last = r
    return last or None


# --------------------------------------------------------------------------- resume
def _attached_source(run_name, attached=Path("/kaggle/input")):
    if not attached.exists():
        return None
    roots = sorted({p.parent for p in attached.rglob("complete/round_*.done")
                    if run_name in p.parts})
    if len(roots) > 1:
        raise RuntimeError(f"Multiple resume trees for {run_name}: {roots}")
    return roots[0].parent if roots else None


def _import_from(src, d, fp):
    """Copy through a staging tree, verify there, then publish one round at a time with its
    marker last. Idempotent: a crash mid-publish leaves that round unmarked, and the next
    attempt re-copies it from the still-mounted source.

    Two rules the first version of this did not have. **The source's own markers bound the
    import**: a source that crashed after writing round 2's artifacts but before its marker
    has committed exactly one round, and promoting round 2 here would invent a completion
    the source never claimed. And **the destination must not be a different history**: two
    runs of the same config share a fingerprint, so without comparing the actual bytes an
    import happily keeps round 1 of one run and takes round 2 of another."""
    stage = d.parent / f".{d.name}.import"
    shutil.rmtree(stage, ignore_errors=True)
    for sub in RESUME_SUBDIRS + ("complete",):
        (stage / sub).mkdir(parents=True, exist_ok=True)
    for sub in RESUME_SUBDIRS + ("complete",):
        peer = src / sub
        if not peer.is_dir(): continue
        for f in peer.iterdir():
            if f.is_file():
                # copyfile, not copy2: a read-only mount's mode would carry across and the
                # first rewrite would die with PermissionError.
                shutil.copyfile(f, stage / sub / f.name)
                os.chmod(stage / sub / f.name, 0o644)

    # 1. What the SOURCE committed: marker AND artifacts, contiguous from round 1.
    src_last = 0
    while ((stage / "complete" / f"round_{src_last + 1:03d}.done").is_file()
           and round_ok(stage, src_last + 1, fp)):
        src_last += 1

    # 2. Refuse outright if the destination already holds a different training history.
    for r in range(1, src_last + 1):
        here = d / "weights" / f"round_{r:03d}.pt"
        if here.is_file() and file_sha(here) != file_sha(stage / "weights" / f"round_{r:03d}.pt"):
            shutil.rmtree(stage, ignore_errors=True)
            raise RuntimeError(
                f"round {r} in {d} and in {src} have the same config but different weights: "
                "these are two different training runs, not one interrupted one. Refusing to "
                "splice them. Detach one source, or start a new run_name.")

    # 3. Publish, marker last, per round.
    published = 0
    for r in range(1, src_last + 1):
        if not round_ok(d, r, fp):
            for sub in RESUME_SUBDIRS:
                for f in (stage / sub).glob(f"round_{r:03d}.*"):
                    shutil.copyfile(f, d / sub / f.name)
                    os.chmod(d / sub / f.name, 0o644)
        # Separate from the copy, and checked separately: a retry after a crash BETWEEN the
        # copy and the marker finds the artifacts already in place, and skipping the mark
        # because of that would strand the round unmarked forever.
        if not (d / "complete" / f"round_{r:03d}.done").is_file():
            mark_complete(d, r)                     # marker last, per round
        published = r
    shutil.rmtree(stage, ignore_errors=True)
    n_mark = len(list((src / "complete").glob("round_*.done"))) if (src / "complete").is_dir() else 0
    print(f"[resume] {src}: {n_mark} marker(s), {src_last} verified, imported 1..{published}"
          if published else
          f"[resume] {src} held no verifiable committed round; starting from 0")
    return published


def resolve_resume(run_name, cfg=None, attached=Path("/kaggle/input")):
    """working/ first, then any attached input (previous kernel output or a checkpoint
    dataset). Returns the last round that is complete AND verified, or None."""
    d = run_dir(run_name)
    fp = fingerprint(cfg) if cfg is not None else None
    src = _attached_source(run_name, attached)
    if src is not None:
        _import_from(src, d, fp)
    rebuild_history(d, fp)
    return last_complete_round(d, fp)


# --------------------------------------------------------------------------- history
def _write_csv(p, rows):
    tmp = Path(p).with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, p)                      # a crash mid-write cannot truncate the live file


def rebuild_history(d, fp=None):
    """history.csv is DERIVED, never authoritative. Every field of it is also in the round's
    metrics file, so a CSV lost or truncated by a crash costs nothing.

    Only the verified contiguous range 1..last goes in. Taking every marker would keep rows
    for rounds after a gap -- rounds whose weights no longer descend from anything on disk."""
    last = last_complete_round(d, fp) or 0
    rows = []
    for r in range(1, last + 1):
        p = d / "metrics" / f"round_{r:03d}.json"
        if not p.is_file(): break
        try: j = json.loads(p.read_text())
        except Exception: break
        rows.append({k: v for k, v in j.items() if k != "per_class"})
    if rows:
        _write_csv(d / "history.csv", rows)
    elif (d / "history.csv").exists():
        (d / "history.csv").unlink()      # a stale CSV outlives the rounds it described
    return len(rows)


def append_history(d, row):
    """Keyed by round: a redone round replaces its line instead of duplicating it."""
    p = d / "history.csv"
    rows = {}
    if p.exists():
        with open(p) as f:
            rows = {int(r["round"]): r for r in csv.DictReader(f)}
    rows[int(row["round"])] = {k: str(v) for k, v in row.items()}
    _write_csv(p, [rows[k] for k in sorted(rows)])

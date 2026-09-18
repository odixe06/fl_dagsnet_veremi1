"""Weights, resume state and the completion marker — three files, one atomic round.

The per-round weights file holds WEIGHTS ONLY and loads with weights_only=True: the
aggregated proxy w_bar_r and every client's personalized model w_s^k as plain state_dict
tensors. RNG and the round counter live in a separate resume bundle keyed by the same
round, so a reader never has to unpickle arbitrary objects to look at a checkpoint.

Persistent state of the method is the round, w_bar_r and the N personalized models. The
per-client proxies w_r^k are not kept: each is replaced by w_bar_r at the start of the
next round. AdamW is re-created per client per round (owner's decision, see nilm.py), so
no optimizer state exists at a round boundary.
"""
import csv, hashlib, json, os, random, shutil
from pathlib import Path
import numpy as np, torch

SUBDIRS = ("weights", "resume", "complete", "metrics", "preds", "confusion", "reports", "logs")

# What a later session imports. `preds` and `logs` are not needed to CONTINUE, but a run
# split across sessions must still be verifiable from its final output alone.
RESUME_SUBDIRS = ("weights", "resume", "metrics", "confusion", "preds", "logs")

# A run tree may start at a round r0 > 1 when the previous sessions were handed over as a
# HANDOFF BUNDLE: only round r0's artifacts, plus this file, which carries the sha256 of
# every weights file 1..r0 taken from the full tree the owner holds locally. Round r0 is
# then anchored two ways — its own bytes must hash to chain[r0], and its prev_sha must be
# chain[r0-1] — so a bundle cannot be assembled from two runs of the same config, and the
# full chain is re-verified locally after the sessions are merged (merge_sessions.py).
HANDOFF = "reports/handoff.json"

# Every input that changes what the numbers mean. `rounds` IS in the list since the
# per-round LR schedule (proj.nilm.lr_at) spans the whole run: the weights at round r
# depend on how many rounds were planned, so a continuation push must plan the same
# total. Paths, world_size, compile, eval_batch, max_seconds and require_resume are
# operational and absent.
FINGERPRINT_KEYS = (
    "patch_len", "stem_ch", "dense_growth", "dense_layers", "incep_modules",
    "fire_modules", "dropout", "num_classes", "n_features",
    "lr", "lr_schedule", "lr_min", "rounds", "weight_decay", "clip",
    "n_clients", "batch", "local_epochs", "seed",
    "data_id", "run_name",
)


def run_dir(run_name, base="/kaggle/working/runs"):
    d = Path(base) / run_name
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


def atomic_np_save(path, arr):
    """np.save appends .npy to a name that lacks it, so write through a handle."""
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


# --- RNG kept as tensors and primitives so the resume bundle also loads weights_only=True.
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


def cpu_sd(model):
    """A detached CPU copy of a module's state_dict: tensors only, so the file it goes
    into loads with weights_only=True."""
    core = getattr(model, "_orig_mod", model)                 # unwrap torch.compile
    return {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}


# --------------------------------------------------------------------------- write
def save_round_weights(global_sd, clients_sd, rnd, cfg, metrics, global_metrics, d):
    """global_sd: state_dict of w_bar_r^t; clients_sd: {cid: state_dict of w_s^k at
    round t} for ALL N clients. `metrics` is the mean over clients of the 10 metrics,
    `global_metrics` the 10 metrics of w_bar_r. The caller writes the marker, and only
    after every other artifact is on disk."""
    prev = d / "weights" / f"round_{rnd - 1:03d}.pt"
    # The hash of the weights this round was trained FROM. Two runs of the same config have
    # the same fingerprint, so without this an import can keep round 1 of run A and take
    # round 2 of run B and call the result one training history.
    prev_sha = file_sha(prev) if rnd > 1 and prev.is_file() else None
    atomic_save({"round": int(rnd),
                 "global": global_sd,
                 "clients": {int(c): sd for c, sd in sorted(clients_sd.items())},
                 "cfg": {k: v for k, v in cfg.items()},       # rebuild recipe
                 "fingerprint": fingerprint(cfg),
                 "prev_sha": prev_sha,
                 "metrics": metrics,
                 "global_metrics": global_metrics},
                d / "weights" / f"round_{rnd:03d}.pt")
    # The RNG here is the DRIVER's, and the driver does not train. Recorded for forensics:
    # worker training RNG is re-derived from (seed, round, client), so continuation does
    # not depend on restoring this.
    atomic_save({"round": int(rnd), "rng": rng_state(),
                 "note": "no optimizer state: AdamW is re-created per client per round; "
                         "worker RNG derives from (seed, round, client)",
                 "fingerprint": fingerprint(cfg)},
                d / "resume" / f"round_{rnd:03d}.pt")
    return d / "weights" / f"round_{rnd:03d}.pt"


def mark_complete(d, rnd):
    (d / "complete" / f"round_{rnd:03d}.done").write_text("")


# --------------------------------------------------------------------------- rebuild
def _load_into(model, sd, expect_params):
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    bad = [k for k, v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        raise RuntimeError(f"non-finite values in checkpoint tensors {bad[:3]}")
    model.load_state_dict(sd, strict=True)
    n = sum(p.numel() for p in model.parameters())
    if expect_params is not None and n != expect_params:
        raise RuntimeError(f"rebuilt {n:,} parameters, expected {expect_params:,}")
    return model.eval()


def load_weights(path, build_model, expect_params=None, clients=None, device="cpu"):
    """Rebuild w_bar_r and the personalized models at exactly this checkpoint, from
    weights alone. Returns (R, {cid: S_k}, raw dict).

    Every assert here exists because its failure is otherwise silent: strict=True catches a
    filtered running_mean/var (eval() would then normalize by 0/1 and report nothing), the
    parameter count catches a cfg that drifted from the one that trained these weights, and
    the finite check catches a diverged tensor that argmax would turn into a plausible label.
    `clients=None` rebuilds every client; pass a list to rebuild a subset."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("fingerprint") != fingerprint(ck["cfg"]):
        raise RuntimeError("weights file fingerprint disagrees with its own cfg")
    R = _load_into(build_model(ck["cfg"]), ck["global"], expect_params).to(device)
    want = sorted(ck["clients"]) if clients is None else list(clients)
    if sorted(ck["clients"]) != list(range(ck["cfg"]["n_clients"])):
        raise RuntimeError(f"checkpoint holds clients {sorted(ck['clients'])[:5]}..., "
                           f"expected 0..{ck['cfg']['n_clients'] - 1}")
    Ss = {int(c): _load_into(build_model(ck["cfg"]), ck["clients"][c], expect_params).to(device)
          for c in want}
    return R, Ss, ck


# --------------------------------------------------------------------------- verify
def read_handoff(d):
    """The handoff record of a tree that starts past round 1, or None. A malformed file
    reads as absent, and an absent record means rounds before the first marker are simply
    missing — which last_complete_round then treats as a gap, never as a start."""
    p = Path(d) / HANDOFF
    if not p.is_file():
        return None
    try:
        h = json.loads(p.read_text())
        h["round"] = int(h["round"])
        h["chain"] = {int(k): str(v) for k, v in h["chain"].items()}
        return h
    except Exception:
        return None


def handoff_record(d, rnd, fp, extra=None):
    """The sha chain 1..rnd of a FULL tree, for a bundle that will carry round rnd alone.
    Refuses a tree whose chain does not verify all the way from round 1. The caller writes
    the dict to HANDOFF inside the bundle, never inside the full tree."""
    d = Path(d)
    if _first_round(d) != 1:
        raise RuntimeError(f"{d} is itself a partial tree; export a handoff from the merged run")
    last = last_complete_round(d, fp)
    if last is None or last < rnd:
        raise RuntimeError(f"{d}: rounds 1..{rnd} do not all verify (last={last}); "
                           "nothing to hand off")
    chain = {str(r): file_sha(d / "weights" / f"round_{r:03d}.pt") for r in range(1, rnd + 1)}
    return {"round": int(rnd), "fingerprint": fp, "chain": chain, **(extra or {})}


def round_ok(d, rnd, fp=None, chain=True):
    """A round counts only if every artifact of that round is present, READABLE, and links
    to the round before it. The marker alone proves nothing. When the round before it is
    not on disk, the link is checked against the handoff chain instead (see HANDOFF), and
    the round's own bytes must match the chain too."""
    w = d / "weights" / f"round_{rnd:03d}.pt"
    r = d / "resume" / f"round_{rnd:03d}.pt"
    m = d / "metrics" / f"round_{rnd:03d}.json"
    c = d / "confusion" / f"round_{rnd:03d}.npy"
    g = d / "confusion" / f"global_{rnd:03d}.npy"
    for path in (w, r, m, c, g):
        if not path.is_file() or path.stat().st_size == 0:
            return False
    try:
        ck = torch.load(w, map_location="cpu", weights_only=True, mmap=True)
        rs = torch.load(r, map_location="cpu", weights_only=True)   # not just "it exists"
        row = json.loads(m.read_text())
        np.load(c); np.load(g)
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
        if rnd > 1 and not prev.is_file():
            h = read_handoff(d)
            if h is None or h["round"] != rnd or (fp is not None and h["fingerprint"] != fp):
                return False                  # a bare round r > 1 is a gap, not a start
            if h["chain"].get(rnd) != file_sha(w) or ck.get("prev_sha") != h["chain"].get(rnd - 1):
                return False
        else:
            want = file_sha(prev) if rnd > 1 and prev.is_file() else None
            if ck.get("prev_sha") != want:
                return False
    return True


def _markers(d):
    return sorted(int(p.stem.split("_")[1]) for p in (d / "complete").glob("round_*.done"))


def _first_round(d):
    """1 for a full tree; the handoff round for a bundle that starts later; the lowest
    marker otherwise (round_ok then rejects it as a gap unless a handoff anchors it)."""
    h = read_handoff(d)
    m = _markers(d)
    if h is not None and not (d / "weights" / f"round_{h['round'] - 1:03d}.pt").is_file():
        return h["round"]
    return m[0] if m else 1


def last_complete_round(d, fp=None):
    """Largest r such that rounds first..r are ALL complete, where first is 1 or the
    handoff round of a bundle. A gap ends the run."""
    last = 0
    m = _markers(d)
    top = m[-1] if m else 0
    for r in range(_first_round(d), top + 1):
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
    attempt re-copies it from the still-mounted source. The source's own markers bound the
    import, and the destination must not already hold a different history."""
    stage = d.parent / f".{d.name}.import"
    shutil.rmtree(stage, ignore_errors=True)
    for sub in RESUME_SUBDIRS + ("complete", "reports"):
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
    if (src / HANDOFF).is_file():
        shutil.copyfile(src / HANDOFF, stage / HANDOFF)
        os.chmod(stage / HANDOFF, 0o644)

    first = _first_round(stage)
    src_last = first - 1
    while ((stage / "complete" / f"round_{src_last + 1:03d}.done").is_file()
           and round_ok(stage, src_last + 1, fp)):
        src_last += 1
    if src_last < first:
        src_last = 0                                   # nothing verifiable, not even round first

    if src_last and first > 1:
        # The destination must not hold rounds the bundle cannot vouch for.
        if any((d / "weights" / f"round_{r:03d}.pt").is_file() for r in range(1, first)):
            shutil.rmtree(stage, ignore_errors=True)
            raise RuntimeError(f"{d} already holds rounds before {first}; a handoff bundle "
                               "cannot be spliced onto a tree that has its own history")
        shutil.copyfile(stage / HANDOFF, d / HANDOFF)
    for r in range(first, src_last + 1):
        here = d / "weights" / f"round_{r:03d}.pt"
        if here.is_file() and file_sha(here) != file_sha(stage / "weights" / f"round_{r:03d}.pt"):
            shutil.rmtree(stage, ignore_errors=True)
            raise RuntimeError(
                f"round {r} in {d} and in {src} have the same config but different weights: "
                "these are two different training runs, not one interrupted one. Refusing to "
                "splice them. Detach one source, or start a new run_name.")

    published = 0
    for r in range(first, src_last + 1):
        if not round_ok(d, r, fp):
            for sub in RESUME_SUBDIRS:
                for f in list((stage / sub).glob(f"round_{r:03d}.*")) + \
                         list((stage / sub).glob(f"global_{r:03d}.*")):
                    shutil.copyfile(f, d / sub / f.name)
                    os.chmod(d / sub / f.name, 0o644)
        if not (d / "complete" / f"round_{r:03d}.done").is_file():
            mark_complete(d, r)                     # marker last, per round
        published = r
    shutil.rmtree(stage, ignore_errors=True)
    n_mark = len(list((src / "complete").glob("round_*.done"))) if (src / "complete").is_dir() else 0
    print(f"[resume] {src}: {n_mark} marker(s), verified to round {src_last}, imported {first}..{published}"
          + (f" (handoff bundle: chain 1..{first} attested, verified locally before export)"
             if first > 1 else "")
          if published else
          f"[resume] {src} held no verifiable committed round; starting from 0")
    return published


def resolve_resume(run_name, cfg=None, attached=Path("/kaggle/input"), d=None):
    """working/ first, then any attached input (previous kernel output or a checkpoint
    dataset). Returns the last round that is complete AND verified, or None."""
    d = run_dir(run_name) if d is None else d
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
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)                      # a crash mid-write cannot truncate the live file


def rebuild_history(d, fp=None):
    """history.csv (mean over clients + global proxy per round) and clients.csv (one row
    per client per round) are DERIVED from metrics/round_NNN.json, never authoritative.
    Only the verified contiguous range 1..last goes in."""
    last = last_complete_round(d, fp) or 0
    rows, crow = [], []
    for r in range(_first_round(d), last + 1):
        p = d / "metrics" / f"round_{r:03d}.json"
        if not p.is_file(): break
        try: j = json.loads(p.read_text())
        except Exception: break
        rows.append({k: v for k, v in j.items() if k not in ("clients", "global")})
        crow += [{"round": r, **{k: v for k, v in c.items() if k != "per_class"}}
                 for c in j["clients"]]
    if rows:
        _write_csv(d / "history.csv", rows)
        _write_csv(d / "clients.csv", crow)
    else:
        for n in ("history.csv", "clients.csv"):
            if (d / n).exists(): (d / n).unlink()  # a stale CSV outlives the rounds it described
    return len(rows)


def append_history(d, row, client_rows):
    """Keyed by round: a redone round replaces its lines instead of duplicating them."""
    p = d / "history.csv"
    rows = {}
    if p.exists():
        with open(p) as f:
            rows = {int(r["round"]): r for r in csv.DictReader(f)}
    rows[int(row["round"])] = {k: str(v) for k, v in row.items()}
    _write_csv(p, [rows[k] for k in sorted(rows)])
    q = d / "clients.csv"
    old = []
    if q.exists():
        with open(q) as f:
            old = [r for r in csv.DictReader(f) if int(r["round"]) != int(row["round"])]
    _write_csv(q, old + [{k: str(v) for k, v in c.items()} for c in client_rows])

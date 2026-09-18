"""AFPHA-DAGSNet federated training driver.

One round: broadcast the global weights, train every client for one full local
epoch on its own GPU-resident rows, aggregate cluster-then-server by sample count,
evaluate the new global model on the entire fixed test set, commit the artifacts.

Nothing here uses the test set to steer training.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import queue
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import afpha
import fldata
import flatpack
from dagsnet import CFG as MODEL_CFG, N_FEATURES, N_PARAMS, build_dagsnet
from metrics import METRIC_KEYS, metrics_from_confusion, per_class_report
from worker import worker_main

TOTAL_ROUNDS = 50


# ------------------------------------------------------------------------- config

def scenario_config(num_clients):
    return {"num_clients": num_clients,
            "batch": 512 if num_clients in (20, 50) else 256,
            "run_name": f"afpha-dagsnet-{num_clients}client"}


def resolve_paths(num_clients, args):
    """Resolve the partition root and the test root by sentinel, not by mount prefix."""
    train_root = fldata.find_root(f"{num_clients}_client/client_stats.json",
                                  args.data_search) / f"{num_clients}_client"
    test_base = fldata.find_root("test/part-00000.parquet", args.test_search)
    return train_root, test_base


# --------------------------------------------------------------------- artifacts

class Run:
    """Output tree. A round is only 'complete' once its marker exists."""

    def __init__(self, root):
        self.root = Path(root)
        for sub in ("weights", "metrics", "confusion", "per_class", "preds",
                    "client_log", "complete", "model_source", "resume"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.history = self.root / "metrics" / "history.csv"
        self.heartbeat = self.root / "heartbeat.jsonl"

    def last_complete_round(self):
        done = sorted(int(p.stem.split("_")[1])
                      for p in (self.root / "complete").glob("round_*.done"))
        return done[-1] if done else 0

    def emit(self, record, wb=None, payload=None, step=None):
        with self.heartbeat.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print("  " + json.dumps(record), flush=True)
        if wb is not None and payload is not None:
            wb.log(payload, step=step)

    def history_row_from_json(self, rnd):
        """Rebuild one history row from the round's committed metrics JSON.

        The CSV is derived data: every column comes from `metrics/round_NNN.json`, which
        is written inside the same commit. That makes the CSV reconstructible, which is
        what lets `append_history` heal a truncated file instead of propagating the loss.
        """
        d = json.loads((self.root / "metrics" / f"round_{rnd:03d}.json").read_text())
        return {"round": str(rnd),
                **{k: f"{float(d[k]):.6f}" for k in METRIC_KEYS},
                **{f"t_{k}": f"{float(v):.1f}" for k, v in d["timing"].items()}}

    def append_history(self, row):
        """Idempotent: a re-run round replaces its row instead of duplicating it.

        Two failure modes are handled here, both observed:

        * The whole file is rewritten on every round, so a crash partway through used to
          leave a CSV holding only the rounds written before the interruption -- committed
          history was lost even though the markers and the per-round JSON survived. The
          write now goes to a temporary file and is published with `os.replace`, which is
          atomic on POSIX: a reader sees either the old file or the new one.
        * A CSV truncated by an *earlier* crash would otherwise stay truncated forever.
          Before writing, every round whose marker exists is restored from its JSON, so
          the next commit repairs the file rather than carrying the gap forward.
        """
        rows = {}
        if self.history.exists():
            with self.history.open() as fh:
                for r in csv.DictReader(fh):
                    rows[int(r["round"])] = r
        for rnd in sorted(int(p.stem.split("_")[1])
                          for p in (self.root / "complete").glob("round_*.done")):
            if rnd not in rows:
                try:
                    rows[rnd] = self.history_row_from_json(rnd)
                except (OSError, KeyError, ValueError, json.JSONDecodeError):
                    pass          # unreadable JSON is a separate problem; do not mask it
        rows[int(row["round"])] = {k: str(v) for k, v in row.items()}
        fields = list(row)
        tmp = self.history.with_name(self.history.name + ".tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for k in sorted(rows):
                w.writerow({f: rows[k].get(f, "") for f in fields})
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.history)

    def commit(self, rnd, global_sd, cm, preds, client_stats, cluster_rows, mu_next, drifts,
               timing, schema, resume_state):
        """Write everything, then the marker. A crash mid-write leaves the round incomplete.

        The resume state is per-round and written before the marker, so the state a
        resume reads always belongs to the round the marker certifies. A single latest
        resume_state.pt cannot give that guarantee: a crash between the marker write
        and the state write leaves the two describing different rounds.
        """
        m = metrics_from_confusion(cm)
        torch.save({k: v for k, v in global_sd.items()},
                   self.root / "weights" / f"round_{rnd:03d}.pt")
        np.save(self.root / "confusion" / f"round_{rnd:03d}.npy", cm)
        np.save(self.root / "preds" / f"round_{rnd:03d}_ypred.npy", preds)
        (self.root / "per_class" / f"round_{rnd:03d}.json").write_text(
            json.dumps(per_class_report(cm, schema.class_names), indent=2))
        (self.root / "metrics" / f"round_{rnd:03d}.json").write_text(
            json.dumps({"round": rnd, **m, "timing": timing,
                        "test_rows": int(cm.sum())}, indent=2))
        (self.root / "client_log" / f"round_{rnd:03d}.json").write_text(
            json.dumps({"round": rnd, "clients": client_stats, "cluster_rows": cluster_rows,
                        "mu_next": mu_next, "drift": drifts}, indent=2))
        self.append_history({"round": rnd, **{k: f"{m[k]:.6f}" for k in METRIC_KEYS},
                             **{f"t_{k}": f"{v:.1f}" for k, v in timing.items()}})
        torch.save(resume_state, self.root / "resume" / f"round_{rnd:03d}.pt")
        (self.root / "complete" / f"round_{rnd:03d}.done").write_text(
            json.dumps({"round": rnd, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        return m


def import_previous(run, search_roots, fingerprint, log, explicit=None):
    """Copy a prior session's run directory into this session's empty working dir.

    A new Kaggle session starts with an empty /kaggle/working, so the previous run
    reaches it read-only through `kernel_sources` under /kaggle/input. Without this
    step `last_complete_round()` sees nothing and round 1 starts over. Only a run whose
    config.json carries the same fingerprint is eligible: a different scientific setting
    must not be silently continued.
    """
    if run.last_complete_round():
        return 0
    found = []
    for root in search_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for cfg_path in sorted(root.glob("**/config.json")):
            src = cfg_path.parent
            if src.resolve() == run.root.resolve():
                continue
            try:
                if json.loads(cfg_path.read_text()).get("fingerprint") != fingerprint:
                    continue
                done = sorted(int(q.stem.split("_")[1])
                              for q in (src / "complete").glob("round_*.done"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if done:
                found.append((src, done[-1]))
    if explicit is not None:
        want = Path(explicit).resolve()
        found = [(s_, n) for s_, n in found if s_.resolve() == want]
        assert found, f"--resume-source {explicit} holds no completed round with fingerprint {fingerprint}"
    if not found:
        return 0
    # The fingerprint identifies the *configuration*, not the run. Two independent
    # trainings of the same scenario share it while holding different weights, so
    # picking the one with the most rounds would silently splice two histories
    # together. Ambiguity is the user's to resolve.
    if len(found) > 1:
        listing = "\n  ".join(f"{s_} ({n} rounds)" for s_, n in sorted(found))
        raise SystemExit(
            f"{len(found)} candidate resume sources share fingerprint {fingerprint}:\n"
            f"  {listing}\n"
            "Refusing to guess which run to continue. Pass --resume-source <dir> to choose.")
    src, last = found[0]
    log(f"importing {last} completed rounds from {src}")
    # Markers are copied last and only after the bundle validates. A marker is the claim
    # 'this round is complete and its artifacts are here'; publishing it while the copy is
    # still in flight makes an interrupted import look like a finished one, and the retry
    # then returns immediately with weights still missing.
    markers = []
    for item in sorted(src.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(src)
        if rel.parts and rel.parts[0] == "complete":
            markers.append((item, rel))
            continue
        dst = run.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        # A same-size file is treated as already imported; a different size means the
        # previous attempt was cut off mid-file, so redo it rather than skipping it.
        if dst.exists() and dst.stat().st_size == item.stat().st_size:
            continue
        # copyfile, not copy2: the source is a read-only mount (/kaggle/input), and
        # copy2 would carry r--r--r-- across. config.json and history.csv are
        # rewritten every round, so the next write would fail with PermissionError.
        part = dst.with_name(dst.name + ".part")
        shutil.copyfile(item, part)
        part.chmod(0o644)
        os.replace(part, dst)
    missing = bundle_gaps(run.root, last)
    assert not missing, (
        f"import from {src} is incomplete; refusing to publish completion markers. "
        f"missing: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    for item, rel in sorted(markers):
        dst = run.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        shutil.copyfile(item, part)
        part.chmod(0o644)
        os.replace(part, dst)
    got = run.last_complete_round()
    assert got == last, f"imported {last} rounds but only {got} are complete after copy"
    return got


ROUND_ARTIFACTS = (("weights", "pt"), ("metrics", "json"), ("confusion", "npy"),
                   ("per_class", "json"), ("client_log", "json"), ("resume", "pt"))


def bundle_gaps(root, last):
    """Every file a round is supposed to have, for rounds 1..last. Empty list == complete."""
    root = Path(root)
    missing = []
    for name in ("config.json", "preds/y_true.npy"):
        if not (root / name).exists():
            missing.append(name)
    for rnd in range(1, last + 1):
        for sub, ext in ROUND_ARTIFACTS:
            p = root / sub / f"round_{rnd:03d}.{ext}"
            if not p.exists() or p.stat().st_size == 0:
                missing.append(f"{sub}/round_{rnd:03d}.{ext}")
        p = root / "preds" / f"round_{rnd:03d}_ypred.npy"
        if not p.exists() or p.stat().st_size == 0:
            missing.append(f"preds/round_{rnd:03d}_ypred.npy")
    return missing


# ------------------------------------------------------------------------ prepack

def prepack(train_root, test_base, schema, scratch, rows, n_test, digests, log):
    """Parquet -> contiguous .npy once per session; workers mmap these.

    The cache key covers the schema, both roots and the row counts. Keying on the train
    root alone would reuse a stale test array after the test source changed.
    """
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    paths = {k: str(scratch / f"{k}.npy")
             for k in ("train_X", "train_y", "test_X", "test_y")}
    # The digests are what make a rewritten source at the same path a cache miss; row
    # counts and paths alone cannot see a file whose values changed but whose shape did not.
    key = {"schema": schema.fingerprint(), "train_root": str(train_root),
           "test_root": str(test_base), "rows": rows.tolist(), "n_test": int(n_test),
           "digests": digests}
    meta_path = scratch / "prepack.json"
    if meta_path.exists() and all(Path(p).exists() for p in paths.values()):
        info = json.loads(meta_path.read_text())
        if info.get("key") == key:
            log("  prepack cache hit")
            return paths, info["spans"]

    t0 = time.perf_counter()
    log("  reading train partitions")
    X, y, spans, got_rows = fldata.load_clients(train_root / "train", schema, log=log)
    assert got_rows.tolist() == rows.tolist(), "row counts changed between scan and decode"
    np.save(paths["train_X"], X); np.save(paths["train_y"], y)
    del X, y
    t_train = time.perf_counter() - t0
    log("  reading test")
    t1 = time.perf_counter()
    Xt, yt = fldata.load_test(test_base / "test", schema, log=log)
    assert len(yt) == n_test, f"test rows changed between scan ({n_test}) and decode ({len(yt)})"
    np.save(paths["test_X"], Xt); np.save(paths["test_y"], yt)
    del Xt, yt
    meta_path.write_text(json.dumps({"key": key, "spans": spans,
                                     "train_decode_s": t_train,
                                     "test_decode_s": time.perf_counter() - t1,
                                     "seconds": time.perf_counter() - t0}))
    log(f"  prepack done in {time.perf_counter() - t0:.0f}s "
        f"(train {t_train:.0f}s, test {time.perf_counter() - t1:.0f}s)")
    return paths, spans


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, required=True, choices=(20, 50, 100))
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--arch", required=True, help="dir holding meta.json and scaler.json")
    ap.add_argument("--data-search", nargs="+", required=True)
    ap.add_argument("--test-search", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=TOTAL_ROUNDS)
    ap.add_argument("--eval-batch", type=int, default=16384)
    ap.add_argument("--compile-mode", default="reduce-overhead",
                    choices=("reduce-overhead", "default", "eager"))
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--heartbeat-clients", type=int, default=5)
    ap.add_argument("--resume-search", nargs="*", default=[],
                    help="Dirs holding a previous session's run output (Kaggle: /kaggle/input). "
                         "Only a run with a matching fingerprint is imported.")
    ap.add_argument("--resume-source", default=None,
                    help="Import from exactly this run directory. Required when several "
                         "candidates under --resume-search share the fingerprint.")
    ap.add_argument("--require-resume", action="store_true",
                    help="Fail before prepack if no matching previous run is found.")
    ap.add_argument("--max-skip-pct", type=float, default=5.0,
                    help="Abort the round if a client skipped more than this share of its AMP "
                         "steps, beyond the --max-skip-abs warm-up allowance.")
    ap.add_argument("--max-skip-abs", type=int, default=8,
                    help="Skips always tolerated per client per round. The GradScaler is reset "
                         "for every client, so each one walks its scale down from 65536 and "
                         "skips a few steps doing so; that is calibration, not divergence. "
                         "Production clients run 384-11,506 steps, so this is well under 1%.")
    ap.add_argument("--worker-devices", type=int, nargs="+", default=None,
                    help="CUDA indices, one worker each. Default: one per GPU, capped at 2. "
                         "Repeat an index to exercise the two-worker protocol on one GPU.")
    ap.add_argument("--wandb-project", default="afpha-dagsnet-veremi")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    t_start = time.perf_counter()
    cfg = scenario_config(args.clients)
    run = Run(args.out)
    log = lambda m: print(m, flush=True)                                   # noqa: E731

    ngpu = torch.cuda.device_count()
    assert ngpu >= 1, "no CUDA device"
    log(f"CUDA devices: {[torch.cuda.get_device_name(i) for i in range(ngpu)]}")

    arch = Path(args.arch)
    schema = fldata.Schema(arch / "meta.json", arch / "scaler.json")
    train_root, test_base = resolve_paths(args.clients, args)
    log(f"train root: {train_root}\ntest root : {test_base}")

    # ---- W&B first: a missing secret must cost seconds, not the whole prepack.
    wb = None
    if not args.no_wandb:
        import wandb
        wb = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                        id=cfg["run_name"], name=cfg["run_name"], resume="allow",
                        mode="online", dir=str(run.root),
                        config={**cfg, "rounds": args.rounds, "model": MODEL_CFG,
                                "n_params": N_PARAMS, "spec": "rebuild.md Proposal A",
                                "cluster_size": afpha.CLUSTER_SIZE, "mu_base": afpha.MU_BASE,
                                "lr_max": afpha.LR_MAX, "lr_min": afpha.LR_MIN,
                                "grad_clip": afpha.GRAD_CLIP, "eval_batch": args.eval_batch,
                                "compile_mode_requested": args.compile_mode})
        wb.define_metric("progress/global_step")
        wb.define_metric("*", step_metric="progress/global_step")
        (run.root / "wandb_run.json").write_text(json.dumps(
            {"entity": wb.entity, "project": wb.project, "run_id": wb.id,
             "run_path": f"{wb.entity}/{wb.project}/{wb.id}", "url": wb.url}, indent=2))
        log(f"W&B: {wb.url}")

    # Footers only: cheap enough to run before prepack, which is what lets
    # --require-resume fail in seconds instead of after a multi-minute decode.
    rows, n_test, digests = fldata.scan_counts(train_root / "train", test_base / "test")
    assert len(rows) == args.clients, f"{len(rows)} client partitions for {args.clients} clients"
    total_rows = int(rows.sum())
    steps_round = fldata.steps_per_round(rows, cfg["batch"])
    log(f"{args.clients} clients, {total_rows:,} rows, {n_test:,} test rows, "
        f"{steps_round:,} optimizer steps/round")

    clusters = afpha.build_clusters(args.clients)
    (run.root / "clusters.json").write_text(json.dumps(clusters, indent=2))

    # reconstruction package: weights alone do not rebuild a model
    shutil.copy(Path(__file__).parent / "dagsnet.py", run.root / "model_source" / "dagsnet.py")
    for f in ("meta.json", "scaler.json"):
        shutil.copy(arch / f, run.root / "model_source" / f)
    (run.root / "model_source" / "model_config.json").write_text(json.dumps(
        {"cfg": MODEL_CFG, "n_features": N_FEATURES, "n_params": N_PARAMS,
         "load": "build_dagsnet(); load_state_dict(torch.load(w, weights_only=True), strict=True)"},
        indent=2))

    ref = build_dagsnet()
    template = flatpack.Template(ref)
    # Everything that changes what the run means. Leaving a constant out means a resume
    # can silently continue under different mathematics.
    fingerprint = hashlib.sha256(json.dumps(
        {"model": MODEL_CFG, "schema": schema.fingerprint(), "clients": args.clients,
         "batch": cfg["batch"], "rounds": args.rounds, "clusters": clusters,
         "spec": "proposal-A",
         "mu_base": afpha.MU_BASE, "mu_eps": afpha.MU_EPS,
         "lr_max": afpha.LR_MAX, "lr_min": afpha.LR_MIN, "clip": afpha.GRAD_CLIP,
         "adam": [list(afpha.ADAM_BETAS), afpha.ADAM_EPS, afpha.ADAM_WEIGHT_DECAY],
         "cluster_size": afpha.CLUSTER_SIZE, "cluster_seed": afpha.CLUSTER_SEED,
         "shuffle_seed": afpha.SHUFFLE_SEED,
         "rows_per_client": rows.tolist(), "n_test": n_test,
         # Content identity, not just shape: without it a resume can continue onto data
         # that was rewritten in place while keeping the same row counts.
         "data_digests": digests,
         }, sort_keys=True).encode()).hexdigest()[:16]

    # ---- resume: import a prior session's output first, then read the marker
    imported = import_previous(run, args.resume_search, fingerprint, log,
                               explicit=args.resume_source)
    start_round = run.last_complete_round()
    if args.require_resume:
        assert start_round, (
            "--require-resume was set but no completed round with fingerprint "
            f"{fingerprint} was found under {list(args.resume_search)}")
    mus = afpha.initial_mu(args.clients)
    if start_round:
        # mu applied in round start_round+1 was derived from round start_round's drift
        state = torch.load(run.root / "resume" / f"round_{start_round:03d}.pt",
                           weights_only=True)
        assert state["round"] == start_round, \
            f"resume state round {state['round']} != last complete {start_round}"
        assert state["fingerprint"] == fingerprint, "fingerprint changed; refusing to resume"
        mus = {int(k): float(v) for k, v in state["mu"].items()}
        gf, gi = template.flatten(torch.load(
            run.root / "weights" / f"round_{start_round:03d}.pt", weights_only=True))
        log(f"resuming after round {start_round}")
    else:
        torch.manual_seed(afpha.CLUSTER_SEED)
        gf, gi = template.flatten(build_dagsnet().state_dict())
        if run.history.exists():
            run.history.unlink()
    paths, spans = prepack(train_root, test_base, schema, args.scratch, rows,
                           n_test, digests, log)
    assert len(spans) == args.clients

    (run.root / "config.json").write_text(json.dumps(
        {**cfg, "rounds": args.rounds, "fingerprint": fingerprint,
         "train_root": str(train_root), "test_root": str(test_base),
         "rows_per_client": rows.tolist(), "steps_per_round": steps_round,
         "n_test": n_test, "eval_batch": args.eval_batch, "imported_rounds": imported,
         "max_skip_pct": args.max_skip_pct,
         "compile_mode_requested": args.compile_mode, "max_hours": args.max_hours,
         "schema_fingerprint": schema.fingerprint(), "clusters": clusters,
         "data_digests": digests}, indent=2))

    if start_round == 0:
        np.save(run.root / "preds" / "y_true.npy", np.load(paths["test_y"]))

    # ---- workers
    ctx = mp.get_context("spawn")
    devices = args.worker_devices or list(range(min(ngpu, 2)))
    nw = len(devices)
    bounds = [round(i * n_test / nw) for i in range(nw + 1)]
    wcfg = {"batch": cfg["batch"], "compile_mode": args.compile_mode,
            "num_classes": schema.num_classes, "eval_batch": args.eval_batch,
            "spans": spans}
    task_q, res_q = ctx.Queue(), ctx.Queue()
    ctrl = [ctx.Queue() for _ in range(nw)]
    procs = [ctx.Process(target=worker_main,
                         args=(r, devices[r], wcfg, paths, (bounds[r], bounds[r + 1]),
                               ctrl[r], task_q, res_q),
                         daemon=True) for r in range(nw)]
    for p in procs:
        p.start()

    ready = {}
    while len(ready) < nw:
        kind, rank, _, info = _get(res_q, procs)
        assert kind == "ready", f"worker {rank} failed before ready: {info}"
        ready[rank] = info
    log(f"workers ready: {json.dumps(ready)}")
    if wb:
        wb.config.update({"workers": ready}, allow_val_change=True)

    order = sorted(range(args.clients), key=lambda c: -int(rows[c]))    # longest-first
    global_step = start_round * steps_round
    deadline = t_start + args.max_hours * 3600
    worst_round = 0.0

    try:
        for rnd in range(start_round + 1, args.rounds + 1):
            t_round = time.perf_counter()
            lr = afpha.lr_for_round(rnd, args.rounds)
            for q in ctrl:
                q.put(("round", rnd, lr, mus, gf, gi))
            for cid in order:
                task_q.put(cid)
            for _ in range(nw):
                task_q.put(None)

            states, stats, done, seen = {}, {}, 0, 0
            while done < nw:
                kind, rank, payload, info = _get(res_q, procs)
                if kind == "round_done":
                    done += 1
                    continue
                assert kind == "client", f"unexpected {kind} during training: {info}"
                _check_client(info, args.max_skip_pct, args.max_skip_abs)
                states[info["client_id"]] = payload            # (float vec, int vec)
                stats[info["client_id"]] = info
                seen += 1
                global_step += info["steps"]
                if seen % args.heartbeat_clients == 0 or seen == args.clients:
                    rec = {"round": rnd, "clients_done": seen, "of": args.clients,
                           "lr": lr, "ce_mean": info["ce_mean"],
                           "grad_norm": info["grad_norm_mean"], "skipped": info["skipped"],
                           "elapsed_s": round(time.perf_counter() - t_round, 1)}
                    run.emit(rec, wb, {"progress/round": rnd, "progress/global_step": global_step,
                                       "progress/clients_done": seen,
                                       "train/loss": info["ce_mean"],
                                       "train/grad_norm": info["grad_norm_mean"],
                                       "train/lr": lr,
                                       "train/global_batch": cfg["batch"],
                                       "train/skip_pct": 100.0 * info["skipped"]
                                       / max(info["steps"], 1)}, global_step)
            assert len(states) == args.clients, f"only {len(states)} clients returned"
            t_train = time.perf_counter() - t_round

            t_agg = time.perf_counter()
            client_rows = {c: stats[c]["rows"] for c in stats}
            gf, gi, cluster_rows = afpha.hierarchical_aggregate(
                {c: v[0] for c, v in states.items()}, {c: v[1] for c, v in states.items()},
                client_rows, clusters)
            drifts = {c: afpha.param_drift(states[c][0], gf, template.n_params) for c in stats}
            mu_used = {c: stats[c]["mu"] for c in stats}
            mus = afpha.next_mu(drifts, client_rows)          # applies from the NEXT round
            t_agg = time.perf_counter() - t_agg

            t_eval = time.perf_counter()
            for q in ctrl:
                q.put(("eval", rnd, gf, gi))
            cm = np.zeros((schema.num_classes, schema.num_classes), dtype=np.int64)
            shard_preds = {}
            got = 0
            while got < 2 * nw:
                kind, rank, payload, info = _get(res_q, procs)
                if kind == "eval":
                    cm += payload
                elif kind == "preds":
                    shard_preds[rank] = payload
                else:
                    raise AssertionError(f"unexpected {kind} during eval: {info}")
                got += 1
            preds = np.concatenate([shard_preds[r] for r in range(nw)])
            assert len(preds) == n_test and int(cm.sum()) == n_test, \
                f"test coverage {len(preds)}/{int(cm.sum())} != {n_test}"
            t_eval = time.perf_counter() - t_eval

            timing = {"train_s": t_train, "aggregate_s": t_agg, "eval_s": t_eval,
                      "round_s": time.perf_counter() - t_round}   # train+aggregate+eval
            m = run.commit(rnd, template.unflatten(gf, gi), cm, preds,
                           [stats[c] for c in sorted(stats)], cluster_rows,
                           {str(k): v for k, v in mus.items()},
                           {str(k): v for k, v in drifts.items()}, timing, schema,
                           resume_state={"round": rnd, "fingerprint": fingerprint,
                                         "mu": {str(k): float(v) for k, v in mus.items()},
                                         "global_step": int(global_step)})

            # Budget on the worst round seen, not the last one: round 1 pays compile and
            # is slow, later rounds vary, and an optimistic estimate is the one that gets
            # the session hard-killed mid-commit. Unlike timing["round_s"], this includes
            # the commit that follows evaluation.
            commit_s = time.perf_counter() - t_round - timing["round_s"]
            worst_round = max(worst_round, time.perf_counter() - t_round)

            run.emit({"round": rnd, "completed": True, **{k: round(m[k], 6) for k in METRIC_KEYS},
                      **{k: round(v, 1) for k, v in timing.items()},
                      "commit_s": round(commit_s, 1),
                      "worst_round_s": round(worst_round, 1)},
                     wb, {"progress/completed_round": rnd,
                          "progress/global_step": global_step,
                          "round/duration_s": timing["round_s"],
                          "round/train_s": t_train, "round/eval_s": t_eval,
                          "round/aggregate_s": t_agg,
                          "round/commit_s": commit_s,
                          "round/worst_round_s": worst_round,
                          "round/max_skip_pct_seen": max(v["skip_pct"] for v in stats.values()),
                          "round/lr": lr,
                          "round/mu_mean": float(np.mean(list(mu_used.values()))),
                          "round/mu_next_mean": float(np.mean(list(mus.values()))),
                          "round/drift_mean": float(np.mean(list(drifts.values()))),
                          **{f"eval/{k}": m[k] for k in METRIC_KEYS}}, global_step)

            remaining = args.rounds - rnd
            if remaining and time.perf_counter() + worst_round * 1.15 > deadline:
                log(f"stopping cleanly after round {rnd}: another round at the worst "
                    f"observed {worst_round / 60:.1f} min would pass the {args.max_hours} h "
                    "budget. Resume attaches this output.")
                break
    finally:
        for q in ctrl:
            q.put(("stop",))
        for p in procs:
            p.join(timeout=120)
        if wb:
            wb.finish()
    log(f"done in {(time.perf_counter() - t_start) / 3600:.2f} h; "
        f"last complete round {run.last_complete_round()}")


def _check_client(info, max_skip_pct, max_skip_abs):
    """Reject a client update before it can reach the aggregator.

    Finite weights are not enough. A client whose AMP steps were all skipped returns the
    global weights unchanged and finite, and averaging it silently reports participation
    that did not happen. Failing the round keeps every committed round intact and lets a
    resume retry it; dropping the client instead would change the participation rate
    mid-run, which the specification fixes at 100%.

    The loss and gradient means arrive already restricted to applied steps, so a
    non-finite value here means the steps that actually updated the model diverged --
    not that the scaler skipped one while calibrating.
    """
    cid = info["client_id"]
    problems = []
    if not info["finite_weights"]:
        problems.append("non-finite weights")
    if not info["finite_loss"]:
        problems.append(f"non-finite mean loss ({info['ce_mean']})")
    if not info["finite_grad"]:
        problems.append(f"non-finite mean gradient norm ({info['grad_norm_mean']})")
    if info["steps"] and info["skipped"] >= info["steps"]:
        problems.append(f"every one of {info['steps']} optimizer steps was skipped")
    else:
        allowed = max(max_skip_abs, math.ceil(max_skip_pct / 100 * info["steps"]))
        if info["skipped"] > allowed:
            problems.append(f"skipped {info['skipped']}/{info['steps']} steps "
                            f"({info['skip_pct']:.2f}%), above the allowance of {allowed} "
                            f"(max({max_skip_abs} warm-up, {max_skip_pct}%))")
    if problems:
        raise RuntimeError(
            f"client {cid} update rejected: {'; '.join(problems)}. "
            f"rows={info['rows']} steps={info['steps']} skipped={info['skipped']} "
            f"lr={info['lr']} mu={info['mu']} device={info['device']}. "
            "Rounds already committed are intact; resume retries this round.")


def _get(q, procs=None, timeout=7200, poll=20.0):
    """Block for a worker message, but notice a dead worker while waiting.

    A worker killed by the OS (OOM most likely) never puts anything on the queue, so a
    single long `get` would sit for the whole timeout. Poll in short slices and check
    liveness between them.
    """
    deadline = time.perf_counter() + timeout
    while True:
        try:
            kind, rank, payload, info = q.get(timeout=poll)
            break
        except queue.Empty:
            if procs:
                dead = [i for i, pr in enumerate(procs)
                        if not pr.is_alive() and pr.exitcode not in (None, 0)]
                if dead:
                    codes = {i: procs[i].exitcode for i in dead}
                    raise RuntimeError(
                        f"worker(s) {codes} exited without reporting. A negative code is a "
                        "signal (-9 is the OOM killer). Committed rounds are intact.")
            if time.perf_counter() > deadline:
                raise TimeoutError(f"no worker message for {timeout}s")
    if kind == "error":
        raise RuntimeError(f"worker {rank} crashed:\n{info['traceback']}")
    return kind, rank, payload, info


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

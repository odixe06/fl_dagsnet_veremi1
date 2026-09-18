"""TinyProto-FP server loop — Algorithm 1, with two persistent GPU workers.

    Server:
      1  build and distribute the mask set {m_j}                       §4.1
      2  for t = 1..T:
      3      every client trains one local epoch                       Eq. (5)
      4      each client returns n_ij ĉ_L[i,j]                         Eq. (10)
      5      ĉ_G[j] <- mean over clients holding j                     Eq. (10)
      6      every client is scored on the full fixed test set         Eq. (12)

Round 1 runs without the prototype term, exactly as the paper specifies ("At the initial round,
each client trains its model without any regularization").

Aggregation is done on client ids in sorted order, never in GPU completion order: floating point
addition is not associative, so letting a race decide the summation order would make the run
irreproducible.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from . import ckpt as C
from .cps import build_masks, mask_stats
from .metrics import (METRIC_KEYS, RULES, aggregate_clients, atomic_write_json, check_metrics,
                      history_columns, history_row, per_class_from_confusion)
from .protos import aggregate_global, communication_cost


def _digest(x) -> str:
    """sha256 over the raw bytes of a tensor or array, in C order."""
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# mu resolution
# ---------------------------------------------------------------------------

def resolve_mu(kind: str, value: float, counts: np.ndarray, K: int) -> tuple[np.ndarray, dict]:
    """Return (mu (K,), provenance).

    `counts` is the (n_clients, K) matrix of n_ij. Three modes, all recorded:

    * `absolute`      -- use `value` for every class. What a grid search pins down.
    * `inv_mean_nij`  -- value / mean(n_ij over nonzero entries), one scalar for all classes.
      The scalar is what makes the grid comparable across the 20/50/100 scenarios, whose n_ij
      differ by 5x.
    * `per_class`     -- value / mean_{i in N_j}(n_ij), one per class. With this, μ_j ĉ_G[j] is
      exactly the SAMPLE-WEIGHTED MEAN of the local prototypes for every class, which is what
      Eq. (4) intends. It is a deviation: the paper writes μ as a single scalar. On a 41:1
      imbalanced dataset a single scalar shrinks a rare class's target to ~9% of the true
      prototype scale and inflates the largest class's to ~3.6x.
    """
    nz = counts[counts > 0]
    mean_all = float(nz.mean()) if nz.size else 1.0
    Nj = (counts > 0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_j = np.where(Nj > 0, counts.sum(axis=0) / np.maximum(Nj, 1), mean_all)
    if kind == "absolute":
        mu = np.full(K, float(value), dtype=np.float64)
    elif kind == "inv_mean_nij":
        mu = np.full(K, float(value) / mean_all, dtype=np.float64)
    elif kind == "per_class":
        mu = float(value) / mean_j
    else:
        raise ValueError(f"unknown mu kind {kind!r}")
    return mu, {"kind": kind, "value": float(value), "mean_nij_all": mean_all,
                "mean_nij_per_class": mean_j.tolist(), "N_j": Nj.tolist(),
                "resolved": mu.tolist()}


# ---------------------------------------------------------------------------
# worker plumbing
# ---------------------------------------------------------------------------

class WorkerPool:
    """Spawned once for the whole run. A worker killed by the OS puts nothing on its queue, so
    every wait polls in short slices and checks liveness between them -- a single long timeout
    turns an OOM kill into a multi-hour hang (perf-federated.md §8c)."""

    def __init__(self, cfg: dict, assignment: list[list[int]], paths: dict, log=print):
        from .fl_worker import worker_main
        ctx = mp.get_context("spawn")
        self.log = log
        self.n = len(assignment)
        self.cmd = [ctx.Queue() for _ in range(self.n)]
        self.res = [ctx.Queue() for _ in range(self.n)]
        self.procs = []
        for r in range(self.n):
            p = ctx.Process(target=worker_main,
                            args=(r, cfg, assignment[r], self.cmd[r], self.res[r], paths),
                            daemon=False)
            p.start()
            self.procs.append(p)

    def broadcast(self, msg: dict) -> None:
        for q in self.cmd:
            q.put(msg)

    def gather(self, expect: str, timeout_s: float = 7200.0) -> list[dict]:
        out: list[dict | None] = [None] * self.n
        deadline = time.time() + timeout_s
        while any(o is None for o in out):
            progressed = False
            for r in range(self.n):
                if out[r] is not None:
                    continue
                try:
                    m = self.res[r].get(timeout=1.0)
                except Exception:
                    m = None
                if m is None:
                    if not self.procs[r].is_alive():
                        code = self.procs[r].exitcode
                        raise RuntimeError(
                            f"worker {r} died with exitcode {code}"
                            + (f" (signal {-code})" if code is not None and code < 0 else "")
                            + f" while waiting for {expect!r}")
                    continue
                progressed = True
                if m.get("cmd") == "error":
                    raise RuntimeError(f"worker {m['rank']} failed:\n{m['traceback']}")
                if m.get("cmd") != expect:
                    raise RuntimeError(f"worker {r} sent {m.get('cmd')!r}, expected {expect!r}")
                out[r] = m
            if not progressed and time.time() > deadline:
                raise TimeoutError(f"waiting for {expect!r} exceeded {timeout_s}s")
        return [o for o in out]                          # type: ignore[misc]

    def stop(self) -> None:
        for q in self.cmd:
            try:
                q.put({"cmd": "stop"})
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run(cfg: dict, paths: dict, out_root: Path, log=print, wandb_run=None) -> dict:
    session_start = float(cfg.get("session_started_at", time.time()))
    cfg = dict(cfg, artifact_version=2,
               scaler=json.loads(Path(paths["scaler"]).read_text()),
               validation_fingerprint=cfg.get("validation_fingerprint", "full-train-fixed-test"))
    if cfg["local_epochs"] != 1:
        raise ValueError("This implementation requires local_epochs=1")
    if sorted(c for group in cfg["assignment"] for c in group) != list(range(cfg["n_clients"])):
        raise ValueError("assignment must contain every client exactly once")
    K = cfg["num_classes"]
    d = cfg["feature_dim"]
    s = cfg["cps_s"]
    T = cfg["rounds"]
    d_run = C.run_dir(out_root, cfg["run_name"])
    class_names = cfg["class_names"]

    # ---- masks (built once, before any round) ------------------------------
    masks = build_masks(d, K, s, seed=cfg["mask_seed"])
    paths = dict(paths, masks=str(d_run / "masks.npy"))
    cfg = dict(cfg, mask_stats=mask_stats(masks))
    log(f"[cps] {cfg['mask_stats']}")

    # ---- resume gate: BEFORE the multi-minute decode -----------------------
    fp = C.fingerprint(cfg)
    cfg["fingerprint"] = fp
    pending = d_run / "import_pending.json"
    if pending.exists():
        C.import_previous(Path(json.loads(pending.read_text())["source"]), d_run)
    start, blob = C.load_resume(d_run, cfg)
    if start == 1 and cfg.get("import_roots"):
        src = C.find_import_source([Path(p) for p in cfg["import_roots"]], cfg["run_name"], fp)
        if src is not None:
            n = C.import_previous(src, d_run)
            log(f"[resume] imported {n} completed rounds from {src}")
            start, blob = C.load_resume(d_run, cfg)
    if cfg.get("require_resume") and start == 1:
        raise SystemExit("require_resume is set and no committed round was found. "
                         "Fix the attachment, not the round counter.")
    log(f"[resume] starting at round {start} of {T} (fingerprint {fp})")
    if (d_run / "masks.npy").exists() and start > 1:
        if not np.array_equal(np.load(d_run / "masks.npy"), masks):
            raise RuntimeError("stored masks differ from the configured masks")
    np.save(d_run / "masks.npy", masks)
    atomic_write_json(d_run / "config.json", cfg)
    C.repair_history(d_run)
    if start > T:
        return {"last_round": start - 1, "run_dir": str(d_run), "already_complete": True}
    if time.time() - session_start >= float(cfg["max_hours"]) * 3600 - 60:
        log("[stop] no session budget left for worker setup")
        return {"last_round": start - 1, "run_dir": str(d_run), "stopped_early": True}

    # ---- workers -----------------------------------------------------------
    assignment = cfg["assignment"]
    pool = WorkerPool(cfg, assignment, paths, log=log)
    try:
        ready = pool.gather("ready", timeout_s=10800.0)
        counts = np.zeros((cfg["n_clients"], K), dtype=np.int64)
        y_true = None
        for m in ready:
            for cid, cc in m["audit_train"]["per_client_class_counts"].items():
                counts[int(cid)] = cc
            if m.get("y_true") is not None:
                y_true = m["y_true"]
        assert y_true is not None, "no worker reported the test labels"
        atomic_write_json(
            d_run / "data_audit.json",
            {"ready": [{k: v for k, v in m.items() if k not in ("preds", "y_true")}
                       for m in ready],
             "per_client_class_counts": counts.tolist()})
        log(f"[setup] workers ready; compile: {[m['compile'] for m in ready]}")

        mu, mu_prov = resolve_mu(cfg["mu_kind"], cfg["mu_value"], counts, K)
        atomic_write_json(d_run / "mu.json", mu_prov)
        log(f"[aps] mu[{cfg['mu_kind']}] = {mu[:4].tolist()} ...")
        mu_t = torch.tensor(mu, dtype=torch.float32)

        comm = communication_cost(counts > 0, K, s, d)
        log(f"[comm] {comm['params_per_round']:,} params/round "
            f"({comm['params_per_round'] / 1e6:.4f} M), "
            f"{comm['compression_vs_fedproto']:.1f}x below dense FedProto")

        # ---- restore client state on resume --------------------------------
        g_sparse = None
        g_has = torch.zeros(K, dtype=torch.bool)
        if blob is not None:
            last = start - 1
            w = torch.load(d_run / "weights" / f"round_{last:03d}.pt",
                           map_location="cpu", weights_only=True)
            pr = torch.load(d_run / "protos" / f"round_{last:03d}.pt",
                            map_location="cpu", weights_only=True)
            for r in range(len(assignment)):
                o = torch.load(d_run / "resume" / f"round_{last:03d}.w{r}.pt",
                               map_location="cpu", weights_only=True)
                pool.cmd[r].put({"cmd": "restore", "weights": w, "optim": o, "protos": pr})
            pool.gather("restored")
            g_sparse = blob["global_sparse"]
            g_has = blob["global_has"]
            C.set_rng_state(blob["rng"])
            log(f"[resume] client state restored from round {last}")

        history: list[dict] = []
        hist_extra = ["train_loss", "reg_loss", "grad_norm", "lr", "seconds", "train_seconds",
                      "eval_seconds", "applied_steps", "skipped_steps", "peak_gb",
                      "comm_params", "mu_mean"]
        hcols = history_columns(hist_extra)
        budget_s = float(cfg["max_hours"]) * 3600.0
        # Seed the round-cost estimate instead of assuming 60 s. A fresh call starts with no
        # measurement, so the first round of every session (and of every sweep candidate) was
        # admitted on a 60 s guess -- and a 100-client round measures ~41 min, which then runs
        # straight past max_hours. Prefer this run's own recorded timings; fall back to the
        # scenario's expected round time from calibration; only then to the old floor.
        worst_round = 0.0
        for tp in sorted((d_run / "logs").glob("timing_*.json")):
            try:
                worst_round = max(worst_round,
                                  float(json.loads(tp.read_text())["round_including_commit_s"]))
            except Exception:
                continue
        if worst_round:
            log(f"[budget] worst observed round so far {worst_round:.0f}s")
        else:
            worst_round = float(cfg.get("expected_round_s") or 0.0)
            if worst_round:
                log(f"[budget] no local timing yet; using expected_round_s {worst_round:.0f}s")

        for rnd in range(start, T + 1):
            if time.time() - session_start + max(60.0, 1.15 * worst_round) >= budget_s:
                log("[stop] session budget exhausted before starting another round")
                break
            t0 = time.time()
            pool.broadcast({"cmd": "round", "round": rnd,
                            "global_proto": g_sparse, "has_global": g_has, "mu": mu_t})
            trained = pool.gather("trained")
            train_s = time.time() - t0

            # ---- Eq. (10): aggregate in sorted client order -----------------
            uploads = sorted([u for m in trained for u in m["uploads"]], key=lambda x: x[0])
            if [u[0] for u in uploads] != list(range(cfg["n_clients"])):
                raise RuntimeError("missing or duplicate client upload; refusing aggregation")
            if not all(np.isfinite(u[1]).all() for u in uploads):
                raise RuntimeError("non-finite prototype upload; refusing aggregation")
            g_comp, n_per_class = aggregate_global(uploads, K, s)
            from .cps import decompress
            g_sparse = torch.from_numpy(decompress(g_comp, masks)).float()      # (K, d)
            g_has = torch.from_numpy(n_per_class > 0)

            # ---- evaluation ------------------------------------------------
            want_preds = bool(cfg.get("save_preds_rounds") and rnd in cfg["save_preds_rounds"])
            pool.broadcast({"cmd": "evaluate", "want_preds": want_preds})
            evaled = pool.gather("evaluated")

            n_cli = cfg["n_clients"]
            cm_p = np.zeros((n_cli, K, K), dtype=np.int64)
            cm_c = np.zeros((n_cli, K, K), dtype=np.int64)
            preds_p = preds_c = None
            for m in evaled:
                for k, cid in enumerate(m["client_ids"]):
                    cm_p[cid] = m["cm_proto"][k]
                    cm_c[cid] = m["cm_clf"][k]
                if want_preds and m["preds"] is not None:
                    if preds_p is None:
                        n_test = m["preds"]["proto"].shape[1]
                        preds_p = np.zeros((n_cli, n_test), dtype=np.uint8)
                        preds_c = np.zeros((n_cli, n_test), dtype=np.uint8)
                    for k, cid in enumerate(m["client_ids"]):
                        preds_p[cid] = m["preds"]["proto"][k]
                        preds_c[cid] = m["preds"]["clf"][k]

            n_test = len(y_true)
            agg = {"proto": aggregate_clients(cm_p), "clf": aggregate_clients(cm_c)}
            for rule in RULES:
                for cid in range(n_cli):
                    arr = cm_p if rule == "proto" else cm_c
                    check_metrics(agg[rule]["per_client"][cid], n_test, int(arr[cid].sum()))
                    if not np.array_equal(arr[cid].sum(1), np.bincount(y_true, minlength=K)):
                        raise RuntimeError(f"client {cid}/{rule}: evaluation support mismatch")

            stats = sorted([s_ for m in trained for s_ in m["stats"]], key=lambda x: x["client_id"])
            eval_s = max(m["eval_seconds"] for m in evaled)
            peak = max(max(m["peak_gb"] for m in trained), max(m["peak_gb"] for m in evaled))
            extra = {
                "train_loss": float(np.mean([x["ce"] for x in stats])),
                "reg_loss": float(np.mean([x["reg"] for x in stats])),
                "grad_norm": float(np.mean([x["grad_norm"] for x in stats])),
                "lr": cfg["lr"],
                "seconds": round(time.time() - t0, 1),
                "train_seconds": round(train_s, 1), "eval_seconds": round(eval_s, 1),
                "applied_steps": int(sum(x["applied_steps"] for x in stats)),
                "skipped_steps": int(sum(x["skipped_steps"] for x in stats)),
                "peak_gb": peak,
                "comm_params": comm["params_per_round"],
                "mu_mean": float(mu.mean()),
            }
            # ---- collect client state before writing anything ----------------
            pool.broadcast({"cmd": "commit", "round": rnd, "run_dir": str(d_run)})
            committed = pool.gather("committed")
            n_par = committed[0]["weights"]["params"].shape[1]
            W = {"client_ids": list(range(n_cli)),
                 "params": torch.zeros(n_cli, n_par),
                 "buffers": torch.zeros(n_cli, committed[0]["weights"]["buffers"].shape[1]),
                 "int_buffers": torch.zeros(n_cli, committed[0]["weights"]["int_buffers"].shape[1],
                                            dtype=torch.long)}
            PL = torch.zeros(n_cli, K, d)
            PC = torch.zeros(n_cli, K, dtype=torch.int64)
            for m in committed:
                for k, cid in enumerate(m["weights"]["client_ids"]):
                    W["params"][cid] = m["weights"]["params"][k]
                    W["buffers"][cid] = m["weights"]["buffers"][k]
                    W["int_buffers"][cid] = m["weights"]["int_buffers"][k]
                for k, cid in enumerate(m["protos"]["client_ids"]):
                    PL[cid] = m["protos"]["local"][k]
                    PC[cid] = m["protos"]["counts"][k]

            # Digests tie the metrics JSON to the tensors it was computed from. Without them a
            # change to a prototype dimension that no mask selects passes every other check: it
            # is never communicated, so the recomputed Eq. (10) aggregate is unaffected -- but
            # Eq. (12) uses the DENSE prototype, so it does change the predictions.
            digests = {"local_protos": _digest(PL), "proto_counts": _digest(PC),
                       "params": _digest(W["params"]), "buffers": _digest(W["buffers"]),
                       "int_buffers": _digest(W["int_buffers"]),
                       "confusion_proto": _digest(cm_p), "confusion_clf": _digest(cm_c)}
            metrics = {"round": rnd, "n_test": n_test, "class_names": class_names,
                       "digests": digests,
                       "aggregate": {r: {k: v for k, v in agg[r].items() if k != "per_client"}
                                     for r in RULES},
                       "per_client": {r: agg[r]["per_client"] for r in RULES},
                       "per_class": {r: per_class_from_confusion(cm_p.sum(0) if r == "proto"
                                                                 else cm_c.sum(0), class_names)
                                     for r in RULES},
                       "extra": extra, "communication": comm,
                       "global_proto_classes": int(g_has.sum())}

            history = C.rebuild_history(d_run, hcols,
                                        lambda j: history_row(j["round"], j["aggregate"], j["extra"]))
            history.append(history_row(rnd, agg, extra))
            preds = {}
            if want_preds and preds_p is not None:
                preds[f"round_{rnd:03d}_proto"] = preds_p
                preds[f"round_{rnd:03d}_clf"] = preds_c
            if not (d_run / "preds" / "y_true.npy").exists():
                preds["y_true"] = y_true

            C.commit_round(
                d_run, rnd,
                weights={"manifest": cfg["packer_manifest"], **W},
                protos={"local": PL, "counts": PC, "global_compressed": torch.from_numpy(g_comp),
                        "global_sparse": g_sparse, "global_has": g_has,
                        "n_clients_per_class": torch.from_numpy(n_per_class)},
                confusion={"proto": cm_p, "clf": cm_c},
                metrics=metrics,
                client_rows=stats, client_columns=list(stats[0].keys()),
                resume={"round": rnd, "fingerprint": fp, "rng": C.rng_state(),
                        "global_sparse": g_sparse, "global_has": g_has,
                        "mu": mu.tolist(), "n_workers": len(assignment)},
                history_rows=history, history_columns=hcols,
                preds=preds or None)

            mp_ = agg["proto"]["mean_over_clients"]
            mc_ = agg["clf"]["mean_over_clients"]
            log(f"[round {rnd:03d}] ce {extra['train_loss']:.4f} reg {extra['reg_loss']:.4f} | "
                f"proto acc {mp_['accuracy']:.4f} F1m {mp_['f1_macro']:.4f} | "
                f"clf acc {mc_['accuracy']:.4f} F1m {mc_['f1_macro']:.4f} | "
                f"{extra['seconds']}s (train {extra['train_seconds']}s eval {extra['eval_seconds']}s) "
                f"peak {peak}GB")
            if wandb_run is not None:
                # Telemetry sits between the commit and the timing write. An exception escaping
                # here kills a run whose round is already safely on disk, and loses the timing
                # the NEXT session's budget gate reads back. Never let it out; drop telemetry
                # for the rest of the session instead of failing the same way every round.
                try:
                    flat = {f"{r}/{k}": agg[r]["mean_over_clients"][k]
                            for r in RULES for k in METRIC_KEYS}
                    flat.update({f"train/{k}": v for k, v in extra.items()})
                    wandb_run.log(flat, step=rnd)
                except Exception as e:
                    log(f"[wandb] logging disabled after {type(e).__name__}: {str(e)[:120]}")
                    wandb_run = None

            worst_round = max(worst_round, time.time() - t0)
            atomic_write_json(d_run / "logs" / f"timing_{rnd:03d}.json",
                              {"round": rnd, "round_including_commit_s": time.time() - t0,
                               "session_elapsed_s": time.time() - session_start})
            elapsed = time.time() - session_start
            cap = cfg.get("rounds_this_session")
            if rnd < T and cap and (rnd - start + 1) >= int(cap):
                atomic_write_json(d_run / "logs" / "stopped_early.json",
                                  {"last_round": rnd, "elapsed_s": elapsed,
                                   "reason": "rounds_this_session"})
                log(f"[stop] session cap of {cap} rounds reached after round {rnd}; "
                    f"resume the next session from round {rnd + 1}")
                break
            if rnd < T and elapsed + 1.15 * worst_round >= budget_s:
                atomic_write_json(d_run / "logs" / "stopped_early.json",
                                  {"last_round": rnd, "elapsed_s": elapsed,
                                   "worst_round_s": worst_round, "reason": "wall_clock_budget"})
                log(f"[stop] wall-clock budget reached after round {rnd}; "
                    f"resume the next session from round {rnd + 1}")
                break
    finally:
        pool.stop()

    return {"last_round": C.last_complete(d_run), "run_dir": str(d_run)}

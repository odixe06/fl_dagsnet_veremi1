"""Re-derive every published number from the artifacts on disk.

Nothing here trusts a number because it was printed once. Per client and per round, the
10 metrics and the per-class block are recomputed from that client's confusion matrix; the
mean/std/min/max over clients are recomputed from the per-client values; the aggregated
proxy's 10 metrics are recomputed from its own matrix; predictions, where stored, rebuild
the confusion matrices; the client logs are checked against the step arithmetic and the
schedule they claim; and all of it is compared against the copies in the weights file,
history.csv and clients.csv.

Every check exists because its absence lets a specific tampered fixture pass: a deleted
history row, a duplicated one, a metric set to NaN (`abs(nan) > tol` is False), a per-class
F1 of 999, deleted logs, a client missing from the log, a proxy metric overwritten, a
resume file overwritten with garbage, and a config claiming 999 test rows.
"""
import csv, json, math
from pathlib import Path
import numpy as np
import torch

from proj.metrics import metrics_from_confusion, per_class_from_confusion, METRIC_KEYS
from proj.nilm import expected_steps, lr_at, ACC_KEYS
from proj import ckpt as C

TOL = 1e-12          # both sides come from the same float64 code path on the same counts
CSV_TOL = 1e-9       # history.csv round-trips through str()


def _finite(x):
    try: return math.isfinite(float(x))
    except (TypeError, ValueError): return False


def _read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def verify_run(run_dir, cfg=None, build_model=None, expect_params=None, y_true_path=None,
               require_rounds=None, full=True):
    """Returns (ok, lines). full=True is the acceptance mode: client logs must be present
    for every round and predictions for every round that claims them."""
    d = Path(run_dir)
    fp = C.fingerprint(cfg) if cfg is not None else None
    last = C.last_complete_round(d, fp) or 0
    # A tree that starts past round 1 is a session resumed from a handoff bundle: rounds
    # before `first` are attested by reports/handoff.json (checked inside round_ok) and are
    # re-verified in full only after the sessions are merged. Nothing here is skipped for
    # the rounds that ARE on disk.
    first = C._first_round(d) if last else 1
    mode = "full" if full else "minimal"
    out = [f"run      : {d}", f"complete : rounds {first}..{last}   (mode: {mode})"
           + (f"   [handoff bundle: rounds 1..{first} attested by {C.HANDOFF}]" if first > 1 else "")]
    bad = []

    mf = d / "reports" / "manifest.json"
    man = json.loads(mf.read_text()) if mf.is_file() else None
    if man is None and full:
        bad.append("reports/manifest.json missing: the run does not describe itself")

    N = int(cfg["n_clients"]) if cfg else (int(man["n_clients"]) if man else None)
    n_test = int(cfg["n_test"]) if cfg and "n_test" in cfg else None
    want_rows = ({int(k): int(v) for k, v in man["client_rows"].items()}
                 if man and "client_rows" in man else None)

    # ---- history.csv / clients.csv: exactly rounds 1..last, once each
    hist, hp = {}, d / "history.csv"
    if hp.is_file():
        rows = _read_csv(hp)
        seen = [int(r["round"]) for r in rows]
        if len(seen) != len(set(seen)):
            bad.append(f"history.csv has duplicate rows for round(s) "
                       f"{sorted({r for r in seen if seen.count(r) > 1})}")
        if sorted(set(seen)) != list(range(first, last + 1)):
            bad.append(f"history.csv covers rounds {sorted(set(seen))}, expected {first}..{last}")
        hist = {int(r["round"]): r for r in rows}
    elif last:
        bad.append("history.csv missing")
    crows, cp = {}, d / "clients.csv"
    if cp.is_file():
        for r in _read_csv(cp):
            crows.setdefault(int(r["round"]), {})[int(r["cid"])] = r
    elif last:
        bad.append("clients.csv missing")

    y_true = None
    if y_true_path is not None and Path(y_true_path).is_file():
        y_true = np.load(y_true_path, mmap_mode="r")

    for r in range(first, last + 1):
        tag = f"round {r:03d}"
        cm = np.load(d / "confusion" / f"round_{r:03d}.npy")
        gcm = np.load(d / "confusion" / f"global_{r:03d}.npy")
        if cm.ndim != 3 or cm.shape[1] != cm.shape[2] or (N is not None and cm.shape[0] != N):
            bad.append(f"{tag}: confusion is {cm.shape}, expected ({N}, C, C)"); continue
        if gcm.shape != cm.shape[1:]:
            bad.append(f"{tag}: proxy confusion is {gcm.shape}, expected {cm.shape[1:]}"); continue
        if (cm < 0).any() or (gcm < 0).any():
            bad.append(f"{tag}: negative counts in a confusion matrix")
        if cfg and cm.shape[1] != int(cfg["num_classes"]):
            bad.append(f"{tag}: {cm.shape[1]} classes, cfg says {cfg['num_classes']}")
        totals = cm.sum(axis=(1, 2))
        if n_test is None: n_test = int(totals[0])
        if not (totals == n_test).all() or gcm.sum() != n_test:
            bad.append(f"{tag}: confusion totals {sorted(set(totals.tolist()))} / proxy "
                       f"{int(gcm.sum())} != n_test {n_test}")

        js = json.loads((d / "metrics" / f"round_{r:03d}.json").read_text())
        ck = torch.load(d / "weights" / f"round_{r:03d}.pt", map_location="cpu",
                        weights_only=True, mmap=True)
        clients = js.get("clients", [])
        if [c.get("cid") for c in clients] != list(range(cm.shape[0])):
            bad.append(f"{tag}: metrics json lists clients "
                       f"{[c.get('cid') for c in clients][:5]}..., expected 0..{cm.shape[0]-1}")
            continue
        per = {k: [] for k in METRIC_KEYS}
        for c in clients:
            cid = int(c["cid"])
            rec = metrics_from_confusion(cm[cid])
            for k in METRIC_KEYS:
                v = c.get(k)
                if not _finite(v) or abs(float(v) - rec[k]) > TOL:
                    bad.append(f"{tag}: client {cid} {k} {v!r} != {rec[k]} recomputed")
                per[k].append(rec[k])
                cv = crows.get(r, {}).get(cid, {}).get(k)
                if crows and (cv is None or not _finite(cv) or abs(float(cv) - rec[k]) > CSV_TOL):
                    bad.append(f"{tag}: clients.csv client {cid} {k} = {cv!r} != {rec[k]}")
            names = [e.get("class") for e in c.get("per_class", [])]
            want_pc = per_class_from_confusion(cm[cid], names) if len(names) == cm.shape[1] else None
            if want_pc is None:
                bad.append(f"{tag}: client {cid} per-class block missing or wrong length")
            else:
                for got, want in zip(c["per_class"], want_pc):
                    for f in ("idx", "support"):
                        if int(got.get(f, -1)) != int(want[f]):
                            bad.append(f"{tag}: client {cid} class {want['idx']} {f} "
                                       f"{got.get(f)} != {want[f]}")
                    for f in ("precision", "recall", "f1"):
                        v = got.get(f)
                        if not _finite(v) or abs(float(v) - want[f]) > TOL:
                            bad.append(f"{tag}: client {cid} class {want['idx']} {f} "
                                       f"{v!r} != {want[f]}")
        # ---- the aggregate row: mean/std/min/max over clients, in json, weights, csv
        for k in METRIC_KEYS:
            v = np.asarray(per[k], dtype=np.float64)
            want = {k: float(v.mean()), f"{k}_std": float(v.std()),
                    f"{k}_min": float(v.min()), f"{k}_max": float(v.max())}
            for kk, wv in want.items():
                for where, val, tol in (("json", js.get(kk), TOL),
                                        ("history.csv", hist.get(r, {}).get(kk), CSV_TOL)):
                    if val is None:
                        bad.append(f"{tag}: {kk} missing in {where}"); continue
                    if not _finite(val):
                        bad.append(f"{tag}: {kk} in {where} is not finite ({val!r})")
                    elif abs(float(val) - wv) > tol:
                        bad.append(f"{tag}: {kk} in {where} = {val} != {wv} recomputed")
            wv = ck.get("metrics", {}).get(k)
            if wv is None or not _finite(wv) or abs(float(wv) - want[k]) > TOL:
                bad.append(f"{tag}: weights file {k} = {wv!r} != {want[k]}")
        # ---- the aggregated proxy: its own 10 metrics + per-class, in json, weights, csv
        grec = metrics_from_confusion(gcm)
        gj = js.get("global", {})
        for k in METRIC_KEYS:
            for where, val, tol in (("json global", gj.get(k), TOL),
                                    ("json row", js.get(f"global_{k}"), TOL),
                                    ("history.csv", hist.get(r, {}).get(f"global_{k}"), CSV_TOL),
                                    ("weights file", ck.get("global_metrics", {}).get(k), TOL)):
                if val is None:
                    bad.append(f"{tag}: proxy {k} missing in {where}"); continue
                if not _finite(val):
                    bad.append(f"{tag}: proxy {k} in {where} is not finite ({val!r})")
                elif abs(float(val) - grec[k]) > tol:
                    bad.append(f"{tag}: proxy {k} in {where} = {val} != {grec[k]} recomputed")
        names = [e.get("class") for e in gj.get("per_class", [])]
        if len(names) != cm.shape[1]:
            bad.append(f"{tag}: proxy per-class block missing or wrong length")
        else:
            for got, want in zip(gj["per_class"], per_class_from_confusion(gcm, names)):
                for f in ("precision", "recall", "f1"):
                    v = got.get(f)
                    if not _finite(v) or abs(float(v) - want[f]) > TOL:
                        bad.append(f"{tag}: proxy class {want['idx']} {f} {v!r} != {want[f]}")
                if int(got.get("support", -1)) != int(want["support"]):
                    bad.append(f"{tag}: proxy class {want['idx']} support mismatch")
        if int(js.get("evaluated", -1)) != cm.shape[0]:
            bad.append(f"{tag}: json 'evaluated' {js.get('evaluated')} != {cm.shape[0]} clients")
        if cfg is not None:
            want_lr = lr_at(cfg, r)
            if not _finite(js.get("lr")) or abs(float(js["lr"]) - want_lr) > 1e-12:
                bad.append(f"{tag}: json lr {js.get('lr')!r} != schedule {want_lr}")

        if fp is not None and ck.get("fingerprint") != fp:
            bad.append(f"{tag}: weights fingerprint {ck.get('fingerprint')} != {fp}")
        if build_model is not None:
            try:
                C.load_weights(d / "weights" / f"round_{r:03d}.pt", build_model, expect_params)
            except Exception as e:
                bad.append(f"{tag}: weights do not rebuild the models: {e}")

        # ---- predictions tie the matrices back to model output, where stored
        pp = d / "preds" / f"round_{r:03d}.u8.npy"
        gp = d / "preds" / f"global_{r:03d}.u8.npy"
        claims = cfg is not None and r in set(cfg.get("preds_rounds", [cfg["rounds"]]))
        if pp.is_file():
            yp = np.load(pp, mmap_mode="r")
            ygp = np.load(gp, mmap_mode="r") if gp.is_file() else None
            if yp.shape != (cm.shape[0], n_test):
                bad.append(f"{tag}: predictions are {yp.shape}, expected ({cm.shape[0]}, {n_test})")
            elif ygp is None or ygp.shape != (n_test,):
                bad.append(f"{tag}: proxy predictions missing or not ({n_test},)")
            elif y_true is not None:
                if len(y_true) != n_test:
                    bad.append(f"{tag}: y_true has {len(y_true)} rows, test has {n_test}")
                else:
                    k = cm.shape[1]
                    yt = np.asarray(y_true, np.int64) * k
                    for cid in range(cm.shape[0]):
                        rebuilt = np.bincount(yt + np.asarray(yp[cid], np.int64),
                                              minlength=k * k).reshape(k, k)
                        if not (rebuilt == cm[cid]).all():
                            bad.append(f"{tag}: client {cid} confusion != stored predictions")
                    rebuilt = np.bincount(yt + np.asarray(ygp, np.int64),
                                          minlength=k * k).reshape(k, k)
                    if not (rebuilt == gcm).all():
                        bad.append(f"{tag}: proxy confusion != stored predictions")
            elif full:
                bad.append(f"{tag}: predictions present but no y_true to check them against")
        elif claims and full:
            bad.append(f"{tag}: cfg claims predictions for this round but none are stored")

        # ---- client logs: every client, step arithmetic and the schedule have to close
        lp = d / "logs" / f"round_{r:03d}.json"
        if not lp.is_file():
            if full:
                bad.append(f"{tag}: no client log; participation is unattested")
        else:
            try: lg = json.loads(lp.read_text())
            except Exception as e:
                bad.append(f"{tag}: client log unreadable: {e}"); lg = {}
            got_ids = sorted(int(e["cid"]) for e in lg.get("clients", []))
            if got_ids != list(range(cm.shape[0])):
                bad.append(f"{tag}: log has clients {got_ids[:6]}..., expected all "
                           f"0..{cm.shape[0] - 1}")
            for e in lg.get("clients", []):
                c = e.get("cid")
                if e.get("applied", 0) + e.get("skipped", 0) != e.get("steps", -1):
                    bad.append(f"{tag}: client {c} applied+skipped != steps")
                if e.get("applied", 0) <= 0:
                    bad.append(f"{tag}: client {c} applied no step")
                if e.get("nonfinite", 0):
                    bad.append(f"{tag}: client {c} applied a step with a non-finite gradient")
                if cfg and "n_k" in e:
                    s = expected_steps(int(e["n_k"]), cfg)
                    if int(e.get("steps", -1)) != s:
                        bad.append(f"{tag}: client {c} ran {e.get('steps')} steps, expected {s}")
                if want_rows is not None and c in want_rows and int(e.get("n_k", -1)) != want_rows[c]:
                    bad.append(f"{tag}: client {c} trained on {e.get('n_k')} rows, "
                               f"manifest says {want_rows[c]}")
                # The rate the client actually used must be the schedule's value for this
                # round: a resumed session that planned a different horizon would otherwise
                # continue the run at a rate the fingerprint never saw.
                if cfg is not None:
                    want_lr = lr_at(cfg, r)
                    if not _finite(e.get("lr")) or abs(float(e["lr"]) - want_lr) > 1e-12:
                        bad.append(f"{tag}: client {c} trained at lr {e.get('lr')!r}, "
                                   f"schedule says {want_lr}")
                for f in ACC_KEYS:
                    if not _finite(e.get(f)):
                        bad.append(f"{tag}: client {c} {f} is not finite ({e.get(f)!r})")

    n_preds = len(list((d / "preds").glob("round_*.u8.npy"))) if (d / "preds").is_dir() else 0
    n_logs = len(list((d / "logs").glob("round_*.json"))) if (d / "logs").is_dir() else 0
    out.append(f"artifacts: {n_preds} prediction files, {n_logs} client logs"
               + ("" if y_true is not None else "   (predictions NOT cross-checked: no y_true)"))
    if require_rounds is not None and last != require_rounds:
        bad.append(f"run is INCOMPLETE: {last} of {require_rounds} rounds")
    out += [f"  FAIL {b}" for b in bad] or ["  all artifact checks passed"]
    return not bad, out

"""Re-derive every published number from the artifacts on disk.

Nothing here trusts a number because it was printed once. The 10 metrics and the per-class
block are recomputed from the confusion matrix; the confusion matrix is rebuilt cell by cell
from the stored predictions; the client logs are checked against the step arithmetic they
claim; and all of it is compared against the copies in the weights file and history.csv.

Every check below exists because its absence let a specific tampered fixture pass: a deleted
history row, a duplicated one, a metric set to NaN (`abs(nan) > tol` is False, so a plain
difference test accepts it), a per-class F1 of 999, deleted predictions, deleted logs, a
resume file overwritten with garbage, and a config claiming 999 test rows.
"""
import csv, json, math
from pathlib import Path
import numpy as np
import torch

from proj.metrics import metrics_from_confusion, per_class_from_confusion, METRIC_KEYS
from proj import ckpt as C

TOL = 1e-12          # both sides come from the same float64 code path on the same counts
CSV_TOL = 1e-9       # history.csv round-trips through str()


def _finite(x):
    try: return math.isfinite(float(x))
    except (TypeError, ValueError): return False


def verify_run(run_dir, cfg=None, build=None, expect_params=None, y_true_path=None,
               require_rounds=None, full=True):
    """Returns (ok, lines).

    full=True is the acceptance mode: predictions and client logs must be present for every
    round. full=False checks only that the checkpoints themselves are consistent -- use it
    while a run is still in flight, never to certify one.
    """
    d = Path(run_dir)
    fp = C.fingerprint(cfg) if cfg is not None else None
    last = C.last_complete_round(d, fp) or 0
    mode = "full" if full else "minimal"
    out = [f"run      : {d}", f"complete : rounds 1..{last}   (mode: {mode})"]
    bad = []

    mf = d / "reports" / "manifest.json"
    man = json.loads(mf.read_text()) if mf.is_file() else None
    if man is None and full:
        bad.append("reports/manifest.json missing: the run does not describe itself")

    n_test = int(cfg["n_test"]) if cfg and "n_test" in cfg else None
    batch = int(cfg["batch"]) if cfg and "batch" in cfg else None
    epochs = int(cfg["local_epochs"]) if cfg and "local_epochs" in cfg else None
    want_clients = ({int(k): int(v) for k, v in man["client_rows"].items()}
                    if man and "client_rows" in man else None)

    # ---- history.csv: exactly rounds 1..last, once each
    hist, hp = {}, d / "history.csv"
    if hp.is_file():
        with open(hp) as f:
            rows = list(csv.DictReader(f))
        seen = [int(r["round"]) for r in rows]
        if len(seen) != len(set(seen)):
            dup = sorted({r for r in seen if seen.count(r) > 1})
            bad.append(f"history.csv has duplicate rows for round(s) {dup}")
        if sorted(set(seen)) != list(range(1, last + 1)):
            bad.append(f"history.csv covers rounds {sorted(set(seen))}, expected 1..{last}")
        hist = {int(r["round"]): r for r in rows}
    elif last:
        bad.append("history.csv missing")

    for r in range(1, last + 1):
        tag = f"round {r:03d}"
        cm = np.load(d / "confusion" / f"round_{r:03d}.npy")
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            bad.append(f"{tag}: confusion is {cm.shape}"); continue
        if (cm < 0).any():
            bad.append(f"{tag}: negative counts in the confusion matrix")
        if cfg and cm.shape[0] != int(cfg["num_classes"]):
            bad.append(f"{tag}: confusion is {cm.shape[0]}x{cm.shape[0]}, "
                       f"cfg says {cfg['num_classes']} classes")
        total = int(cm.sum())
        if n_test is not None and total != n_test:
            bad.append(f"{tag}: confusion covers {total} rows, cfg declares n_test={n_test}")
        elif n_test is None:
            n_test = total                      # no cfg: at least demand they agree
        if total != n_test:
            bad.append(f"{tag}: covers {total} rows, an earlier round covered {n_test}")

        recomputed = metrics_from_confusion(cm)
        js = json.loads((d / "metrics" / f"round_{r:03d}.json").read_text())
        ck = torch.load(d / "weights" / f"round_{r:03d}.pt", map_location="cpu",
                        weights_only=True)
        for k in METRIC_KEYS:
            if k not in js:
                bad.append(f"{tag}: metrics json has no {k}"); continue
            # isfinite FIRST: NaN fails every comparison, so a difference test alone
            # silently accepts a metric that was overwritten with NaN.
            for where, val in (("json", js[k]), ("weights", ck.get("metrics", {}).get(k)),
                               ("history.csv", hist.get(r, {}).get(k))):
                if val is None: continue
                if not _finite(val):
                    bad.append(f"{tag}: {k} in {where} is not a finite number ({val!r})")
                elif abs(float(val) - recomputed[k]) > (CSV_TOL if where == "history.csv"
                                                        else TOL):
                    bad.append(f"{tag}: {k} in {where} = {val} != {recomputed[k]} "
                               "recomputed from the confusion matrix")
            if k not in ck.get("metrics", {}):
                bad.append(f"{tag}: the weights file carries no {k}")
            if hist and r in hist and k not in hist[r]:
                bad.append(f"{tag}: history.csv has no {k} column")

        # ---- per-class, recomputed field by field. Support alone is not a check: it comes
        # straight from the row sums and stays right while precision/recall/F1 are anything.
        names = [e.get("class") for e in js.get("per_class", [])]
        want_pc = per_class_from_confusion(cm, names) if len(names) == cm.shape[0] else None
        if want_pc is None:
            bad.append(f"{tag}: per-class block missing or wrong length")
        else:
            for got, want in zip(js["per_class"], want_pc):
                for f in ("idx", "support"):
                    if int(got.get(f, -1)) != int(want[f]):
                        bad.append(f"{tag}: class {want['idx']} {f} {got.get(f)} != {want[f]}")
                for f in ("precision", "recall", "f1"):
                    v = got.get(f)
                    if not _finite(v) or abs(float(v) - want[f]) > TOL:
                        bad.append(f"{tag}: class {want['idx']} {f} {v!r} != {want[f]}")

        if fp is not None and ck.get("fingerprint") != fp:
            bad.append(f"{tag}: weights fingerprint {ck.get('fingerprint')} != {fp}")
        if build is not None:
            try:
                C.load_weights(d / "weights" / f"round_{r:03d}.pt", build,
                               expect_params=expect_params)
            except Exception as e:
                bad.append(f"{tag}: weights do not rebuild the model: {e}")

        # ---- predictions tie the matrix back to model output
        pp = d / "preds" / f"round_{r:03d}.u8.npy"
        if not pp.is_file():
            if full:
                bad.append(f"{tag}: no predictions; the confusion matrix is unattested")
        else:
            yp = np.load(pp, mmap_mode="r")
            if len(yp) != total:
                bad.append(f"{tag}: {len(yp)} predictions for {total} test rows")
            elif y_true_path is not None and Path(y_true_path).is_file():
                yt = np.load(y_true_path, mmap_mode="r")
                if len(yt) != len(yp):
                    bad.append(f"{tag}: y_true has {len(yt)} rows, predictions {len(yp)}")
                else:
                    k = cm.shape[0]
                    rebuilt = np.bincount(np.asarray(yt, np.int64) * k
                                          + np.asarray(yp, np.int64),
                                          minlength=k * k).reshape(k, k)
                    if not (rebuilt == cm).all():
                        bad.append(f"{tag}: confusion does not match the stored predictions")
            elif full:
                bad.append(f"{tag}: predictions present but no y_true to check them against")

        # ---- client logs: the step arithmetic has to close
        lp = d / "logs" / f"round_{r:03d}.json"
        if not lp.is_file():
            if full:
                bad.append(f"{tag}: no client log; participation is unattested")
        else:
            try: cl = json.loads(lp.read_text())
            except Exception as e:
                bad.append(f"{tag}: client log unreadable: {e}"); cl = []
            got_ids = sorted(int(e["cid"]) for e in cl) if cl else []
            if want_clients is not None and got_ids != sorted(want_clients):
                bad.append(f"{tag}: log has {len(got_ids)} clients, manifest has "
                           f"{len(want_clients)}")
            for e in cl:
                c = e.get("cid")
                if e.get("applied", 0) + e.get("skipped", 0) != e.get("steps", -1):
                    bad.append(f"{tag}: client {c} applied+skipped != steps")
                if e.get("applied", 0) <= 0:
                    bad.append(f"{tag}: client {c} applied no optimizer step")
                if batch and epochs and "n_k" in e:
                    want_steps = epochs * math.ceil(int(e["n_k"]) / batch)
                    if int(e.get("steps", -1)) != want_steps:
                        bad.append(f"{tag}: client {c} ran {e.get('steps')} steps, "
                                   f"{epochs}*ceil({e['n_k']}/{batch}) = {want_steps}")
                if want_clients is not None and c in want_clients \
                        and int(e.get("n_k", -1)) != want_clients[c]:
                    bad.append(f"{tag}: client {c} trained on {e.get('n_k')} rows, "
                               f"manifest says {want_clients[c]}")
                for f in ("ce", "kd", "gnorm"):
                    if not _finite(e.get(f)):
                        bad.append(f"{tag}: client {c} {f} is not finite ({e.get(f)!r})")

    n_preds = len(list((d / "preds").glob("round_*.u8.npy"))) if (d / "preds").is_dir() else 0
    n_logs = len(list((d / "logs").glob("round_*.json"))) if (d / "logs").is_dir() else 0
    out.append(f"artifacts: {n_preds} preds, {n_logs} client logs"
               + ("" if y_true_path and Path(y_true_path).is_file()
                  else "   (predictions NOT cross-checked: no y_true)"))
    if require_rounds is not None and last != require_rounds:
        bad.append(f"run is INCOMPLETE: {last} of {require_rounds} rounds")
    out += [f"  FAIL {b}" for b in bad] or ["  all artifact checks passed"]
    return not bad, out

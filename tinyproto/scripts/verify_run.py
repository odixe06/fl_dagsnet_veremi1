#!/usr/bin/env python
"""Verify a committed TinyProto run against ITSELF.

The point of `references/verifying-artifacts.md` §5: a verifier built on freshly generated
random inputs tests the formula and passes on tampered artifacts. Every check below compares
one stored value against another stored value, or against what the configuration implies.

    python scripts/verify_run.py RUN_DIR [--require-complete]

Exit 0 = every check passed for the rounds that exist. `--require-complete` additionally
demands that the last completed round equals `config.json["rounds"]`; without it, a 3-round
directory whose config says 50 rounds passes, which is correct for a mid-run check and wrong
as evidence that a scenario finished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cps import compress, mask_stats  # noqa: E402
from src.metrics import METRIC_KEYS, RULES, metrics_from_confusion  # noqa: E402
from src.model import build_model  # noqa: E402
from src import ckpt as C
from src.driver import resolve_mu
from src.cps import decompress
from src.metrics import aggregate_clients, per_class_from_confusion


def digest(x) -> str:
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


class Checker:
    def __init__(self):
        self.fail: list[str] = []
        self.passed = 0

    def check(self, ok: bool, msg: str) -> bool:
        if ok:
            self.passed += 1
        else:
            self.fail.append(msg)
        return ok

    def report(self) -> int:
        print(f"\n{self.passed} checks passed, {len(self.fail)} failed")
        for f in self.fail:
            print(f"  FAIL  {f}")
        return 1 if self.fail else 0


def verify(run_dir: Path, require_complete: bool = False, deep_weights: bool = True) -> int:
    c = Checker()
    cfg = json.loads((run_dir / "config.json").read_text())
    c.check(cfg.get("fingerprint") == C.fingerprint(cfg), "config fingerprint mismatch")
    try:
        C.validate_integrity(run_dir)
    except (RuntimeError, OSError, ValueError) as e:
        c.check(False, str(e))
        return c.report()
    K = cfg["num_classes"]
    d = cfg["feature_dim"]
    s = cfg["cps_s"]
    n_cli = cfg["n_clients"]
    done = sorted(int(p.name[6:9]) for p in (run_dir / "complete").glob("round_*.done"))

    print(f"run       : {run_dir}")
    print(f"scenario  : {cfg['scenario']}  clients={n_cli}  s={s}  d={d}  rounds={cfg['rounds']}")
    print(f"completed : {len(done)} rounds {done[:3]}{'...' if len(done) > 3 else ''}"
          f"{done[-1:] if len(done) > 3 else ''}")

    c.check(bool(done), "no completed rounds")
    if not done:
        return c.report()
    c.check(done == list(range(1, done[-1] + 1)), f"rounds are not contiguous 1..{done[-1]}: {done}")
    if require_complete:
        c.check(done[-1] == cfg["rounds"],
                f"--require-complete: last round {done[-1]} != config rounds {cfg['rounds']}")

    # ---- masks -------------------------------------------------------------
    masks = np.load(run_dir / "masks.npy")
    c.check(masks.shape == (K, d), f"mask shape {masks.shape} != ({K}, {d})")
    c.check(bool((masks.sum(1) == s).all()), "some mask does not have exactly s ones")
    st = mask_stats(masks)
    c.check(st == cfg["mask_stats"], "recomputed mask_stats disagrees with config.json")

    # ---- assignment --------------------------------------------------------
    flat = sorted(x for b in cfg["assignment"] for x in b)
    c.check(flat == list(range(n_cli)), "assignment is not exactly the client set 0..n-1")

    audit = json.loads((run_dir / "data_audit.json").read_text())
    counts_cfg = np.array(audit["per_client_class_counts"], dtype=np.int64)
    c.check(counts_cfg.shape == (n_cli, K), "per_client_class_counts has the wrong shape")

    # ---- history vs per-round JSON ----------------------------------------
    import csv
    with (run_dir / "metrics" / "history.csv").open() as fh:
        history_rows = list(csv.DictReader(fh))
        hist = {int(r["round"]): r for r in history_rows}
    c.check(len(history_rows) == len(hist), "duplicate rounds in history.csv")
    c.check(sorted(hist) == done, f"history.csv rounds {sorted(hist)} != markers {done}")

    # existence first: a missing artifact must be a reported failure, not a traceback
    for rnd in done:
        for rel in (f"weights/round_{rnd:03d}.pt", f"protos/round_{rnd:03d}.pt",
                    f"confusion/round_{rnd:03d}.npz", f"metrics/round_{rnd:03d}.json",
                    f"client_log/round_{rnd:03d}.csv"):
            if not (run_dir / rel).exists():
                c.check(False, f"round {rnd} is marked complete but {rel} is missing")
    if c.fail:
        return c.report()

    ref_y_true = None
    yt_path = run_dir / "preds" / "y_true.npy"
    if yt_path.exists():
        ref_y_true = np.load(yt_path)
    c.check(ref_y_true is not None, "missing y_true.npy")

    n_test = None
    for rnd in done:
        tag = f"round {rnd}"
        m = json.loads((run_dir / "metrics" / f"round_{rnd:03d}.json").read_text())
        cm = np.load(run_dir / "confusion" / f"round_{rnd:03d}.npz")
        cm_p, cm_c = cm["proto"], cm["clf"]
        c.check(cm_p.shape == (n_cli, K, K), f"{tag}: confusion shape {cm_p.shape}")

        if n_test is None:
            n_test = int(cm_p[0].sum())
        for cid in range(n_cli):
            c.check(int(cm_p[cid].sum()) == n_test,
                    f"{tag}: client {cid} proto confusion covers {int(cm_p[cid].sum())} != {n_test}")
            c.check(int(cm_c[cid].sum()) == n_test,
                    f"{tag}: client {cid} clf confusion covers {int(cm_c[cid].sum())} != {n_test}")
        # every client's true-class marginal must be the SAME (one fixed global test set)
        c.check(bool((cm_p.sum(2) == cm_p.sum(2)[0]).all()),
                f"{tag}: clients disagree on the test set's class support")
        if ref_y_true is not None:
            c.check(bool((cm_p[0].sum(1) == np.bincount(ref_y_true, minlength=K)).all()),
                    f"{tag}: confusion support != bincount(y_true)")

        # metrics recomputed from the stored confusion must equal the stored metrics
        for rule, arr in (("proto", cm_p), ("clf", cm_c)):
            expected_agg = aggregate_clients(arr)
            for key in ("mean_over_clients", "std_over_clients", "min_over_clients",
                        "max_over_clients", "pooled"):
                c.check(all(np.isfinite(m["aggregate"][rule][key][k]) and
                            abs(expected_agg[key][k] - m["aggregate"][rule][key][k]) < 1e-9
                            for k in METRIC_KEYS), f"{tag}/{rule}: invalid aggregate {key}")
            c.check(m["per_class"][rule] == per_class_from_confusion(arr.sum(0), cfg["class_names"]),
                    f"{tag}/{rule}: per-class report mismatch")
            per = m["per_client"][rule]
            for cid in range(n_cli):
                got = metrics_from_confusion(arr[cid])
                bad = [k for k in METRIC_KEYS if not np.isfinite(per[cid][k]) or abs(got[k] - per[cid][k]) > 1e-9]
                c.check(not bad, f"{tag}/{rule}: client {cid} metrics {bad} disagree with confusion")
            mean = {k: float(np.mean([p[k] for p in per])) for k in METRIC_KEYS}
            bad = [k for k in METRIC_KEYS
                   if abs(mean[k] - m["aggregate"][rule]["mean_over_clients"][k]) > 1e-9]
            c.check(not bad, f"{tag}/{rule}: mean_over_clients {bad} is not the mean of per_client")
            pooled = metrics_from_confusion(arr.sum(0))
            bad = [k for k in METRIC_KEYS if abs(pooled[k] - m["aggregate"][rule]["pooled"][k]) > 1e-9]
            c.check(not bad, f"{tag}/{rule}: pooled {bad} disagrees with the summed confusion")
            # the single-label collapse identity
            a = m["aggregate"][rule]["pooled"]["accuracy"]
            for k in ("precision_micro", "recall_micro", "f1_micro", "recall_weighted"):
                c.check(abs(m["aggregate"][rule]["pooled"][k] - a) < 1e-9,
                        f"{tag}/{rule}: collapse identity broken at {k}")
            # history.csv is derived from this JSON
            for k in METRIC_KEYS:
                c.check(abs(float(hist[rnd][f"{rule}_{k}"])
                            - m["aggregate"][rule]["mean_over_clients"][k]) < 1e-6,
                        f"{tag}/{rule}: history.csv {k} disagrees with round JSON")

        # ---- weights -------------------------------------------------------
        w = torch.load(run_dir / "weights" / f"round_{rnd:03d}.pt",
                       map_location="cpu", weights_only=True)
        c.check(list(w["client_ids"]) == list(range(n_cli)), f"{tag}: weight client_ids wrong")
        c.check(w["params"].shape == (n_cli, cfg["packer_manifest"]["n_params"]),
                f"{tag}: params shape {tuple(w['params'].shape)}")
        c.check(bool(torch.isfinite(w["params"]).all()), f"{tag}: non-finite parameter")
        c.check(bool(torch.isfinite(w["buffers"]).all()), f"{tag}: non-finite BatchNorm buffer")
        c.check(bool((w["buffers"][:, :] == w["buffers"][:, :]).all()), f"{tag}: NaN buffer")

        # ---- prototypes ----------------------------------------------------
        pr = torch.load(run_dir / "protos" / f"round_{rnd:03d}.pt",
                        map_location="cpu", weights_only=True)
        cnt = pr["counts"].numpy().astype(np.int64)
        c.check(bool((cnt == counts_cfg).all()),
                f"{tag}: prototype counts n_ij disagree with the decoded class counts")
        loc = pr["local"].numpy()
        absent = cnt == 0
        c.check(float(np.abs(loc[absent]).max(initial=0.0)) == 0.0,
                f"{tag}: a class with n_ij = 0 has a non-zero local prototype")
        c.check(bool(np.isfinite(loc).all()), f"{tag}: non-finite local prototype")
        c.check(float(loc.min()) >= 0.0,
                f"{tag}: local prototype has a negative entry (features are post-ReLU)")

        # chained state: recompute Eq. (10) from the stored local prototypes and counts
        g = np.zeros((K, s), dtype=np.float64)
        nj = np.zeros(K, dtype=np.int64)
        for cid in range(n_cli):
            comp = compress(loc[cid].astype(np.float64), masks)
            present = cnt[cid] > 0
            g[present] += comp[present] * cnt[cid][present, None]
            nj += present.astype(np.int64)
        nz = nj > 0
        g[nz] /= nj[nz][:, None]
        stored = pr["global_compressed"].numpy().astype(np.float64)
        rel = np.abs(g - stored).max() / max(np.abs(stored).max(), 1e-12)
        c.check(rel < 1e-5,
                f"{tag}: recomputed global prototype differs from the stored one (rel {rel:.2e})")
        c.check(bool((pr["n_clients_per_class"].numpy() == nj).all()),
                f"{tag}: |N_j| disagrees with the counts")
        sparse = pr["global_sparse"].numpy()
        c.check(np.allclose(sparse, decompress(stored, masks), rtol=1e-6, atol=1e-6),
                f"{tag}: sparse and compressed global prototypes disagree")
        c.check(float(np.abs(sparse * (1 - masks)).max(initial=0.0)) == 0.0,
                f"{tag}: global sparse prototype is non-zero outside its mask")

        # ---- digests: metrics JSON vs the tensors it was computed from ------
        dg = m.get("digests")
        if dg:
            for name, got in (("local_protos", digest(pr["local"])),
                              ("proto_counts", digest(pr["counts"])),
                              ("params", digest(w["params"])),
                              ("buffers", digest(w["buffers"])),
                              ("int_buffers", digest(w["int_buffers"])),
                              ("confusion_proto", digest(cm_p)),
                              ("confusion_clf", digest(cm_c))):
                c.check(dg.get(name) == got,
                        f"{tag}: {name} digest {got[:12]} != recorded {str(dg.get(name))[:12]} "
                        "-- the stored tensor is not the one the metrics describe")

        # ---- saved predictions must rebuild the confusion exactly -----------
        for rule, arr in (("proto", cm_p), ("clf", cm_c)):
            pp = run_dir / "preds" / f"round_{rnd:03d}_{rule}.npy"
            if rnd in cfg.get("save_preds_rounds", []):
                c.check(pp.exists(), f"{tag}/{rule}: required predictions missing")
            if pp.exists() and ref_y_true is not None:
                yp = np.load(pp, mmap_mode="r")
                c.check(yp.shape == (n_cli, len(ref_y_true)),
                        f"{tag}/{rule}: predictions shape {yp.shape}")
                yt = ref_y_true.astype(np.int64)
                for cid in range(n_cli):
                    rebuilt = np.bincount(yt * K + yp[cid].astype(np.int64),
                                          minlength=K * K).reshape(K, K)
                    c.check(bool((rebuilt == arr[cid]).all()),
                            f"{tag}/{rule}: client {cid} predictions do not rebuild the confusion")

        # ---- client log ----------------------------------------------------
        with (run_dir / "client_log" / f"round_{rnd:03d}.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        ids = sorted(int(r["client_id"]) for r in rows)
        c.check(ids == list(range(n_cli)), f"{tag}: client log ids {ids[:5]}... != 0..{n_cli-1}")
        for r in rows:
            cid = int(r["client_id"])
            c.check(int(r["rows"]) == int(counts_cfg[cid].sum()),
                    f"{tag}: client {cid} trained on {r['rows']} rows, data says "
                    f"{int(counts_cfg[cid].sum())}")
            import math
            want = math.ceil(int(r["rows"]) / cfg["batch"])
            c.check(int(r["applied_steps"]) > 0 and
                    int(r["applied_steps"]) + int(r["skipped_steps"]) == int(r["steps"]),
                    f"{tag}: client {cid} invalid applied/skipped steps")
            c.check(int(r["steps"]) == want,
                    f"{tag}: client {cid} took {r['steps']} steps, ceil(rows/batch) = {want}")
        c.check(abs(float(hist[rnd]["lr"]) - cfg["lr"]) < 1e-12,
                f"{tag}: logged lr != config lr (no scheduler is configured)")
        if rnd == 1:
            c.check(abs(float(hist[rnd]["reg_loss"])) == 0.0,
                    "round 1 must train without the prototype regularizer")
        else:
            c.check(int(m["global_proto_classes"]) > 0,
                    f"{tag}: no global prototype was available")

    # ---- resume state ------------------------------------------------------
    last = done[-1]
    rb = torch.load(run_dir / "resume" / f"round_{last:03d}.pt",
                    map_location="cpu", weights_only=True)
    c.check(int(rb["round"]) == last, "resume blob round != last marker")
    c.check(rb["fingerprint"] == cfg["fingerprint"], "resume fingerprint != config fingerprint")
    expected_mu, _ = resolve_mu(cfg["mu_kind"], cfg["mu_value"], counts_cfg, K)
    c.check(np.allclose(rb["mu"], expected_mu, rtol=1e-12, atol=0), "resume mu mismatch")
    pr = torch.load(run_dir / "protos" / f"round_{last:03d}.pt",
                    map_location="cpu", weights_only=True)
    c.check(torch.allclose(rb["global_sparse"], pr["global_sparse"]),
            "resume global prototype disagrees with the committed one")
    shards = sorted((run_dir / "resume").glob(f"round_{last:03d}.w*.pt"))
    c.check(len(shards) == len(cfg["assignment"]),
            f"{len(shards)} optimizer shards for {len(cfg['assignment'])} workers")
    for rank, shard in enumerate(shards):
        opt = torch.load(shard, map_location="cpu", weights_only=True)
        c.check(opt["client_ids"] == cfg["assignment"][rank], "optimizer shard ownership mismatch")
        c.check("rng" in opt, "worker RNG missing")
    stale = [p.name for p in (run_dir / "resume").glob("round_*.pt")
             if not p.name.startswith(f"round_{last:03d}")]
    c.check(not stale, f"superseded resume blobs were not pruned: {stale[:4]}")

    # ---- strict model reconstruction from the saved weights ----------------
    if deep_weights:
        w = torch.load(run_dir / "weights" / f"round_{last:03d}.pt",
                       map_location="cpu", weights_only=True)
        from src.model import FlatPacker
        model = build_model(cfg["n_features"], cfg["model_cfg"])
        packer = FlatPacker(model)
        c.check(packer.manifest() == cfg["packer_manifest"],
                "the flat layout in config.json does not match the current model")
        for cid in (0, n_cli // 2, n_cli - 1):
            sd = packer.to_state_dict(w["params"][cid], w["buffers"][cid], w["int_buffers"][cid])
            fresh = build_model(cfg["n_features"], cfg["model_cfg"])
            try:
                fresh.load_state_dict(sd, strict=True)
                ok = True
            except Exception as e:                       # noqa: BLE001
                ok = False
                c.fail.append(f"client {cid}: strict load failed: {e}")
            if ok:
                c.check(True, "")
                c.check(all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point()),
                        f"client {cid}: non-finite tensor in the reconstructed state_dict")

    return c.report()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--no-deep-weights", action="store_true")
    a = ap.parse_args()
    sys.exit(verify(Path(a.run_dir), a.require_complete, not a.no_deep_weights))

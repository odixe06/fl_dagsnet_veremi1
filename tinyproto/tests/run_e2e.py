"""End-to-end TinyProto-FP run on the fixture: two workers, real commit contract, real metrics.

Usage:
    python tests/run_e2e.py --out /tmp/e2e --rounds 3 [--crash-at 2] [--require-resume]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ckpt import file_digest  # noqa: E402
from src.data import find_client_root, find_test_root, footer_fingerprint, plan_gpus  # noqa: E402
from src.driver import run  # noqa: E402
from src.model import CFG as MODEL_CFG, FlatPacker, build_model  # noqa: E402


def make_cfg(fixture: Path, n_clients: int, rounds: int, devices: list[str],
             *, batch=256, eval_batch=8192, mu_kind="inv_mean_nij", mu_value=1.0,
             cps_s=50, compile_=False, run_name="e2e", extra=None) -> tuple[dict, dict]:
    meta = json.loads((ROOT / "knowledge" / "meta.json").read_text())
    root = find_client_root([fixture], n_clients)
    test_dir = find_test_root([fixture])
    rows = {}
    import pyarrow.parquet as pq
    for cid in range(n_clients):
        fs = sorted((root / "train" / f"client_id={cid:03d}").glob("*.parquet"))
        rows[cid] = sum(pq.ParquetFile(f).metadata.num_rows for f in fs)
    assignment = plan_gpus(rows, len(devices))
    model = build_model(66, MODEL_CFG)
    cfg = {
        "run_name": run_name, "scenario": f"{n_clients}client",
        "n_clients": n_clients, "num_classes": 16, "n_features": 66, "feature_dim": 256,
        "class_names": meta["class_names"], "feature_cols": meta["feature_cols"],
        "model_cfg": MODEL_CFG, "packer_manifest": FlatPacker(model).manifest(),
        "cps_s": cps_s, "mask_seed": 42,
        "rounds": rounds, "local_epochs": 1, "batch": batch, "eval_batch": eval_batch,
        "lr": 1e-3, "weight_decay": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
        "clip": 1.0, "lam": 1.0, "amp": True, "compile": compile_, "eval_group": 1,
        "mu_kind": mu_kind, "mu_value": mu_value,
        "seed": 42, "devices": devices, "assignment": assignment,
        "max_hours": 24.0, "save_preds_rounds": [rounds],
        "data_fingerprint": footer_fingerprint(sorted((root / "train").rglob("*.parquet"))),
        # part of the mu-selection signature, exactly as the notebook config cell sets it
        "scaler_fingerprint": file_digest(ROOT / "knowledge" / "scaler.json"),
        "test_fingerprint": footer_fingerprint(sorted(test_dir.glob("*.parquet"))),
    }
    cfg.update(extra or {})
    paths = {"client_root": str(root), "test_dir": str(test_dir),
             "scaler": str(ROOT / "knowledge" / "scaler.json")}
    return cfg, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="/tmp/tinyproto_fixture")
    ap.add_argument("--out", default="/tmp/tinyproto_e2e")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run-name", default="e2e")
    ap.add_argument("--require-resume", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--session-rounds", type=int, default=0)
    ap.add_argument("--cps-s", type=int, default=50)
    args = ap.parse_args()

    imports = os.environ.get("TINYPROTO_IMPORT_ROOTS", "")
    cfg, paths = make_cfg(Path(args.fixture), args.clients, args.rounds,
                          [args.device] * args.workers, compile_=args.compile,
                          run_name=args.run_name, cps_s=args.cps_s,
                          extra={"amp": args.device != "cpu", "require_resume": args.require_resume,
                                 "rounds_this_session": args.session_rounds or None,
                                 "import_roots": [p for p in imports.split(":") if p]})
    res = run(cfg, paths, Path(args.out))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

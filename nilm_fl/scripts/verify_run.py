#!/usr/bin/env python
"""Verify a downloaded run directory offline: per-client and proxy metrics == confusion ==
weights == history/clients CSV, predictions rebuild the matrices, logs close.

    python scripts/verify_run.py papers/nilm-li-2024/runs/pulls/20c/runs/nilm_20c --require-rounds 50

Reads only artifacts. y_true defaults to the run's own reports/y_true.u8.npy.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papers/nilm-li-2024"))
from proj.verify import verify_run
from proj.model import build_model, N_PARAMS

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--y-true", type=Path, default=None)
ap.add_argument("--require-rounds", type=int, default=None)
ap.add_argument("--no-rebuild", action="store_true", help="skip rebuilding the models")
a = ap.parse_args()

# The run's own manifest carries the cfg it trained under, so the fingerprint is checked
# against what the run declared rather than against whatever CFG happens to be current.
mf = a.run_dir / "reports" / "manifest.json"
cfg = json.loads(mf.read_text())["cfg"] if mf.is_file() else None
if cfg is None:
    print("  no reports/manifest.json: fingerprints will not be checked")
yt = a.y_true or (a.run_dir / "reports" / "y_true.u8.npy")
ok, lines = verify_run(a.run_dir, cfg=cfg,
                       build_model=None if a.no_rebuild else build_model,
                       expect_params=None if a.no_rebuild else N_PARAMS,
                       y_true_path=yt, require_rounds=a.require_rounds)
print("\n".join(lines))
sys.exit(0 if ok else 1)

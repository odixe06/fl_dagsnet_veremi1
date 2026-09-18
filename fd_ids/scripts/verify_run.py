#!/usr/bin/env python
"""Verify a downloaded run directory offline: metrics == confusion == weights == history.

    python scripts/verify_run.py papers/fd-ids-2025/runs/fdids_20c --require-rounds 50

Reads only artifacts. Pass --y-true to also rebuild every confusion matrix from the stored
predictions; without it, rounds with predictions are reported but not cross-checked.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papers/fd-ids-2025"))
from proj.verify import verify_run
from proj.model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--y-true", type=Path, default=None, help="test_y.u8.npy from the prepack")
ap.add_argument("--require-rounds", type=int, default=None)
ap.add_argument("--no-rebuild", action="store_true", help="skip rebuilding the model")
a = ap.parse_args()

# The run's own manifest carries the cfg it trained under, so the fingerprint is checked
# against what the run declared rather than against whatever CFG happens to be current.
mf = a.run_dir / "reports" / "manifest.json"
cfg = json.loads(mf.read_text())["cfg"] if mf.is_file() else None
if cfg is None:
    print("  no reports/manifest.json: fingerprints will not be checked")

ok, lines = verify_run(a.run_dir, cfg=cfg,
                       build=None if a.no_rebuild else build_model,
                       expect_params=None if a.no_rebuild else 395_024,
                       y_true_path=a.y_true, require_rounds=a.require_rounds)
print("\n".join(lines))
sys.exit(0 if ok else 1)

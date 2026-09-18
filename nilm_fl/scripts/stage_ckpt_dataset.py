#!/usr/bin/env python
"""Stage a pulled (or merged) run tree as a Kaggle checkpoint dataset for the cross-account
resume route.

    # full tree (every round; the route measured on pFedES)
    python scripts/stage_ckpt_dataset.py papers/nilm-li-2024/runs/pulls/20c_s1/runs/nilm_20c \\
        --owner minhtrit06 --slug nilm-20c-ckpt-s1 --out papers/nilm-li-2024/runs/ckpt_ds/20c_s1
    # handoff bundle: the LAST round only + reports/handoff.json (sha chain 1..r)
    python scripts/stage_ckpt_dataset.py <run_dir> --owner A --slug S --out DIR --last-only

Layout is <out>/runs/<run_name>/... on purpose: `kaggle datasets create -r zip -t` zips each
top-level directory and Kaggle unpacks the zip's CONTENTS at the dataset root, so the extra
`runs/` level is what keeps `<run_name>` in the mount path (ckpt._attached_source filters on it).
Files are hard-linked, not copied. --last-only verifies the whole chain 1..r locally first
(proj.ckpt.handoff_record) and records it; the next session imports round r alone and checks
its bytes and its prev_sha against that record. Then:
    kaggle_as.py <owner> -- kaggle datasets create -p <out> -r zip -t
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/nilm-li-2024"))
from proj import ckpt as C

SUBDIRS = ("weights", "resume", "metrics", "confusion", "preds", "logs", "complete", "reports")
TOP = ("history.csv", "clients.csv")

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--owner", required=True)
ap.add_argument("--slug", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--last-only", action="store_true",
                help="handoff bundle: last verified round + sha chain instead of the whole tree")
a = ap.parse_args()

dst = a.out / "runs" / a.run_dir.name
if dst.exists():
    sys.exit(f"{dst} exists; remove it first")
manifest = json.loads((a.run_dir / "reports" / "manifest.json").read_text())
fp = manifest["fingerprint"]
last = C.last_complete_round(a.run_dir, fp)
if last is None:
    sys.exit(f"{a.run_dir}: no verified round")

def link(f, t):
    t.parent.mkdir(parents=True, exist_ok=True)
    os.link(f, t)

n = 0
if a.last_only:
    rec = C.handoff_record(a.run_dir, last, fp, extra={
        "run_name": a.run_dir.name, "data_id": manifest.get("data_id"),
        "content_id": manifest.get("content_id"), "source": str(a.run_dir.resolve()),
        "exported": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    for sub in C.RESUME_SUBDIRS + ("complete",):
        for f in sorted((a.run_dir / sub).glob(f"*_{last:03d}.*")):
            link(f, dst / sub / f.name); n += 1
    link(a.run_dir / "reports" / "manifest.json", dst / "reports" / "manifest.json"); n += 1
    (dst / C.HANDOFF).write_text(json.dumps(rec, indent=1)); n += 1
else:
    for sub in SUBDIRS:
        for f in sorted((a.run_dir / sub).iterdir()) if (a.run_dir / sub).is_dir() else []:
            if f.is_file(): link(f, dst / sub / f.name); n += 1
    for name in TOP:
        if (a.run_dir / name).is_file(): link(a.run_dir / name, dst / name); n += 1
(a.out / "dataset-metadata.json").write_text(json.dumps({
    "title": a.slug, "id": f"{a.owner}/{a.slug}", "licenses": [{"name": "CC0-1.0"}]}, indent=2))
size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
print(f"{n} files, {size/1e6:.0f} MB -> {dst}  ({a.owner}/{a.slug}, "
      f"{'handoff round ' + str(last) if a.last_only else 'full tree 1..' + str(last)})")

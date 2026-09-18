#!/usr/bin/env python
"""CPU-only Kaggle probe: does a checkpoint dataset mount and import as the production
resume gate (proj/ckpt.py, embedded byte-for-byte) will see it, up to the expected round?

    python scripts/gen_ckpt_probe.py --owner minhtrit06 --dataset minhtrit06/nilm-20c-ckpt-s1 \
        --run-name nilm_20c --expect-round 33 --out papers/nilm-li-2024/notebook/20c_ckpt_probe

Costs no GPU quota. Push with `kaggle_as.py <owner> -- kaggle kernels push -p <out>`; the kernel
ends COMPLETE only if the fingerprint of the mounted manifest matches and exactly
--expect-round rounds import and verify (skill reference multi-account.md, "Probe the gate").
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "papers/nilm-li-2024"
CKPT = (ROOT / "proj/ckpt.py").read_text()

ap = argparse.ArgumentParser()
ap.add_argument("--owner", required=True)
ap.add_argument("--dataset", required=True, help="owner/slug of the checkpoint dataset")
ap.add_argument("--run-name", required=True)
ap.add_argument("--expect-round", type=int, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

slug = f"{a.run_name.replace('_', '-')}-ckpt-probe"
cells = [
    {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
     "source": "import os\nos.makedirs('/kaggle/working/proj', exist_ok=True)\n"
               "open('/kaggle/working/proj/__init__.py', 'w').close()\n"
               "open('/kaggle/working/proj/ckpt.py', 'w').write(" + repr(CKPT) + ")\n"},
    {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
     "source": f'''import json, sys, shutil
from pathlib import Path
sys.path.insert(0, "/kaggle/working")
import proj.ckpt as C
RUN, EXPECT = {a.run_name!r}, {a.expect_round}
inp = Path("/kaggle/input")
marks = sorted(p for p in inp.rglob("complete/round_*.done") if RUN in p.parts)
print("markers:", len(marks), "under", sorted({{str(p.parent.parent) for p in marks}}))
mf = [p for p in inp.rglob("reports/manifest.json") if RUN in p.parts]
assert len(mf) == 1, mf
m = json.loads(mf[0].read_text())
fp = C.fingerprint(m["cfg"])
print("fingerprint mounted", m["fingerprint"], "recomputed", fp)
assert fp == m["fingerprint"], "fingerprint mismatch: manifest cfg does not hash to its own fingerprint"
last = C.resolve_resume(RUN, m["cfg"])      # the production gate, same code, CPU
print("last verified round:", last)
assert last == EXPECT, f"expected {{EXPECT}}, got {{last}}"
shutil.rmtree("/kaggle/working/runs", ignore_errors=True)   # do not commit GBs of copies
print("PROBE_OK", RUN, last)
'''},
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                   "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "probe.ipynb").write_text(json.dumps(nb, indent=1))
(a.out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{a.owner}/{slug}", "title": slug, "code_file": "probe.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": False,
    "dataset_sources": [a.dataset], "kernel_sources": [], "competition_sources": []}, indent=2))
print(f"{a.out}/probe.ipynb -> {a.owner}/{slug} (dataset {a.dataset}, expect round {a.expect_round})")

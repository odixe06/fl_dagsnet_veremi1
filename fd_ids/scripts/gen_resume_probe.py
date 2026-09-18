#!/usr/bin/env python
"""Generate a CPU probe that runs the real resume gate against a checkpoint dataset.

`kernel_sources` does not cross accounts: Kaggle prints one warning, reports the push as
successful and starts the kernel with an empty /kaggle/input (multi-account.md, measured
2026-09-09). The cross-account route is the previous session's run tree re-uploaded as a
dataset owned by the account that will run. Whether that mount resolves -- the right
directory level, no tabular rewrite of history.csv, every round's four artifacts readable --
is a question a CPU kernel answers for zero GPU quota, before the GPU chain reaches it.

The probe calls `proj.ckpt.resolve_resume` unmodified with the cfg the run declared in its
own manifest, so the import, the fingerprint check and `round_ok` on every round are the
production code path. It then asserts the verified round is exactly the one the previous
session committed: an import that silently stops short would make the GPU session RETRAIN
those rounds under the same run name, and W&B would drop the non-monotonic steps without a
word.

    python scripts/gen_resume_probe.py --owner minhtriethihi --clients 20 --expect 42 \\
        --dataset-source minhtriethihi/fdids-20c-v2-ckpt --run-tag _v2

Never hand-edit the .ipynb; edit here and regenerate.
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "papers/fd-ids-2025/proj"
RUNTIME = json.loads((ROOT / "knowledge/runtime.json").read_text())


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "source": s,
                     "execution_count": None, "outputs": []}


def build(K, owner, run_tag, expect, dataset_sources):
    run_name = f"fdids_{K}c{run_tag}"
    title = f"FD-IDS resume probe {K}c"
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    cells = [md(f"""# Resume probe — {run_name}

CPU only. Mounts the checkpoint dataset and runs the production resume gate
(`proj/ckpt.py::resolve_resume`) against it. Must end with `RESUME PROBE OK` and
`last verified round: {expect}`; anything else means the GPU continuation would either die
at its gate or, worse, retrain rounds it should have imported.
"""),
             code("import os\nos.makedirs('/kaggle/working/proj', exist_ok=True)\n"
                  "open('/kaggle/working/proj/__init__.py', 'w').close()"),
             code("%%writefile /kaggle/working/proj/ckpt.py\n"
                  + (PROJ / "ckpt.py").read_text().rstrip()),
             code(f'''import csv, json, sys
from pathlib import Path
sys.path.insert(0, "/kaggle/working")
from proj import ckpt as C

RUN, EXPECT = "{run_name}", {expect}
trees = sorted({{p.parent.parent for p in Path("/kaggle/input").rglob("complete/round_*.done")}})
print("resume trees under /kaggle/input:", trees)
src = C._attached_source(RUN)
print("attached source          :", src)
assert src is not None, "no resume tree carries the run name: wrong mount layout"
cfg = json.loads((src / "reports" / "manifest.json").read_text())["cfg"]
assert cfg["run_name"] == RUN, cfg["run_name"]
print("fingerprint (manifest cfg):", C.fingerprint(cfg))
n_mark = len(list((src / "complete").glob("round_*.done")))
print("markers at source        :", n_mark)

last = C.resolve_resume(RUN, cfg)
print("last verified round      :", last)
d = C.run_dir(RUN)
rows = list(csv.DictReader(open(d / "history.csv")))
print("rebuilt history rows     :", len(rows), "rounds", rows[0]["round"], "..", rows[-1]["round"])
for sub in C.RESUME_SUBDIRS:
    print(f"  imported {{sub:9s}}: {{len(list((d / sub).iterdir()))}} files")
assert last == EXPECT == n_mark, f"expected {{EXPECT}} (markers {{n_mark}}), verified {{last}}"
assert len(rows) == EXPECT and int(rows[-1]["round"]) == EXPECT
print("RESUME PROBE OK")''')]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {"accelerator": "none", "isGpuEnabled": False,
                   "isInternetEnabled": False, "language": "python",
                   "sourceType": "notebook"}}}
    for c in nb["cells"]:
        c["source"] = [l + "\n" for l in c["source"].split("\n")]
    # Same pinned image as the run that wrote the checkpoints, so torch.load reads them
    # with the torch that saved them. No GPU, no internet: the probe needs neither.
    meta = {"id": f"{owner}/{slug}", "title": title,
            "code_file": f"resume_probe_{K}c.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_internet": False,
            "docker_image": RUNTIME["docker_image"],
            "dataset_sources": list(dataset_sources), "kernel_sources": [],
            "competition_sources": []}
    return nb, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--clients", type=int, required=True)
    ap.add_argument("--expect", type=int, required=True,
                    help="the round the previous session committed last")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--dataset-source", action="append", required=True)
    a = ap.parse_args()
    nb, meta = build(a.clients, a.owner, a.run_tag, a.expect, a.dataset_source)
    out = ROOT / "papers/fd-ids-2025/notebook" / f"resume_probe_{a.clients}c"
    out.mkdir(parents=True, exist_ok=True)
    (out / meta["code_file"]).write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{a.clients:>4}c -> {out}/{meta['code_file']}  (expect round {a.expect})")

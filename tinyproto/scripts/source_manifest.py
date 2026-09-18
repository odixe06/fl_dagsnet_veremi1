#!/usr/bin/env python
"""Pin the exact code, runtime and mu a production run was started with.

The config fingerprint covers the SCIENCE (architecture, data, hyper-parameters, mu) but not the
code that implements it. Editing `src/` and regenerating between two sessions of the same run
keeps the fingerprint identical and passes the validator, while the algorithm has changed and
the second half of the run no longer matches the first. This manifest closes that gap: write it
once when a run starts, verify it before every later session.

    python scripts/source_manifest.py write   [--manifest papers/tinyproto-lee-2026/production_manifest.json]
    python scripts/source_manifest.py verify  [--manifest ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(p.name for p in (ROOT / "src").glob("*.py"))
# Criterion: pin what can change the ANSWER to "is session N running the same algorithm as
# session 1?". gen_notebooks.py qualifies -- it embeds src/ into the next session's notebook and
# builds its CFG. gen_production.py does NOT: it only chooses owner, max_hours and the session
# chain, none of which enters the fingerprint or the algorithm, and the review explicitly allows
# re-deriving the budget between sessions. Pinning it would turn every legitimate budget edit
# into a false algorithm-drift alarm. validate_notebooks.py cannot change what a pushed notebook
# does either; weakening it is caught by the regression suite, not by this manifest.
EXTRA = ["scripts/gen_notebooks.py"]
MU = ["papers/tinyproto-lee-2026/mu-datasets/tinyproto-fp-mu-%sclient/mu_sweep.json" % n
      for n in (20, 50, 100)]


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def collect() -> dict:
    files = {f"src/{m}": digest(ROOT / "src" / m) for m in MODULES}
    for rel in EXTRA + MU:
        p = ROOT / rel
        if p.exists():
            files[rel] = digest(p)
    runtime = json.loads((ROOT / "knowledge" / "runtime.json").read_text())
    return {"files": files, "docker_image": runtime["docker_image"],
            "n_modules": len(MODULES)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["write", "verify"])
    ap.add_argument("--manifest", default=str(ROOT / "papers/tinyproto-lee-2026/production_manifest.json"))
    a = ap.parse_args()
    path, now = Path(a.manifest), collect()
    if a.mode == "write":
        path.write_text(json.dumps(now, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path} ({len(now['files'])} files, image {now['docker_image'][:23]}…)")
        return 0
    if not path.exists():
        print(f"NO MANIFEST at {path}; write one before the first production session")
        return 1
    was = json.loads(path.read_text())
    bad = []
    if was.get("docker_image") != now["docker_image"]:
        bad.append(f"docker_image changed:\n  was {was.get('docker_image')}\n  now {now['docker_image']}")
    for rel in sorted(set(was["files"]) | set(now["files"])):
        o, n = was["files"].get(rel), now["files"].get(rel)
        if o != n:
            bad.append(f"{rel}: {'added' if o is None else 'removed' if n is None else 'CHANGED'}")
    if bad:
        print("PRODUCTION MANIFEST MISMATCH -- the run was started with different code:")
        for b in bad:
            print("  " + b)
        print("\nA later session must run the SAME algorithm as the earlier ones. Restore the\n"
              "pinned source, or start a new run with a new run_name.")
        return 1
    print(f"manifest OK: {len(now['files'])} files and the runtime image match the pinned run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

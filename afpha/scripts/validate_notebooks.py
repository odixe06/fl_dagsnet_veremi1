#!/usr/bin/env python
"""Contract checks on the generated notebooks, before any push.

Usage: conda run -n nckh python scripts/validate_notebooks.py

A notebook is the only thing Kaggle actually executes, and the failures that hurt most
are the ones that cost a version or a run rather than raising locally: a missing
kernelspec (papermill refuses to start), a title whose slug does not match the metadata
id (the notebook lands at a URL the status calls cannot address), an embedded module that
has drifted from `src/` (Kaggle runs code that was never tested), or a live W&B key in a
notebook that is not private.

Only booleans about the credential are printed, never the credential.
Exit code 0 means every check passed.
"""
import json
import netrc
import os
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OWNER = "minhtran0601"
EXPECT = {"20": {"batch": 512, "max_hours": 6.0},
          "50": {"batch": 512, "max_hours": 6.0},
          "100": {"batch": 256, "max_hours": 9.5}}
ROUNDS = 50


def netrc_key():
    try:
        auth = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai")
    except (OSError, netrc.NetrcParseError):
        return None
    return auth[2] if auth else None


def slugify(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def main():
    key = netrc_key()
    fails = []

    for stage in sorted((ROOT / "notebooks").iterdir()):
        if not stage.is_dir():
            continue
        slug = stage.name
        nb = nbf.read(stage / f"{slug}.ipynb", as_version=4)
        meta = json.loads((stage / "kernel-metadata.json").read_text())
        try:
            nbf.validate(nb)
        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            fails.append(f"{slug}: nbformat invalid: {exc}")

        if not nb.metadata.get("kernelspec", {}).get("name"):
            fails.append(f"{slug}: no metadata.kernelspec -- papermill will refuse to start")
        if meta["id"] != f"{OWNER}/{slug}":
            fails.append(f"{slug}: metadata id {meta['id']} does not match the staging dir")
        if slugify(meta["title"]) != slug:
            fails.append(f"{slug}: title slugifies to {slugify(meta['title'])}, not {slug}")
        for field, want in (("is_private", True), ("enable_gpu", True),
                            ("enable_internet", True), ("machine_shape", "NvidiaTeslaT4")):
            if meta.get(field) != want:
                fails.append(f"{slug}: {field}={meta.get(field)!r}, expected {want!r}")

        src = "".join("".join(c["source"]) for c in nb.cells)
        is_train = "train-" in slug
        has_key = bool(key) and key in src
        if "__WANDB_KEY__" in src:
            fails.append(f"{slug}: the W&B placeholder was never replaced")
        if has_key and meta.get("is_private") is not True:
            fails.append(f"{slug}: carries a live W&B key but is not private")
        if is_train and not has_key:
            fails.append(f"{slug}: training notebook has no W&B credential")

        max_hours = None
        if is_train:
            n = slug.split("train-")[1].replace("client", "")
            want = EXPECT[n]
            for name, expect in (("BATCH", want["batch"]), ("ROUNDS", ROUNDS),
                                 ("MAX_HOURS", want["max_hours"])):
                m = re.search(rf"^{name}\s*=\s*([\d.]+)", src, re.M)
                if m is None:
                    fails.append(f"{slug}: no {name} assignment")
                    continue
                got = float(m.group(1))
                if got != float(expect):
                    fails.append(f"{slug}: {name}={got}, expected {expect}")
                if name == "MAX_HOURS":
                    max_hours = got
        print(f"{slug:34s} kernelspec ok  key={'yes' if has_key else 'no '}  "
              f"max_hours={max_hours}")

    # Every embedded module must be byte-identical to the file it was generated from,
    # otherwise Kaggle runs code this repo has never tested.
    n_files = 0
    for stage in sorted((ROOT / "notebooks").iterdir()):
        if not stage.is_dir():
            continue
        nb = nbf.read(stage / f"{stage.name}.ipynb", as_version=4)
        payload = None
        for cell in nb.cells:
            body = "".join(cell["source"])
            if body.startswith("MODULES = "):
                payload = json.loads(body[len("MODULES = "):])
        if payload is None:
            fails.append(f"{stage.name}: no MODULES cell")
            continue
        for name, text in payload.items():
            origin = ROOT / "src" / name
            if not origin.exists():
                origin = ROOT / "architecture" / name
            if not origin.exists():
                fails.append(f"{stage.name}: embedded {name} has no source counterpart")
            elif origin.read_text() != text:
                fails.append(f"{stage.name}: embedded {name} differs from "
                             f"{origin.relative_to(ROOT)} -- rebuild the notebooks")
            else:
                n_files += 1
        for f in sorted((ROOT / "src").glob("*.py")):
            if f.name not in payload:
                fails.append(f"{stage.name}: src/{f.name} is not embedded")

    print(f"\n{n_files} embedded files byte-identical to their source")
    for f in fails:
        print("FAIL  " + f)
    print("all notebook checks passed" if not fails else f"{len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

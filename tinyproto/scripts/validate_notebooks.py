#!/usr/bin/env python
"""Catch the notebook failures that cost a Kaggle version or a whole GPU session.

Every check here corresponds to a failure that actually happened in this project's history:

* a notebook without `metadata.kernelspec` never runs -- papermill raises
  `ValueError: No kernel name found` before the first cell;
* the slug comes from the TITLE, not from the metadata `id`, so a title that slugifies to
  something else creates a different kernel and every later status/output call addresses a slug
  that does not exist;
* an invalid or missing `machine_shape` is silently coerced to one P100, and Kaggle's PyTorch
  build then fails on sm_60 with a message that never mentions accelerators;
* an embedded module that has drifted from `src/` means the notebook is not running the code
  that was tested;
* a live credential in a notebook that is not private.

    python scripts/validate_notebooks.py [DIR]
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT = ROOT / "papers" / "tinyproto-lee-2026" / "notebook"
SECRET = re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[=:]\s*['\"][A-Za-z0-9_\-]{16,}")


def slugify(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def validate(d: Path, fails: list[str]) -> None:
    meta_p = d / "kernel-metadata.json"
    if not meta_p.exists():
        fails.append(f"{d.name}: no kernel-metadata.json (kaggle kernels push requires that name)")
        return
    meta = json.loads(meta_p.read_text())
    nb_p = d / meta["code_file"]

    def ck(ok, msg):
        if not ok:
            fails.append(f"{d.name}: {msg}")

    ck(nb_p.exists(), f"code_file {meta['code_file']} does not exist")
    if not nb_p.exists():
        return
    ck("/" in meta["id"] and meta["id"].count("/") == 1,
       f"id {meta['id']!r} must be OWNER/SLUG, not a bare kernel name")
    owner, slug = meta["id"].split("/")
    ck(slugify(meta["title"]) == slug,
       f"title {meta['title']!r} slugifies to {slugify(meta['title'])!r}, not {slug!r} "
       "-- Kaggle would create a different kernel")
    ck(nb_p.stem == slug, f"notebook filename {nb_p.stem!r} != slug {slug!r}")
    ck(meta.get("kernel_type") == "notebook", "kernel_type must be 'notebook'")
    ck(meta.get("is_private") is True, "is_private must be explicitly true on every push")
    ck(meta.get("enable_gpu") is True, "enable_gpu must be true")
    ck(meta.get("machine_shape") == "NvidiaTeslaT4",
       f"machine_shape is {meta.get('machine_shape')!r}; anything else is silently coerced "
       "to a single P100 (sm_60), which Kaggle's PyTorch cannot use")
    ck(bool(meta.get("dataset_sources")), "no dataset_sources")

    # The runtime contract. Kaggle's rolling image has changed torch under this project, and a
    # notebook that declares the accelerator in only one of the two places has come up with zero
    # GPUs. Both are checked here so a regression fails locally instead of on a burnt session.
    runtime = json.loads((ROOT / "knowledge" / "runtime.json").read_text())
    ck(meta.get("docker_image") == runtime["docker_image"],
       f"docker_image {meta.get('docker_image')!r} != the verified image in "
       "knowledge/runtime.json; an unpinned image silently changes torch/python")

    # Production must be able to reach THIS scenario's mu, by exactly one of two routes.
    if slug.startswith("tinyproto-fp-train-"):
        # slug is tinyproto-fp-train-<n>client, optionally with a -sN session suffix
        tail = slug[len("tinyproto-fp-train-"):]
        n = tail.split("-")[0]
        m_sess = re.fullmatch(r"(\d+client)(?:-s(\d+))?", tail)
        ck(m_sess is not None, f"train slug {slug!r} is not <n>client[-sN]")
        session = int(m_sess.group(2)) if (m_sess and m_sess.group(2)) else 0
        sweep = "tinyproto-fp-mu-sweep" + ("" if n == "20client" else f"-{n}")
        ks = list(meta.get("kernel_sources") or [])
        ds = list(meta.get("dataset_sources") or [])

        # A sweep for a DIFFERENT scenario must never be attached: the config cell selects by
        # scenario, so a foreign sweep either matches nothing or silently supplies a wrong mu.
        foreign = [k for k in ks if "tinyproto-fp-mu-sweep" in k and not k.endswith("/" + sweep)]
        ck(not foreign, f"kernel_sources carries a sweep for another scenario: {foreign}")

        own_sweep = [k for k in ks if k.endswith("/" + sweep)]
        if own_sweep:                       # route A: same account, read the sweep's own output
            ck(len(own_sweep) == 1, f"the sweep is attached more than once: {own_sweep}")
        else:                               # route B: cross-account, read a shared mu dataset
            # Counting datasets is not enough: any third dataset satisfied the old check while
            # carrying no mu at all. Name the mu source explicitly for THIS scenario.
            mu_ds = [d for d in ds if d.rsplit("/", 1)[-1] == f"tinyproto-fp-mu-{n}"]
            ck(len(mu_ds) == 1,
               f"no sweep attached and dataset_sources {ds} does not carry exactly one "
               f"'tinyproto-fp-mu-{n}' dataset: production cannot reach mu for this scenario")
            other_mu = [d for d in ds
                        if "tinyproto-fp-mu-" in d and d.rsplit("/", 1)[-1] != f"tinyproto-fp-mu-{n}"]
            ck(not other_mu, f"dataset_sources carries mu for another scenario: {other_mu}")

        prev = [k for k in ks if k not in own_sweep]
        # A resume source can arrive by kernel output (same account only) or by dataset
        # (the ONLY route that works across accounts -- measured 2026-09-09: Kaggle drops a
        # cross-account kernel source at push time and mounts nothing). Both are "a previous
        # session"; the checks below must see either, and must never see both, because
        # find_import_source refuses two candidates sharing a fingerprint.
        prev_ds = [d for d in ds if d.rsplit("/", 1)[-1].startswith("tinyproto-fp-resume-")]
        ck(len(prev) <= 1,
           f"kernel_sources {ks} carries more than one resume source; find_import_source "
           "refuses to guess between candidates that share a fingerprint")
        ck(len(prev_ds) <= 1,
           f"dataset_sources {ds} carries more than one resume dataset; find_import_source "
           "refuses to guess between candidates that share a fingerprint")
        ck(not (prev and prev_ds),
           f"{slug} attaches BOTH a resume kernel source {prev} and a resume dataset {prev_ds}; "
           "that is two candidates with one fingerprint, which find_import_source rejects")
        resume_srcs = prev + prev_ds

        # --- session-chain relationships -------------------------------------------------
        # A session slug is a promise about what the push continues. Check the promise, not just
        # the shape: a -s2 that resumes nothing silently trains from round 1 again.
        if session >= 2:
            ck(bool(resume_srcs), f"{slug} is session {session} but attaches no previous "
                                  "session by either route; it would start over from round 1")
        if session == 1:
            ck(not resume_srcs,
               f"{slug} is session 1 but attaches a previous session {resume_srcs}")
        for src in prev:
            src_slug = src.split("/", 1)[-1]
            ck(src_slug != slug, f"{slug} attaches ITSELF as its resume source")
            ck(src_slug.startswith(f"tinyproto-fp-train-{n}"),
               f"resume source {src!r} is a different scenario from {slug!r}; importing it "
               "would load the wrong run or die on a fingerprint mismatch")
            if session >= 2:
                ck(src_slug.endswith(f"-s{session - 1}"),
                   f"{slug} (session {session}) must continue session {session - 1}, "
                   f"but attaches {src_slug!r}")
            ck(src.split("/", 1)[0] == owner,
               f"{slug} is owned by {owner!r} but attaches kernel output of "
               f"{src.split('/', 1)[0]!r}. Kaggle silently drops a cross-account kernel "
               "source and mounts NOTHING; use --resume-dataset instead.")
        for src in prev_ds:
            ck(src.rsplit("/", 1)[-1].startswith(f"tinyproto-fp-resume-{n}"),
               f"resume dataset {src!r} is a different scenario from {slug!r}")

        # require_resume promises a previous session was attached. Catch the mismatch here,
        # not after the session starts and dies on an empty import.
        # Parse, do not grep the raw file: .ipynb is JSON, so cell source appears with
        # escaped quotes and a naive substring check silently never matches.
        body = "\n".join(c.source for c in nbf.read(nb_p, as_version=4).cells)
        wants_resume = '"require_resume": True' in body
        ck(wants_resume == bool(resume_srcs),
           f"require_resume is {wants_resume} but a previous-session source is "
           f"{'present' if resume_srcs else 'absent'}; these must agree")

    nb = nbf.read(nb_p, as_version=4)
    ks = nb.metadata.get("kernelspec") or {}
    ck(ks.get("name") == "python3",
       "metadata.kernelspec.name must be 'python3' -- without it papermill refuses to start")
    ck(bool(nb.metadata.get("language_info")), "metadata.language_info is missing")
    kg = nb.metadata.get("kaggle") or {}
    ck(kg.get("accelerator") == "nvidiaTeslaT4" and kg.get("isGpuEnabled") is True,
       f"metadata.kaggle is {kg or 'missing'}; the accelerator must be declared in the NOTEBOOK "
       "too, not only in kernel-metadata.json -- notebooks missing this block got 0 GPUs")
    ck((nb.metadata.get("language_info") or {}).get("version", "").startswith("3.12"),
       f"language_info.version {(nb.metadata.get('language_info') or {}).get('version')!r} does "
       "not match the pinned image's python 3.12")
    ck(len(nb.cells) > 5, f"only {len(nb.cells)} cells")

    code = [c for c in nb.cells if c.cell_type == "code"]
    src_all = "\n".join(c.source for c in code)
    ck("assert n == 2" in src_all and "sm_75" in src_all,
       "the notebook does not assert 2 GPUs with sm_75 before training")
    ck("%%writefile" in src_all, "no %%writefile cells -- workers could not import the modules")

    # embedded modules must be byte-identical to src/
    embedded = {}
    for c in code:
        first, _, body = c.source.partition("\n")
        if first.startswith("%%writefile"):
            embedded[Path(first.split()[-1]).name] = body
    for name, body in embedded.items():
        p = SRC / name
        ck(p.exists(), f"embeds {name} which does not exist in src/")
        if p.exists():
            ck(body == p.read_text(),
               f"embedded {name} has drifted from src/{name} -- regenerate with gen_notebooks.py")
    ck(set(embedded) >= {"model.py", "driver.py", "fl_worker.py"},
       f"missing core modules among the embedded ones: {sorted(embedded)}")

    for i, c in enumerate(code):
        first = c.source.split("\n", 1)[0]
        if first.startswith("%%writefile"):
            continue                       # module bodies are checked against src/ above
        try:
            ast.parse(c.source.replace("%%", "#%%"))
        except SyntaxError as e:
            fails.append(f"{d.name}: code cell {i} does not parse: {e}")
        if SECRET.search(c.source):
            fails.append(f"{d.name}: code cell {i} looks like it contains a live credential")

    for name, body in embedded.items():
        try:
            ast.parse(body)
        except SyntaxError as e:
            fails.append(f"{d.name}: embedded {name} does not parse: {e}")


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    dirs = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    if not dirs:
        print(f"no notebook directories under {base}")
        return 1
    fails: list[str] = []
    for d in dirs:
        validate(d, fails)
        meta = json.loads((d / "kernel-metadata.json").read_text()) \
            if (d / "kernel-metadata.json").exists() else {}
        n = len([f for f in fails if f.startswith(d.name + ":")])
        print(f"  [{'FAIL' if n else ' OK '}] {meta.get('id', d.name):<45} "
              f"{(d / meta.get('code_file', '')).stat().st_size / 1024:.0f} KB"
              if meta.get("code_file") and (d / meta["code_file"]).exists()
              else f"  [FAIL] {d.name}")
    print(f"\n{len(dirs)} notebooks, {len(fails)} problems")
    for f in fails:
        print(f"  FAIL  {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

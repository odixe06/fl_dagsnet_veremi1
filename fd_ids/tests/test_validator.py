"""The static gate must actually reject a broken notebook.

Every mutation below passed the gate at some point. A validator that only ever says "all
checks passed" is indistinguishable from one that checks nothing, so each case here runs the
real script over a real (mutated) copy of the repository and demands a non-zero exit.
"""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = "papers/fd-ids-2025/notebook/20c/kernel-metadata.json"
# The notebook name comes from kernel-metadata, exactly as the validator reads it. Hardcoding
# fdids_20c.ipynb here broke every mutation case with FileNotFoundError the moment the 20c
# notebook was regenerated with --run-tag as fdids_20c_v2.ipynb (bug #41).
NB = "papers/fd-ids-2025/notebook/20c/" + json.loads((ROOT / MD).read_text())["code_file"]


def sandbox(tmp):
    for rel in ("scripts", "knowledge", "papers/fd-ids-2025/proj",
                "papers/fd-ids-2025/notebook"):
        shutil.copytree(ROOT / rel, tmp / rel, dirs_exist_ok=True)
    return tmp


def run(tmp):
    r = subprocess.run([sys.executable, str(tmp / "scripts/validate_notebooks.py")],
                       capture_output=True, text=True, cwd=tmp)
    return r.returncode, (r.stdout + r.stderr).strip()


def edit_cell(tmp, match, fn):
    p = tmp / NB
    nb = json.loads(p.read_text())
    hit = 0
    for c in nb["cells"]:
        if c["cell_type"] == "code" and match in "".join(c["source"]):
            c["source"] = fn(c["source"]); hit += 1; break
    assert hit == 1, f"no cell matched {match!r}"
    p.write_text(json.dumps(nb, indent=1))


def case(name, mutate):
    with tempfile.TemporaryDirectory() as t:
        tmp = sandbox(Path(t))
        mutate(tmp)
        code, out = run(tmp)
        assert code != 0, f"{name}: validator ACCEPTED it\n{out}"
        print(f"   rejected: {name}")


def main():
    with tempfile.TemporaryDirectory() as t:
        code, out = run(sandbox(Path(t)))
        assert code == 0, f"baseline must pass\n{out}"
    print("   baseline passes")

    # 1. a scientific parameter overwritten after the CFG literal the parser reads
    case("CFG['lr'] = 0.25 after the declaration",
         lambda tmp: edit_cell(tmp, "from proj.driver import run as train",
                               lambda src: ["CFG['lr'] = 0.25\n"] + src))
    # 2. cache marker written before the files it vouches for
    case("cache marker written before np.save",
         lambda tmp: edit_cell(tmp, "MF.write_text", lambda src:
                               [l for l in src if "MF.write_text" not in l][:1]
                               + ["    MF.write_text(json.dumps(want))\n"]
                               + [l for l in src if "MF.write_text" not in l][1:]))
    # 3. an unrelated dataset swapped into the metadata
    def swap_ds(tmp):
        p = tmp / MD; m = json.loads(p.read_text())
        m["dataset_sources"][1] = "someone/unrelated-dataset"
        p.write_text(json.dumps(m, indent=2))
    case("centralized dataset replaced", swap_ds)
    # 4. an embedded module that no longer matches its source
    case("embedded module gone stale",
         lambda tmp: edit_cell(tmp, "%%writefile /kaggle/working/proj/fdids.py",
                               lambda src: src + ["# drifted\n"]))
    # 5. a name no earlier cell binds
    case("name never bound by an earlier cell",
         lambda tmp: edit_cell(tmp, "t_origin=T0", lambda src:
                               [l.replace("t_origin=T0", "t_origin=T_START") for l in src]))
    # 6. rounds silently changed away from the paper's schedule
    case("rounds changed to 5",
         lambda tmp: edit_cell(tmp, "CFG = dict(", lambda src:
                               [l.replace("rounds=50,", "rounds=5,") for l in src]))
    # 7. is_private turned off on a notebook that carries a credential
    def unprivate(tmp):
        p = tmp / MD; m = json.loads(p.read_text())
        m["is_private"] = False
        p.write_text(json.dumps(m, indent=2))
    case("is_private turned off", unprivate)
    # 8. a checkpoint dataset attached to a notebook that claims to start from scratch:
    #    resolve_resume would import and continue those rounds under a fresh-run label
    def fresh_with_ckpt(tmp):
        edit_cell(tmp, "CFG = dict(", lambda src:
                  [l.replace("require_resume=True,", "require_resume=False,") for l in src])
        p = tmp / MD; m = json.loads(p.read_text())
        m["dataset_sources"].append("someone/fdids-20c-v2-ckpt")
        p.write_text(json.dumps(m, indent=2))
    case("checkpoint dataset attached to a fresh run", fresh_with_ckpt)
    # 9. a continuation with nothing attached: dies at the gate, after the queue wait
    def resume_without_source(tmp):
        edit_cell(tmp, "CFG = dict(", lambda src:
                  [l.replace("require_resume=False,", "require_resume=True,") for l in src])
        p = tmp / MD; m = json.loads(p.read_text())
        m["dataset_sources"] = m["dataset_sources"][:2]; m["kernel_sources"] = []
        p.write_text(json.dumps(m, indent=2))
    case("require_resume with no source attached", resume_without_source)

    print("\nALL 10 VALIDATOR CHECKS PASSED (1 baseline + 9 mutations rejected)")


if __name__ == "__main__":
    main()

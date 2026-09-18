# Pulling outputs back

Kaggle output is transient — the kernel's files live only as that version's committed output, and a later push overwrites them. Everything the run produced comes home to `papers/<slug>/runs/<run_name>/`.

Pull **whenever the kernel leaves `running`** — on `complete` *and* on `error`. An errored kernel still commits the files it wrote before dying, so pulling on failure is what recovers the checkpoints that make the next attempt resume instead of restart. Skipping the pull on failure is the single easiest way to lose a night of training.

## What comes back

| Directory | Contents | Size |
|---|---|---|
| `checkpoints/` | `ckpt_round_XXX.pt` (one per round) + `last.pt` | the bulk |
| `metrics/` | `round_XXX.json`, `history.csv` | tiny |
| `preds/` | `round_XXX.npz` — y_true, y_pred, y_prob | medium |
| `confusion/` | `round_XXX.npy` | tiny |
| `reports/` | per-class `classification_report` text | tiny |
| `logs/` | training log, the kernel's stdout/stderr | small |
| `meta.json` | frozen class map + normalization stats | tiny |
| **the executed `.ipynb`** | the notebook **with every rendered output** | small |

Pull the small directories always. For `checkpoints/`, pull `last.pt` plus every `ckpt_round_*.pt` by default — the user wants one checkpoint per round kept — but check free disk first and report the total rather than silently filling the drive.

### The executed notebook

Kaggle commits the run's `.ipynb` **with its outputs embedded** — audit tables, per-round log lines, the metric table, the curves, the confusion matrix — welded to the code and prose that produced them. It is what a reviewer actually reads, and the loose files under `runs/` cannot reconstruct it: they hold the numbers but not the narrative, the equations, or the deviation table beside them.

`kaggle kernels output` does not return the executed notebook; `kernels pull` returns source.
Fetch `__notebook__.ipynb` with the callable MCP output-download tool, or use the bundled
`download_executed_notebook.py` CLI-OAuth helper below. Preserve the exact kernel version. Save it to:

```
papers/<slug>/notebook/<name>.executed.ipynb
```

**Beside the pushed source notebook, never over it.** Two files with two jobs: `<name>.ipynb` is the clean input you keep editing and re-pushing, `<name>.executed.ipynb` is frozen evidence of one particular run. Overwriting the source with the executed copy means the next push carries a notebook fat with stale outputs, and the diff of what you actually changed disappears under them.

When several attempts produced results worth keeping, suffix by run: `<name>.executed.<run_name>.ipynb`.

```python
import json, shutil
from pathlib import Path

def stash_executed(pulled_dir, nb_dir, name, run_name=None):
    # Move the committed .ipynb out of the pulled output and beside the source notebook.
    nbs = [p for p in Path(pulled_dir).rglob("*.ipynb") if ".executed." not in p.name]
    if not nbs:
        print("no executed notebook in the pulled output — check the kernel committed")
        return None
    src = max(nbs, key=lambda p: p.stat().st_size)   # the one carrying outputs is the big one
    suffix = f".executed.{run_name}.ipynb" if run_name else ".executed.ipynb"
    dst = Path(nb_dir) / (name + suffix)
    assert dst != Path(nb_dir) / f"{name}.ipynb", "never overwrite the source notebook"
    shutil.copy2(src, dst)
    n_out = sum(len(c.get("outputs", [])) for c in json.loads(dst.read_text())["cells"])
    print(f"executed notebook -> {dst}  ({dst.stat().st_size/2**20:.1f} MB, {n_out} rendered outputs)")
    if n_out == 0:
        print("  WARNING: zero outputs — the kernel likely died before producing any, "
              "or this is the input copy rather than the commit.")
    return dst
```

A pulled notebook with zero rendered outputs is a signal, not a file to quietly stash: either the kernel died before the first cell produced anything, or what came back was the input rather than the commit. Say which, rather than reporting the pull as clean.


## How

1. **MCP first.** Inspect the current Kaggle MCP tool inventory for `download_notebook_output_zip`, `download_notebook_output`, and the matching output-list operation. Do not assume a fixed callable prefix. Call the available read operation with the full kernel slug and route the returned artifact into `papers/<slug>/runs/<run_name>/`.

2. **CLI fallback**, if the MCP server exposes no output tool:
   ```bash
   conda run -n nckh kaggle kernels output <user>/<kernel-slug> -p papers/<slug>/runs/<run_name>/
   ```
   First verify `conda run -n nckh kaggle --version` and a read-only API call. If `nckh` lacks Kaggle or OAuth is inactive, ask the user to run `conda run -n nckh kaggle auth login`; do not create another environment or request a pasted token.

   `kernels output` does not list Kaggle's special `__notebook__.ipynb` evidence file. Download
   that file through the same CLI OAuth without exposing the signed redirect URL:

   ```bash
   conda run -n nckh python \
     .agents/skills/kaggle-training-notebook/scripts/download_executed_notebook.py \
     <owner>/<kernel-slug> --output papers/<slug>/notebook/<name>.executed.ipynb
   ```

   Pass `--version N` when preserving evidence from an older kernel version rather than the latest.

3. **Verify, then record.** Compare against what the run should have produced.

```python
import json, hashlib, csv
from pathlib import Path

def verify_pull(run_dir: Path, expected_rounds: int):
    run_dir = Path(run_dir)
    ck = sorted((run_dir / "checkpoints").glob("ckpt_round_*.pt"))
    got = {int(p.stem.split("_")[-1]) for p in ck}
    missing = sorted(set(range(expected_rounds)) - got)

    hist = run_dir / "metrics" / "history.csv"
    rows = list(csv.DictReader(open(hist))) if hist.exists() else []

    total = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    print(f"checkpoints : {len(ck)}/{expected_rounds}  rounds present: {sorted(got)}")
    print(f"missing     : {missing or 'none'}")
    print(f"history.csv : {len(rows)} rows")
    print(f"last.pt     : {'yes' if (run_dir/'checkpoints'/'last.pt').exists() else 'MISSING'}")
    print(f"pulled size : {total/2**30:.2f} GB")

    # a truncated .pt is worse than a missing one - it fails at load time, later
    import torch
    for p in ck + [run_dir/"checkpoints"/"last.pt"]:
        if p.exists():
            try: torch.load(p, map_location="cpu", weights_only=False)
            except Exception as e: print(f"CORRUPT {p.name}: {e}")
    return missing
```

### Count bytes, not filenames

`kaggle kernels output` exits 0 while files are still missing **and** can leave **zero-byte
files** behind when a pull is interrupted. Both were seen on one run (2026-09-10): the first pull
of a 50-round scenario returned 35 weight filenames, of which one was 0 bytes, and 15 rounds were
absent entirely. An inventory that counts filenames reports 35/50 and silently blesses a corrupt
file; the round with the empty checkpoint only fails much later, at load or integrity-check time.

```bash
# what is actually usable
ls -l <run>/weights | awk '$5>0' | wc -l
# clear the debris before re-pulling, or the CLI may treat a 0-byte file as already present --
# but list first: some 0-byte files are REAL (see below) and must not be deleted
find <pull_dir> -type f -size 0 -not -name '*.done' -not -name '__init__.py'
kaggle kernels output <owner>/<slug> -p <pull_dir> -o --file-pattern '<regex for the gap>'
```

**Not every 0-byte file is debris.** Drivers that commit rounds atomically write an empty
completion marker last (`complete/round_NNN.done` in pFedES), and a copied package carries an
empty `__init__.py`. A blanket `find -size 0 -delete` erases the markers and the verifier then
reports every round as incomplete (seen 2026-09-14: 29 and 22 zero-byte files in two clean
pulls, all markers/`__init__.py`). Match the deletion to the artifact types that are never
legitimately empty (`weights/*.pt`, `confusion/*.npy`, `metrics/*.json`, `history.csv`).

Repeat until a size-aware inventory shows every round present, then let `verify_run.py`-style
integrity hashes have the last word. `-o/--force` matters on the retry: without it the CLI skips
files it believes are up to date.

A missing round is not automatically a problem — a run killed at the 12 h cap legitimately has fewer checkpoints than `ROUNDS`. Report which rounds exist and let the resume path continue from there. A *corrupt* checkpoint is a problem: re-pull it, and if it is still bad, resume from the newest round that loads.

## Selective pulls and completeness

When only specific missing files are needed, use the MCP output-download tool with `filePath`
or the CLI's `--file-pattern` after checking its local help. A selective pull is not a complete
archive. The CLI can exit successfully with files still missing: compare the local inventory
against expected round artifacts, then repeat the same pull for missing files. Recheck the
inventory instead of trusting exit status. Verify all 10 metrics in JSON/CSV as specified in
[`metrics.md`](metrics.md), including every round that will appear in `report.md`.

## Incremental pulls

Checkpoints are the expensive part and they never change once written, so skip what is already local and byte-identical:

```python
def needs_pull(local: Path, remote_size: int) -> bool:
    return not local.exists() or local.stat().st_size != remote_size
```

Keep a `runs/<run_name>/.pull-manifest.json` of `{filename: (size, mtime, sha256_of_first_1MB)}` so repeated pulls across a multi-attempt run stay cheap.

## After every pull

Three things, in order:

1. **Update `rebuild.md`** — append the Run history row (attempt, date, rounds completed, outcome, fix) and refresh the Results table from `history.csv`.
2. **Report to the user** — rounds completed this attempt, the 10 metrics for the latest round, total pulled size, and whether anything is missing or corrupt.
3. **Stage the resume** — if rounds remain, the pulled `checkpoints/` is what gets re-attached on the next push (as a Kaggle Dataset, or via the previous kernel's output). Say which one is being used.
4. **Write the report when the run is final** — turn the verified pull into `papers/<slug>/report.md` using [`reporting.md`](reporting.md). Generate figures locally from `history.csv` and confusion matrices; notebook plots displayed inline are not standalone report assets.

`history.csv` from a later attempt contains the earlier attempt's rows too when it resumed from their checkpoint, so merge rather than overwrite: concatenate, `drop_duplicates(subset="round", keep="last")`, sort by round.

## A run that spanned several sessions is still ONE run

Quota, the 12 h cap, or an account switch splits a run across sessions, and each session's
output arrives as its own pull. **The working directory should end up with one run directory,
not one per session** — the split is an accident of infrastructure, and a reader comparing
builds should not have to reassemble it. Provenance is kept, but it is kept *inside* the
merged run rather than as parallel trees.

```bash
conda run -n nckh python .agents/skills/kaggle-training-notebook/scripts/merge_sessions.py \
  --out runs_merged pull_session1/ pull_session2/
```

The script's real job is refusing a merge that would be a lie:

* every artifact present in more than one pull must be **byte-identical**, otherwise the later
  session retrained a round instead of resuming it and the halves are not one run;
* `history.csv` rows must agree on every overlapping round;
* `meta.json` must match — different data means nothing else is comparable;
* the merged round sequence must be **contiguous**, so a "50-round" directory cannot quietly
  contain 49.

What it keeps per session instead of merging: `logs/` (each session writes its own
`heartbeat.jsonl`, and only the one that ran out of budget has `stopped_early.json`), the
kernel logs, and `config.json` — `max_hours`, `deadline_ts` and `wandb_run_id` legitimately
differ per session, so they are recorded, not compared. `logs/sessions.json` says which rounds
each session actually produced and lists exactly which config fields differed.

Measured 2026-09-04 on build 4 here: 46 rounds from one account, 4 from another, rounds 0–45
byte-identical across both pulls, merged into 50 contiguous rounds. That identity check *is*
the evidence that the resume worked — report it, do not assume it.

**Verify the merged directory before deleting the pulls**, and delete them only once the merged
run passes: contiguous rounds, `last.pt` at the final round with finite weights, and the final
confusion matrix summing to the test row count.

## Stop the session when the pull is done

Weekly GPU quota is finite and a session that is still up keeps spending it. The moment every
output is verified on local disk — checkpoints, metrics, preds, confusion, reports, logs, and the
executed `.ipynb` — cancel the notebook session and confirm from the status that it is no longer
`running`, then report the remaining quota.

Follow [`stop-session.md`](stop-session.md) after verifying local artifacts and checking whether
any session still needs stopping. In this project's measured setup, automatic cancellation is
unavailable; request the manual UI action promptly and corroborate the outcome with the correct
account's quota and run-specific heartbeat. A replacement push does not stop the original worker
and must not be used. For an explicit immediate-stop request, stop takes priority over the pull;
retrieve committed version-specific outputs afterward and record what was lost.

## A finished run can carry the PREVIOUS session's "stopped early" record

Resume copies the earlier session's `logs/` into the working directory, and a driver typically
writes its stop-reason file **only when it stops early**. A session that runs to the final round
therefore never overwrites it, and the completed run directory still holds the *earlier* session's
record.

Seen on a 50-round run (2026-09-10): 50/50 rounds committed, every per-round artifact present —
and `logs/stopped_early.json` saying `{"last_round": 31, "reason": "wall_clock_budget"}`, which was
session 1's.

**Never decide "did this run finish?" from that file.** Ask the artifacts:

```bash
python scripts/verify_run.py <run_dir> --require-complete
```

If a project wants the stop file to be trustworthy, the driver must delete a stale one when a
session starts — but do not make that change *in the middle of a multi-session chain*, where a
pinned-source manifest exists precisely to guarantee later sessions run the same code as earlier
ones. Record it and fix it after the chain completes.

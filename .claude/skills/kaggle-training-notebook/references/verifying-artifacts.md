# Verifying a run's artifacts, and surviving a crash mid-commit

A training run's value is entirely in the files it leaves behind. Two failure classes attack
those files, and neither shows up in a passing training loop:

* **Crash-safety** — the process dies partway through writing, and what survives is a
  directory that *looks* consistent while describing two different states.
* **Verification blindness** — the checker passes on artifacts that are wrong, so "9/9 passed"
  becomes evidence of nothing.

Every item below was reproduced on a real run in this repo, not reasoned about. All of them
survived an earlier review that had already declared the pipeline verified.

## 1 — Write order: the marker is a promise, publish it last

Adopt one rule and apply it to every multi-file write: **the file that certifies a unit of work
is written after everything that unit needs, and never before.**

```
weights -> confusion -> predictions -> per-class -> metrics -> client log
        -> history -> resume state -> complete/round_NNN.done      <-- last
```

Two bugs found by taking this seriously:

* A single `resume_state.pt` rewritten each round cannot be correct. A crash between the marker
  write and the state write leaves the two describing different rounds, and the next session
  asserts out. Write the resume state **per round** (`resume/round_NNN.pt`) *before* the marker,
  so the state a resume reads always belongs to the round the marker certifies.
* An **import** from a previous session has the same shape and is easier to get wrong. Copying
  files in path order copies `complete/` early — alphabetically it comes before `weights/` — so
  an interrupted copy publishes markers for rounds whose weights never arrived. Worse, the retry
  then sees a marker, concludes there is nothing to do, and returns 0. Copy the markers **last**,
  and only after a completeness check over rounds `1..N` passes.

While copying: a destination file that already exists is only safe to skip if its **size matches
the source**. A file cut off mid-write has the right name and the wrong length, and skipping it
turns a recoverable partial import into a permanent one. Copy through `name.part` then
`os.replace`, so a file is never visible at a partial length.

## 2 — Derived files must be atomic and reconstructible

A CSV that accumulates one row per round is usually rewritten whole on every append. That makes
a crash mid-rewrite able to destroy history that was already committed — the markers and the
per-round JSON survive, and the CSV comes back holding only the rounds written before the
interruption.

Two properties fix it together, and neither is sufficient alone:

```python
# 1. atomic publish: a reader sees the old file or the new one, never a truncated one
tmp = path.with_name(path.name + ".tmp")
with tmp.open("w", newline="") as fh:
    ...
    fh.flush()
    os.fsync(fh.fileno())
os.replace(tmp, path)

# 2. self-healing: rebuild any committed round that is missing from the file
for rnd in completed_rounds():
    if rnd not in rows:
        rows[rnd] = row_from_json(rnd)      # the CSV is derived; the JSON is the source
```

Atomicity stops *this* write from losing data. Reconstruction repairs damage an *earlier* crash
already did — without it a truncated file stays truncated for the rest of the run. Design derived
artifacts so the reconstruction is always possible: every column of the CSV should come from a
per-round file written inside the same commit.

## 3 — Row counts do not identify data

A fingerprint over shape — row counts, file paths, class count — cannot tell two datasets apart
when one is a rewrite of the other at the same path. Rewriting a parquet file with different
feature values and permuted labels leaves every count identical. The consequences are quiet and
severe: a prepack/decode cache reports a hit and serves the *old* arrays, and a resume continues
a run onto data that is not the data it trained on.

Use content identity, and take it from the **footer** so it stays cheap enough to compute before
the expensive decode:

```python
md = pq.ParquetFile(path).metadata
parts = [path.name, path.stat().st_size, md.num_rows, md.num_row_groups, md.num_columns]
for g in range(md.num_row_groups):
    rg = md.row_group(g)
    for c in range(rg.num_columns):
        col = rg.column(c)
        st = col.statistics
        parts += [col.total_compressed_size,
                  None if st is None else (st.min, st.max, st.null_count)]
```

Hash that per file, fold it into **both** the run fingerprint and the cache key, and record it in
`config.json`. Modification time and file size alone are too weak — an in-place rewrite can
preserve size, and mtime changes on operations that do not change content.

This has to stay footer-only. The fingerprint is also what lets `--require-resume` fail in
seconds instead of after a multi-minute decode; a digest that reads columns destroys that.

## 4 — A configuration fingerprint is not a run identity

Two independent trainings of the same scenario share every constant, so they share the
fingerprint — while holding completely different weights. Code that searches for "a previous run
with my fingerprint" and silently takes the one with the most completed rounds will splice two
unrelated histories together and report success.

When more than one candidate matches, **refuse and name them**:

```
3 candidate resume sources share fingerprint 81eed03e:
  /kaggle/input/run-a (12 rounds)
  /kaggle/input/run-b (31 rounds)
Refusing to guess which run to continue. Pass --resume-source <dir> to choose.
```

Offer an explicit selector rather than a heuristic. "Most rounds" is not more likely to be
correct — it is just more likely to be *expensive* when wrong.

## 5 — A verifier that checks formulas does not check the run

This is the finding that mattered most. A verifier can look thorough — recomputing metrics,
exercising aggregation identities, validating coefficient ranges — and still pass on artifacts
that are materially wrong, because those checks run on **freshly generated random inputs**
rather than on the values the run actually stored.

Four injections into four independent copies of a run that had passed "9/9, 0 failed". Every one
still passed:

| Injection | What was never checked |
|---|---|
| One accuracy in `history.csv` set to 0.999999 | The CSV was never compared against the per-round JSON it is derived from |
| Every `mu` in a resume checkpoint set to 999 | Resume state was never compared with the committed per-round log |
| A single `NaN` written into one weight tensor | `load_state_dict(strict=True)` checks *shape*, never values |
| One client's log entry replaced by a copy of another's | The client id set was never required to be exactly `0..n-1` |

The rule that falls out: **check stored values against other stored values, and against the
schedule the configuration implies.** Concretely, for every completed round:

* every metric in the CSV equals the per-round JSON within the display rounding tolerance;
* every float tensor in the checkpoint is finite — `strict=True` is a structural check, add
  `torch.isfinite(v).all()` explicitly;
* the flat/packed round trip is exact on **every** round, not only the last;
* per-class report equals what the confusion matrix implies;
* predictions rebuild the confusion matrix exactly, and cover the full test set;
* client ids are exactly the expected set, with no duplicates, and each client's row count
  matches `config.json`;
* the learning rate each client used equals the schedule for that round;
* chained state — a coefficient derived from round `t` and applied in round `t+1` — matches both
  the formula **and** the resume checkpoint that carries it across a session boundary;
* seeds are a pure function of `(round, client)`, so scheduling cannot leak into the RNG.

Keep the random-input checks too. They catch a broken formula; they just cannot catch a broken
run, and one is not a substitute for the other.

## 6 — Exit 0 does not mean the run finished

A verifier certifies *the rounds that exist*. That is the right default — you want to check a
mid-run checkpoint — but it means a 3-round directory whose config says 50 rounds passes
everything. Add an explicit, opt-in gate and require it before any completion claim:

```
--require-complete   ->  assert done[-1] == config["rounds"]
```

Never quote a final number from a directory that has not passed that gate, and say "rounds 1–N
of M" whenever it has not.

## 7 — Keep the verifier and the notebook validator in the repo

Both checks get run after every source change, across sessions, and from different machines. Ad
hoc snippets in a scratch directory are lost the moment the scratch directory is cleaned — which
happens between sessions — and the temptation is then to skip the check rather than rewrite it.
Make them committed scripts with an exit code, and run them as a build step:

```
scripts/verify_run.py <run_dir> [--require-complete]
scripts/validate_notebooks.py
```

The notebook validator earns its place separately: it catches the failures that cost a version or
a run instead of raising locally — a missing `kernelspec`, a title whose slug disagrees with the
metadata id, an embedded module that has drifted from `src/`, or a live credential in a notebook
that is not private.

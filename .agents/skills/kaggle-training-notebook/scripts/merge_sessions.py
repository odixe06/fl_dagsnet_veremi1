#!/usr/bin/env python3
"""Merge the pulls of a run that spanned several Kaggle sessions into ONE run directory.

A run stopped by quota and continued later (possibly on another account) arrives as two or
more output pulls. A resumed pull carries either every earlier round (resume copied the whole
attached tree into /kaggle/working) or, when the session was resumed from a HANDOFF BUNDLE,
only the round it resumed from plus the rounds it trained — so the merge is a verification
that the sessions really are one run (every overlapping artifact byte-identical, every
overlapping history row equal), and the union of the per-round files is the easy part.

What this refuses to paper over:

  * an overlapping artifact whose bytes differ between pulls — that means the later session
    RETRAINED the round instead of resuming it, and the halves are not one run;
  * a history.csv whose overlapping rows disagree, for the same reason;
  * a gap in the round sequence, which would silently produce a "50-round" directory with 49
    rounds in it.

Provenance is not discarded: each session's kernel log is kept under logs/sessions/, and
logs/sessions.json records which rounds each session actually produced. The merged directory
reads as a single run; the record of how it was produced stays next to it.

    merge_sessions.py --out runs_merged  pull_session1/ pull_session2/ ...

Pulls must be given in chronological order. --run-name picks the run when a pull holds several.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import shutil
import sys
from pathlib import Path

# Every per-round directory any project here writes. Absent ones are skipped, so listing a
# name costs nothing -- but OMITTING one is not free: the merge then quietly produces a run
# directory without the weights in it and still reports success. TinyProto names its
# checkpoints weights/ and also writes protos/, client_log/, complete/ and resume/; a merge
# that dropped those would lose the models, the prototypes and the round markers.
ROUND_DIRS = ("checkpoints", "weights", "protos", "resume", "complete", "client_log",
              "metrics", "preds", "confusion", "reports")
# Files that legitimately differ between sessions because each session rewrites them from
# scratch or appends to its own copy. They are NOT evidence of retraining; the last session's
# version wins, except the per-round CSVs, which are merged by round key (see KEYED) after
# their overlapping rows were verified equal.
ROLLING = ("history.csv", "clients.csv", "last.pt", "best.pt")
# Per-round tables: one row per round (history.csv) or per (round, cid) (clients.csv). A
# session resumed from a handoff bundle holds only its own rounds, so "last wins" would leave
# the merged run with a truncated table; the union keyed by round is what a single session
# would have written.
KEYED = {"history.csv": ("round",), "clients.csv": ("round", "cid")}
# Per-session provenance: written fresh by every session with that session's own budget,
# resume flag, startup time or W&B id. Never compared; each copy is kept under
# logs/sessions/<n>/ and the destination holds the last session's. Some drivers write this as
# config.json at the run root, others as reports/manifest.json. handoff.json is the sha chain
# a session resumed from; it is provenance of that session, not a property of the merged run.
PER_SESSION = ("config.json", "manifest.json", "handoff.json")


def find_run(pull: Path, run_name: str | None) -> Path:
    """The <pull>/**/runs/<run_name> directory, whatever Kaggle nested it under."""
    cands = [p for p in pull.rglob("*") if p.is_dir() and p.parent.name == "runs"
             and (p / "metrics").is_dir()]
    if run_name:
        cands = [p for p in cands if p.name == run_name]
    if not cands:
        sys.exit(f"no run directory with a metrics/ subdir under {pull}")
    if len({p.name for p in cands}) > 1:
        sys.exit(f"several runs under {pull}: {sorted(p.name for p in cands)} — pass --run-name")
    return cands[0]


def rounds_in(run: Path) -> set[int]:
    return {int(p.stem.split("_")[-1]) for p in (run / "metrics").glob("round_*.json")}


def merge_keyed(runs: list[Path], name: str, key: tuple) -> tuple[list[str], list[dict]]:
    """Union of a per-round CSV across sessions, keyed by `key`. Overlapping rows must be
    equal. Returns (fieldnames, rows sorted by key); ([], []) when no session has the file."""
    fields, merged = [], {}
    for run in runs:
        # metrics/history.csv in some drivers, history.csv at the run root in others
        f = next((c for c in (run / "metrics" / name, run / name) if c.exists()), None)
        if f is None:
            continue
        with open(f) as fh:
            rd = csv.DictReader(fh)
            if rd.fieldnames and not fields:
                fields = list(rd.fieldnames)
            for row in rd:
                k = tuple(int(row[c]) for c in key)
                if k in merged and merged[k] != row:
                    sys.exit(f"{name} row for {dict(zip(key, k))} differs in {run} — the later "
                             f"session retrained it instead of resuming. Not one run.")
                merged[k] = row
    return fields, [merged[k] for k in sorted(merged)]


def check_history(runs: list[Path]) -> list[dict]:
    """Overlapping rows must agree. Returns the merged history, one row per round."""
    return merge_keyed(runs, "history.csv", ("round",))[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pulls", nargs="+", type=Path, help="session pulls, oldest first")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()

    runs = [find_run(p, a.run_name) for p in a.pulls]
    print(f"run name: {runs[0].name}")
    for pull, run in zip(a.pulls, runs):
        r = sorted(rounds_in(run))
        print(f"  {pull.name:<24} rounds {r[0]}..{r[-1]}  ({len(r)})")

    history = check_history(runs)
    print(f"history.csv rows agree on every overlap; union covers {len(history)} rounds")

    # Copy oldest first so a later session's version of a shared file lands last — but only
    # after proving the two versions are identical, so "last wins" can never change content.
    out = a.out
    for sub in ROUND_DIRS + ("logs",):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "logs" / "sessions").mkdir(parents=True, exist_ok=True)

    clashes = 0
    for i, run in enumerate(runs, 1):
        for sub in ROUND_DIRS:
            src = run / sub
            if not src.is_dir():
                continue
            for f in sorted(src.iterdir()):
                if not f.is_file():
                    continue
                dst = out / sub / f.name
                if f.name in PER_SESSION:
                    (out / "logs" / "sessions" / str(i)).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, out / "logs" / "sessions" / str(i) / f.name)
                    if f.name == "handoff.json":
                        continue            # provenance only: never part of the merged run
                elif dst.exists() and f.name not in ROLLING:
                    if not filecmp.cmp(f, dst, shallow=False):
                        sys.exit(f"{sub}/{f.name} differs between sessions — the later one "
                                 f"retrained this round. The halves are not one run.")
                elif dst.exists():
                    clashes += 1
                shutil.copy2(f, dst)
        # Files at the run root — meta.json, config.json, and any frozen artifact the run
        # inherited. The first version of this script copied only the per-round directories
        # and silently produced a run directory with no meta.json in it.
        for f in sorted(run.iterdir()):
            if not f.is_file():
                continue
            dst = out / f.name
            # meta.json describes the DATA — classes, feature columns, row counts. If it
            # differs the sessions trained on different inputs and nothing else matters.
            # config.json legitimately differs (max_hours, deadline_ts, wandb_run_id are
            # per session), so it is kept per session under logs/sessions/<n>/ instead of
            # being compared; the root copy is the last session's.
            if dst.exists() and f.name == "meta.json" and not filecmp.cmp(f, dst, shallow=False):
                sys.exit("meta.json differs between sessions — they did not train on the "
                         "same data. Not one run.")
            shutil.copy2(f, dst)
            if f.name in PER_SESSION:
                (out / "logs" / "sessions" / str(i)).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out / "logs" / "sessions" / str(i) / f.name)
        # logs/ is per-session by nature: each session writes its own heartbeat.jsonl and only
        # the session that ran out of budget has stopped_early.json. Merging them by name would
        # either lose one or falsely report a mismatch, so keep them side by side.
        # -- except per-ROUND log files (logs/round_NNN.*, e.g. one JSON of per-client stats
        # per round): those are round artifacts like weights, resumed sessions import them,
        # and they merge under the same byte-identity rule.
        if (run / "logs").is_dir():
            d = out / "logs" / "sessions" / str(i)
            d.mkdir(parents=True, exist_ok=True)
            for f in sorted((run / "logs").iterdir()):
                if not f.is_file():
                    continue
                if f.name.startswith("round_"):
                    dst = out / "logs" / f.name
                    if dst.exists() and not filecmp.cmp(f, dst, shallow=False):
                        sys.exit(f"logs/{f.name} differs between sessions — the later one "
                                 f"retrained this round. The halves are not one run.")
                    shutil.copy2(f, dst)
                else:
                    shutil.copy2(f, d / f.name)

    # provenance: whole-session logs and who produced which rounds
    prov = []
    for i, (pull, run) in enumerate(zip(a.pulls, runs), 1):
        d = out / "logs" / "sessions" / str(i)
        d.mkdir(parents=True, exist_ok=True)
        for log in sorted(pull.glob("*.log")):
            shutil.copy2(log, d / log.name)
        r = sorted(rounds_in(run))
        prov.append({"session": i, "pull": str(pull), "kernel_logs":
                     [p.name for p in sorted(pull.glob("*.log"))],
                     "rounds_present": [r[0], r[-1]], "n_rounds": len(r)})
    new_by_session, seen = [], set()
    for run, p in zip(runs, prov):
        rs = rounds_in(run)
        p["rounds_produced"] = sorted(rs - seen)
        seen |= rs
        new_by_session.append(len(p["rounds_produced"]))
    (out / "logs" / "sessions.json").write_text(json.dumps(prov, indent=2))

    # The per-round tables: the union over sessions, not the last session's copy.
    for name, key in KEYED.items():
        fields, rows = merge_keyed(runs, name, key)
        if rows:
            with open(out / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader(); w.writerows(rows)
            print(f"  {name}: {len(rows)} rows, union over sessions keyed by {'/'.join(key)}")

    have = sorted(rounds_in(out))
    # Contiguity is checked from the run's OWN first round, not from 0. Rounds are 1-indexed in
    # some projects and 0-indexed in others; `range(have[-1] + 1)` reported a phantom gap at
    # round 0 for every 1-indexed run and aborted a merge that was in fact complete.
    missing = sorted(set(range(have[0], have[-1] + 1)) - set(have))
    if missing:
        sys.exit(f"merged directory has gaps at rounds {missing} — not a complete run")
    if have[0] not in (0, 1):
        sys.exit(f"merged directory starts at round {have[0]}: rounds before it are missing "
                 f"from every pull, so this is not the whole run")
    for sub in ROUND_DIRS:
        n = len(list((out / sub).glob("round_*"))) or len(list((out / sub).glob("*round_*")))
        if n < len(have):
            print(f"  warning: {sub}/ has {n} per-round files for {len(have)} rounds")

    print(f"\nmerged -> {out}")
    print(f"  rounds {have[0]}..{have[-1]} contiguous, {len(have)} total")
    print(f"  new rounds per session: {new_by_session}")
    print(f"  {clashes} rolling files taken from the last session ({', '.join(ROLLING)})")
    print(f"  per-session logs kept separately under {out / 'logs' / 'sessions'}")
    print(f"  provenance in {out / 'logs' / 'sessions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

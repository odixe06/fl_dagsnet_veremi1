#!/usr/bin/env python
"""Generate the whole production session chain from one plan, then validate it.

The chain is data, not a sequence of commands typed by hand: every session must agree with the
one before it on scenario, owner and slug, and a typo in any of those is only caught after a
push. Keeping the plan here means `--session`/`--resume-from` are derived, never retyped.

    python scripts/gen_production.py [--out DIR] [--only 20client|50client|100client]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MU_DATASET = "odixe0502/tinyproto-fp-mu-{n}client"

# (scenario, [(session, owner, max_hours)]) -- max_hours is re-derived from real quota before
# each push; see CONTEXT.md 5d. Session boundaries are decided by the driver's budget gate,
# not by these numbers, which only cap the session.
# (session, owner, max_hours, resume_dataset). resume_dataset is None for the normal route,
# where session N attaches session N-1's kernel output -- which works ONLY within one account.
#
# 20client finished all 50 rounds in session 1 (2026-09-09); its entry is the record of what ran.
# max_hours are re-derived from session 1's measured round times (logs/timing_*.json).
#
# 100client changed shape on 2026-09-09. The original plan handed s3 (khanhmay0304) to s4
# (minhtrit06) by kernel source. A probe proved that cannot work: Kaggle drops a cross-account
# kernel source at push time with a one-line warning, reports the push as successful, and mounts
# an EMPTY /kaggle/input. So the whole remainder moved to minhtrit06 (29.91 h, enough for the
# 34 remaining rounds at ~2336 s each) and the ONE unavoidable account boundary was moved to the
# earliest and cheapest point in the chain: s1 -> s2, where the payload is 16 rounds (2.8 GB,
# already on local disk from the s1 pull) instead of 43 rounds at s3 -> s4. s3 and s4 are then
# same-account and resume the ordinary way.
PLAN = {
    "20client":  [(1, "minhtran0601", 11.0, None)],
    # 50client finished all 50 rounds in session 2 (2026-09-10). Record, not work to do.
    "50client":  [(1, "odixe0502", 11.0, None), (2, "odixe0502", 7.9, None)],
    # s3/s4 sized from session 2's measured 2345 s/round: 18 rounds remain, and 18 do NOT fit
    # in one session. Reaching round 50 inside s3 would need max_hours 11.99, i.e. exactly
    # Kaggle's 12 h platform cap, leaving the driver no room to stop cleanly first -- and a
    # platform kill can take the whole session's output with it. So s3 takes 33..49 and s4
    # takes the single remaining round.
    "100client": [(1, "khanhmay0304", 11.0, None),
                  (2, "minhtrit06", 11.0, "minhtrit06/tinyproto-fp-resume-100client-r16"),
                  (3, "minhtrit06", 11.5, None),
                  # s4 was planned for minhtrit06 (kernel-output resume, no upload). The user
                  # moved it to minhtran0601 on 2026-09-11, so it needs the dataset route again.
                  # max_hours 2.0, not 3.0: minhtran0601 has only 2.13 h of GPU quota left, and
                  # the gate before round 50 needs max_hours > 0.10 h (setup+import, measured in
                  # s3) + 1.15 * 2544 s (s3 worst round) = 0.92 h, so 2.0 clears the gate with
                  # margin while staying under the quota that would kill the session outright.
                  (4, "minhtran0601", 2.0,
                   "minhtran0601/tinyproto-fp-resume-100client-r49")],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "papers/tinyproto-lee-2026/production"))
    ap.add_argument("--only", default="", choices=["", *PLAN])
    a = ap.parse_args()
    out = Path(a.out)
    made = []
    for scenario, sessions in PLAN.items():
        if a.only and scenario != a.only:
            continue
        n = scenario.replace("client", "")
        for session, owner, max_hours, resume_ds in sessions:
            cmd = [sys.executable, str(ROOT / "scripts/gen_notebooks.py"),
                   "--out", str(out), "--only", f"train{n}", "--owner", owner,
                   "--mu-dataset", MU_DATASET, "--session", str(session),
                   "--max-hours", str(max_hours)]
            if resume_ds:
                cmd += ["--resume-dataset", resume_ds]
            elif session > 1:
                prev_owner = dict((s, o) for s, o, _, _ in sessions)[session - 1]
                if prev_owner != owner:
                    print(f"{scenario} s{session}: {owner} cannot read {prev_owner}'s kernel "
                          f"output. Give this session a resume_dataset in PLAN.")
                    return 2
                cmd += ["--resume-from",
                        f"{owner}/tinyproto-fp-train-{scenario}-s{session - 1}"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                print(r.stdout + r.stderr)
                return r.returncode
            made.append(f"{owner}/tinyproto-fp-train-{scenario}-s{session}")
    for m in made:
        print("  " + m)
    print(f"\n{len(made)} production notebooks -> {out}")
    return subprocess.run([sys.executable, str(ROOT / "scripts/validate_notebooks.py"),
                           str(out)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())

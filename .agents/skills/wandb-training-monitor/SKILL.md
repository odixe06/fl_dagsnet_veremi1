---
name: wandb-training-monitor
description: Monitor and analyze active or completed Weights & Biases model-training runs. Use when the user asks for W&B run status, progress, health, metric trends, anomalies, or ETA. Do not use it to start, stop, or modify a run unless separately authorized.
---

# W&B Training Monitor

Inspect the actual W&B run before drawing conclusions. Treat monitoring as read-only: it does
not authorize changing config, resuming, stopping, deleting, or pushing anything.

Read `CONTEXT.md` and its later dated updates before selecting a run. Use the callable tools
provided by the current Codex session or the local helper below; do not assume Claude-specific
tools exist. Timestamp live observations and distinguish them from verified final artifacts.

## Authentication and target

Use the project conda environment (`nckh` in this repository). Prefer credentials already stored
by `wandb login` in `~/.netrc`; this repository does not need a persistent `.env`. If a user
temporarily supplies `.env`, [`scripts/inspect_run.py`](scripts/inspect_run.py) can read
`WANDB_API_KEY` in-process. Never print it, put it in a command argument, or commit it. Embedding it
in a notebook is permitted only when the repository records the owner's explicit acceptance of
private remote version-history exposure; credential injection belongs to the Kaggle training
workflow, not this read-only monitor. Remove a temporary `.env` after `wandb login` is verified.

Require an exact `entity/project/run_id`. Prefer the locator persisted by the notebook as
`wandb_run.json`; otherwise use the run URL or ask for the missing identity. Never guess the
latest run when that could select a different experiment.

```bash
conda run -n nckh python \
  .agents/skills/wandb-training-monitor/scripts/inspect_run.py \
  --run ENTITY/PROJECT/RUN_ID
```

## What to inspect

Report the run state and URL, latest W&B update time, completed round and global step, training
loss, skip percentage, gradient norm, maximum absolute logit, softmax entropy when logged, latest evaluation
metrics, and ETA when round durations are available. Compare against the run's own earlier points
and configured tripwires; an absolute gradient norm is not an anomaly by itself.

Flag at least:

- a running run with no new heartbeat past the chosen stale interval;
- failed, crashed, or killed state;
- non-finite loss or diagnostic values;
- skip rate above the notebook's abort threshold;
- abrupt loss/logit/gradient growth or entropy collapse relative to earlier heartbeats;
- missing expected evaluation metrics after a completed round;
- disagreement between W&B state and Kaggle kernel state;
- training loss that rises and stays up while `skip_pct` remains 0; finite gradients alone
  do not establish healthy training. Inspect which tripwires the actual notebook implements.

## Deciding whether a Kaggle run is actually alive

No single source answers this. All three were wrong on their own at least once here.

| source | what it really reports |
|---|---|
| `kaggle kernels status` | the state of the newest **batch version**, not session liveness. A stop-notebook pushed as a new version reads `COMPLETE` while the earlier session keeps training. |
| W&B `run.state` | the last state the client wrote. A hard-cancelled kernel never reaches `wandb.finish()`, so the run stays `running` forever as an orphan. |
| `get_accelerator_quota` -> `time_reserved` | account-wide accelerator reservation; it cannot identify which of several runs is active. |

Match the exact notebook session/version and W&B run. Use a fresh history `_timestamp` with
advancing `progress/global_step` as evidence of recent progress, and correlate it with that
session's status. Account quota is corroborating evidence, not a per-run liveness test. Otherwise
say which source claims what and call the cause unknown; never silently pick one. An orphan `running` state is a
reporting artifact, not evidence that hardware is busy — and it cannot be repaired through the
read-only API, so record the discrepancy rather than editing history.

## Repeated monitoring

When the user explicitly asks to monitor or babysit an active run, inspect immediately and then
roughly every 10–15 minutes or after each newly completed round, whichever comes first. Report
only material progress or anomalies. Use the product's wait/monitor mechanism when available and
do not claim monitoring continues after the current task ends.

## Notebook logging contract

For notebooks instrumented for later monitoring, log a heartbeat no more often than every few
hundred training steps and log all evaluation metrics once per round. Use stable keys such as:

- `progress/round`, `progress/completed_round`, `progress/global_step`, `progress/round_step`;
- `train/loss`, `train/skip_pct`, `train/grad_norm`, `train/abs_logit_max`,
  `train/softmax_entropy`;
- `round/duration_s` and `eval/<metric>`.

The evaluation contract is exactly: `accuracy`, `precision_macro`, `precision_micro`,
`precision_weighted`, `recall_macro`, `recall_micro`, `recall_weighted`, `f1_macro`,
`f1_micro`, `f1_weighted`, each prefixed with `eval/` in W&B. Check all 10 for a completed
round, even though five values coincide in single-label multiclass classification. The helper's
default snapshot emphasizes accuracy and macro F1; inspect full round history through the
read-only W&B API to check the other eight. Missing keys in a filtered snapshot do not prove they
were never logged. Match the round identity; do not combine stale metrics from different rounds.
A concise progress update may focus on `f1_macro`, but a results report must include all 10;
follow [`reporting.md`](../kaggle-training-notebook/references/reporting.md) and verify local
artifacts before writing final values to `report.md`.

For a verified continuation, use a stable run ID and `resume="allow"`; this restores W&B history,
not model checkpoints. Confirm the checkpoint round and monotonic steps before reusing an ID.
Persist only the run identity, project,
entity, and URL in `wandb_run.json`. On Kaggle, retrieve credentials with
`UserSecretsClient().get_secret("wandb_key")`; fail before expensive preparation if the secret or
Internet access is unavailable. Never persist the secret itself.

Kaggle CLI/API pushes do not carry a secret attachment selected in the Notebook Editor into the
new batch version. For a secret-backed run, upload and validate the private source, then have the
owner enable `wandb_key` in **Add-ons > Secrets** and launch that source from the editor with
**Save Version > Save & Run All**. Do not respond to the missing-secret failure by pushing again;
that creates another batch version with the same limitation. A documented owner-approved inline
key or private credential dataset is an alternative, but creating either is outside this monitor's
read-only scope.

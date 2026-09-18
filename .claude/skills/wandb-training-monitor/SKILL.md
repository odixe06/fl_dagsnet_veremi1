---
name: wandb-training-monitor
description: Monitor and analyze active or completed Weights & Biases training runs. Use for progress, health, metrics, anomalies or ETA; do not start, stop or modify a run unless separately authorized.
---

# W&B training monitor

Read the maintained [shared skill](../../../.agents/skills/wandb-training-monitor/SKILL.md).
Resolve its relative paths from its own directory. Use `CONTEXT.md` and the exact persisted
`entity/project/run_id` to select the run; never guess the newest run.

Use the current host's callable tools or the shared local helper. Monitoring is read-only.
Account-wide Kaggle reservations are not proof that a particular notebook is alive; correlate
the exact session with fresh W&B step progress. Follow the shared 10-metric reporting contract.

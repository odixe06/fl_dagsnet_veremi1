# Stop a Kaggle notebook session

**There is no reliable programmatic way to stop a running Kaggle session from here. Ask the
user to stop it by hand, and say so immediately instead of trying things first.**

Both automated routes were tried against live sessions in this project and both failed. What
follows is the evidence, so nobody re-derives it at the cost of another run.

## The one route that works

1. Resolve the exact `<owner>/<kernel-slug>` and confirm it is the run the user means.
2. **Notify the user in the current Codex conversation immediately.** Do not assume Claude
   Remote Control or `PushNotification` is available, and do not send external messages without
   authorization. The session burns quota while waiting for the manual stop.
3. Issue a **MANUAL ACTION** block in the reply with the exact URL and the exact clicks.
4. Wait. Do not push anything at the kernel while waiting.
5. When they say it is stopped, **verify independently** — see *Proving it stopped*.

```
## ⚠ MANUAL ACTION — only you can stop this session

Open https://www.kaggle.com/code/<owner>/<slug>
  -> "Stop session" (or Run -> Cancel run) -> confirm

Tell me when it is stopped and I will verify with get_accelerator_quota.
```

Keep the notification body under 200 characters, one line, and lead with the cost of waiting:
`"Kaggle session <slug> still running and burning GPU quota — needs you to stop it in the UI"`.

### Say it early

The value of the manual route is entirely in how fast the user hears about it. Every minute
spent attempting an automated stop is a minute of quota spent. Do not attempt, fail, then ask
— ask first, and only investigate while waiting.

## Why MCP cancellation is not the answer

`cancel_notebook_session` needs an active `kernelSessionId`, and **no read available here
returns one.** `get_notebook_session_status` returns `{"status": "RUNNING"}` and nothing else;
`get_notebook_info` returns the kernel's metadata `id`, which is a different identifier.

**Measured 2026-09-04.** Calling `cancel_notebook_session` with the metadata id `133064204`
returned `Permission 'kernelSessions.cancel' was denied`. So even a correct-looking id fails:
the `KGAT_` bearer used here does not carry the cancel scope. Do not substitute the kernel
metadata ID for a session ID, and do not read a permission error as "wrong id" — try it at most
once, then go manual.

As of Kaggle CLI 2.2.4, `KaggleApi.cancel_kernel_run(owner, slug)` does not exist. The CLI can
prove status but cannot cancel:

```bash
conda run -n nckh kaggle kernels status <owner>/<kernel-slug>
```

## Why a replacement push is not the answer either

The idea: push a one-cell CPU notebook over the same `<owner>/<slug>` so the new version
supersedes the running one. It does supersede the version. **It does not stop the worker.**

**Measured 2026-09-03.** A one-cell CPU stop notebook was pushed over a running TPU kernel. The
replacement reached `COMPLETE`, its stop marker verified — and the original TPU session **kept
training for roughly 18 more minutes**, logging W&B heartbeats from step 2,250 through 5,750 at
an unbroken 76–78 s cadence. The whole time, `kaggle kernels status` returned `COMPLETE`,
because it reports the newest batch **version**, which was the replacement. The run was recorded
as stopped at step 2,250; it was not.

It also costs something. The push increments the remote version and makes the stop notebook the
latest source, so the training notebook must be pushed back before the next run — and pushing it
back **starts training again**. Uncommitted output from the interrupted version may be
unrecoverable.

So: a replacement push adds a version, destroys the source, may lose artifacts, and does not
stop anything. Do not use it. If a previous session's notes recommend it, they predate the
2026-09-03 measurement.

## Proving it stopped

None of these prove a session stopped: `kernels status`, a replacement's `COMPLETE` state, a
verified stop marker, or a W&B run whose state reads `running`.

```bash
# time_reserved is non-zero exactly while a session holds an accelerator reservation
mcp: get_accelerator_quota   ->   gpu_quota.time_reserved / tpu_quota.time_reserved
```

Verify the target session ID is terminated, corroborated by a W&B heartbeat that
has gone stale and a `progress/global_step` that has stopped advancing. An account-wide
`time_reserved == 0` is supporting evidence, not a requirement when another session is running.
A W&B run whose kernel
was killed never executes `wandb.finish()`, so its state stays `running` forever — an orphan,
not evidence of live hardware, and not repairable through the read-only API. Record the
discrepancy rather than editing history.

Note that `time_reserved` is per account. Two sessions can run at once on one account (measured
2026-09-04), so a non-zero reservation may belong to the *other* run. Inspect the target session
instead of inferring the number or identity of live sessions from reserved time.

## Artifact warning

Kaggle commits notebook output only when a version terminates. A hard stop can discard
checkpoints, metrics, logs, and the executed notebook. Unless the user asks to stop immediately,
say what will be lost **before** they click:

* how many rounds are committed vs. in flight (read it from W&B `progress/completed_round`);
* whether a clean-stop deadline (`max_hours`) would land sooner than the manual stop anyway —
  if the run is 20 minutes from its own budgeted stop, waiting keeps every artifact.

Never delete the kernel as a substitute for cancellation.

## Do you actually need to stop it?

Check before asking the user to interrupt work:

* **Two sessions can run concurrently on one account.** Measured 2026-09-04: a 50-round training
  run and a 4-round resume ran side by side on `minhtran0601`, both `RUNNING`, both progressing.
  Wanting to start something else is not a reason to stop what is running.
* **Quota is the real constraint, not concurrency.** `get_accelerator_quota` shows
  `total_time_allowed - time_used - time_reserved`. If the free remainder covers the new run,
  launch it and leave the old one alone.
* A run near its own `max_hours` will stop cleanly on its own and commit everything. Read the
  deadline before proposing a stop that throws artifacts away.

## Reporting the outcome

Say which source you checked and what it returned. "The kernel reports `COMPLETE`" is not the
same claim as "the session stopped", and on 2026-09-03 the two disagreed for 18 minutes. When
the sources disagree, report the disagreement; do not pick the convenient one.

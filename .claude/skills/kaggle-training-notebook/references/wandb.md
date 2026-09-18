# Weights & Biases on a Kaggle run

[`observability.md`](observability.md) starts from "Kaggle publishes a notebook's output only
when the kernel stops." W&B is the one exception: an instrumented run streams metrics out
**while it is still running**. That changes operations more than anything else in this skill —
a diverging run is now visible at step 500 instead of after an 8-hour kernel finally stops.

It does not change the artifact rules. W&B holds scalars; checkpoints, predictions, confusion
matrices and `history.csv` still only appear when the kernel commits. Never report a final
number from W&B that has no pulled artifact behind it.

## 1 — Credentials

Order of preference:

1. **Kaggle secret** — `UserSecretsClient().get_secret("wandb_key")`. Nothing lands in source.
2. **Inline key in a private notebook** — only with the owner's recorded, explicit acceptance
   that the key will live in Kaggle's private version history forever. Read it from `~/.netrc`
   with `scripts/embed_wandb_key.py`; never print it, never put it in a shell argument.

   **Standing authorization from the owner of this environment (2026-09-07):** embedding the
   key directly is approved for **any** project this owner builds notebooks for, on the stated
   grounds that the notebooks are private and run locally. This replaces the earlier
   project-scoped exception; it no longer needs re-asking per project.

   Preconditions, which are part of the authorization rather than optional extras: the target
   `kernel-metadata.json` must carry `is_private: true` and the generator must assert that flag
   before writing; the key is read from `~/.netrc` at build time, never printed, never passed as
   a shell argument; the generated `.ipynb` files go in `.gitignore`.

   The cost, which must be restated each time the option is used rather than assumed known: the
   key becomes **permanent** in Kaggle's private version history, sharing or publishing the
   notebook leaks it, and rotating the key means rebuilding and re-pushing every notebook that
   carries it. How to undo it is §1.1 — read that before promising the key can be taken back.

   The authorization is for this owner's own credentials on their own accounts. It is not
   consent to embed a key handed over for someone else, and not consent to make such a
   notebook public.

   Prefer the secret when one is attached and fall back to the inline key, so the same
   notebook works from the editor and from an API push — and **pass the key to
   `wandb.login` as an argument**, never only through the environment:

   ```python
   if not SECRET.get("login") and _INLINE_WANDB_KEY:
       os.environ["WANDB_API_KEY"] = _INLINE_WANDB_KEY     # for wandb.Api() later
       wandb.login(key=_INLINE_WANDB_KEY, anonymous="never", relogin=True)
   ```

   **`WANDB_API_KEY` + `relogin=True` is not a login.** `relogin=True` tells wandb to throw
   away whatever credential it holds and authenticate again, and that path does not consult
   `WANDB_API_KEY` — it looks for a netrc file, then prompts. A Kaggle kernel has neither a
   netrc nor a terminal, so the env-var-only form dies with `UsageError: No API key
   configured. Use 'wandb login' to log in.` On a developer machine the same code passes,
   because `~/.netrc` is sitting there from a previous `wandb login` and silently satisfies
   the lookup. This cost a burned kernel version here on 2026-09-07 and is invisible to
   every local test until you delete the netrc.

   Reproduce it locally before trusting any W&B auth path, by making the machine look like
   a kernel — no netrc, no terminal:

   ```python
   os.environ["HOME"] = tempfile.mkdtemp()      # no ~/.netrc under here
   os.environ["WANDB_CONFIG_DIR"] = os.environ["HOME"]
   os.environ.pop("WANDB_API_KEY", None)
   ```

   Then assert both that the old form raises and that `wandb.login(key=...)` returns True
   and `wandb.Api().default_entity` resolves.
3. **Private credential dataset** — a separate fallback, also requires authorization.

**The trap that costs a launch:** a Kaggle **CLI/API push cannot attach an editor-selected
secret** (confirmed again in 2026-09: `save_notebook` has no secrets parameter in its schema) to the batch version it creates. Enabling `wandb_key` in *Add-ons > Secrets* and then
pushing again produces another version with no secret. Two attempts were burned that way. Either
the owner launches from the editor with **Save Version > Save & Run All**, or you use the
documented inline-key exception. Pushing a third time is not a fix.

Locally, `wandb login` puts the key in `~/.netrc` and `wandb.Api()` finds it. Do not keep a
`.env`; if one is used for a single login, delete it afterwards. `WANDB_API_KEY` left in the
environment can shadow a good `~/.netrc` credential — unset it before debugging an auth failure.

### 1.1 — Removing an inline key safely

Order matters. Only the first step revokes anything; every later step is cleanup, and doing
cleanup first leaves a live key sitting in Kaggle's version history.

1. **Rotate at W&B first** — *Settings > API keys*, revoke the old key, create a new one.
   A pushed Kaggle version cannot be scrubbed: deleting the newest version does not delete the
   older ones, and history lives as long as the notebook does. Revoking at the source is the
   only action that makes an already-pushed string worthless. Do not skip it on the grounds
   that the notebook was private.
2. **Re-authenticate locally**: `wandb login --relogin`, so `~/.netrc` holds the new key.
3. **Rebuild** — `--no-embed-wandb-key` to produce notebooks with no key at all, or a plain
   rebuild to embed the new one. Rebuilding touches local files only; it changes nothing that
   is already on Kaggle.
4. **Push the rebuilt notebooks**, so the newest version no longer carries the old string. To
   return to the secret path, the owner must enable `wandb_key` in *Add-ons > Secrets* from the
   editor — the push itself still cannot attach it.
5. **Scrub local copies**: delete the generated `.ipynb` files and any pulled kernel output
   that embedded them. If the key ever reached a git commit, `.gitignore` does not help; the
   commit still holds it, and a history rewrite plus force push is needed *in addition to*
   step 1, never instead of it.
6. **Verify**: grep the repo, the staging dirs and any pulled outputs for the old key's first
   eight characters and confirm no match survives outside `~/.netrc`.

Rotate **between** runs. A revoked key can start failing the metric uploads of a run that is
already streaming; training itself is unaffected because W&B sits off the training path, but
the tail of that run's history can be lost.

## 2 — Fail fast, before the expensive part

Put `wandb.login` + `wandb.init` **before** parquet decode, cache build and model construction.
A missing secret or disabled Internet must cost seconds, not the 20 minutes of data preparation
that precede round 0. Both aborted attempts here stopped correctly in cell 3.

```python
WB_RUN = wandb.init(
    entity=CFG.wandb_entity, project=CFG.wandb_project,
    id=CFG.wandb_run_id,          # STABLE id -> a resumed kernel continues the same run
    name=CFG.run_name, config=asdict(CFG),
    resume="allow", mode="online", force=True,
    dir=str(RUN_DIR / "wandb"),
)
WB_RUN.define_metric("progress/global_step")
WB_RUN.define_metric("*", step_metric="progress/global_step")
(RUN_DIR / "wandb_run.json").write_text(json.dumps({
    "entity": WB_RUN.entity, "project": WB_RUN.project, "run_id": WB_RUN.id,
    "run_path": f"{WB_RUN.entity}/{WB_RUN.project}/{WB_RUN.id}", "url": WB_RUN.url}, indent=2))
```

`wandb_run.json` is the locator every later session and every monitor call reads. It carries
identity only — **never the secret**.

A fixed `id` with `resume="allow"` is what makes a cross-session resume land in one run instead
of scattering across three. It also means a *relaunch with no checkpoint* appends to the same
history: `progress/global_step` will restart at a low value while `_timestamp` keeps climbing.
Read history in timestamp order, not step order, when a run may have been relaunched.

## 3 — The logging contract

Stable keys, so a monitor script written once keeps working:

| key | when |
|---|---|
| `progress/round`, `progress/round_step`, `progress/global_step` | every heartbeat |
| `progress/completed_round` | once per finished round |
| `train/loss`, `train/skip_pct`, `train/grad_norm` | every heartbeat |
| `train/global_batch` | every heartbeat — **required if batch varies by round** |
| failure-mode quantities (`train/Wqkv_norm`, `train/abs_logit_max`, `train/softmax_entropy`, `train/max_abs_weight`) | every heartbeat |
| `round/duration_s`, `eval/<metric>` for all 10 metrics | once per finished round |

Import `METRIC_KEYS` from the notebook's metrics module and construct the evaluation payload
from the same dictionary written to per-round JSON/CSV:

```python
WB_RUN.log({"progress/completed_round": rnd,
            **{f"eval/{key}": metrics[key] for key in METRIC_KEYS}}, step=global_step)
```

All 10 keys are required; logging only accuracy and macro/weighted F1 is incomplete.
Use direct indexing so a missing metric cannot silently disappear. W&B snapshots are interim
evidence with a capture timestamp; the final notebook and `report.md` tables use pulled artifacts.

Cadence: **one heartbeat every few hundred optimizer steps**, the same window as the JSONL
heartbeat, and one round summary. Here 250 steps was ~77 s — frequent enough to catch a
step-500 explosion, rare enough that the host reads cost nothing.

Log a batch-size key whenever a schedule changes the batch. Without it the step axis silently
changes meaning halfway through the chart and rows/second becomes uncomputable after the fact.

Keep W&B on the **same window** as the on-device accumulators, so the host reads are shared:

```python
def emit(rec, wandb_payload=None, global_step=None):
    with HEART.open("a") as fh: fh.write(json.dumps(rec) + "\n")
    print("  " + json.dumps(rec), flush=True)
    if wandb_payload is not None:
        WB_RUN.log(wandb_payload, step=global_step)
```

W&B is an **addition** to the JSONL heartbeat, never a replacement: the JSONL file is committed
with the run output and survives an account, network or quota problem. `step=` must increase
monotonically or W&B drops the row.

Log from **rank 0 / the single SPMD process only**. Under DDP every rank would open its own run.

## 4 — Reading a run back

Use [`../../wandb-training-monitor/scripts/inspect_run.py`](../../wandb-training-monitor/scripts/inspect_run.py)
with an exact `entity/project/run_id`; never guess the newest run.

Two API traps, both hit here:

* **`scan_history(min_step=...)` takes a W&B step value, not a row offset.** With heartbeats at
  step 250, 500, 750 …, `lastHistoryStep - 40` discards nearly the whole history and hides every
  relative trend. Scan all rows, then tail them in Python.
* **Rows are sparse.** A round-summary row contains no `train/*` keys. Coalesce forward across
  rows before reading "the latest value", or a round boundary looks like a stall.

## 5 — Liveness: three sources, and none of them alone is authoritative

This is the lesson that cost the most confusion in this project.

| source | what it actually reports |
|---|---|
| `kaggle kernels status` | the state of the newest **batch version** — *not* whether an accelerator session is alive. A stop-notebook pushed as a new version reads `COMPLETE` while the previous session keeps running. |
| W&B `run.state` | the last state the client wrote. A hard-cancelled kernel never runs `wandb.finish()`, so the run stays `running` **forever** — an orphan. |
| `get_accelerator_quota` → `time_reserved` | account-wide accelerator reservation; cannot identify which of several notebooks is running. **Measured 2026-09-07: it stayed `0s` throughout a confirmed-RUNNING session.** Do not use `time_reserved > 0` as a liveness test. |
| `get_accelerator_quota` → `time_used` | increments **live** while a session runs (504s → 1251s across two polls of one running kernel). Two rising readings a few minutes apart are real evidence that *some* session is burning quota — the closest thing to a liveness signal quota offers, still account-wide. |

**Parse the status string, not the output.** `kaggle kernels status` prints the state as
`KernelWorkerStatus.RUNNING`, but a transient CLI or auth failure makes the wrapper write its
own line to stderr — under conda, `ERROR conda.cli.main_run:execute(148): ... failed`. A watch
loop that greps the combined output for a bare `ERROR` reports a healthy run as dead, and the
natural next move — pulling logs, re-pushing — is exactly the wrong one. Match
`KernelWorkerStatus\.[A-Z_]*` and treat "no match" as *unknown*, not as failure:

```bash
kstate() {
  kaggle kernels status "$SLUG" 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z_]*' | head -1
}
case "$(kstate)" in
  KernelWorkerStatus.RUNNING|KernelWorkerStatus.QUEUED|"") ;;   # keep waiting
  *) echo "terminal: $(kstate)"; exit 1;;
esac
```

This cost a false "training died" report here on 2026-09-07 while the kernel was running
normally. A monitor that cries wolf is worse than no monitor: the next real alarm is discounted.

Match the exact session/version and W&B run. A fresh `_timestamp` with rising
`progress/global_step` is evidence of recent training progress; correlate it with session status.
Quota can corroborate account activity but cannot establish per-run liveness. Report a W&B/Kaggle disagreement as a
disagreement — say which source says what — rather than picking one silently.

## 6 — What to alarm on

Compare a run against **its own earlier points**, not against absolute thresholds; an absolute
gradient norm means nothing without the trajectory.

* state `failed` / `crashed` / `killed`;
* `running` with no new row past the stale interval — cause unknown until quota confirms it;
* any non-finite logged value;
* skip percentage above the notebook's own abort threshold;
* gradient norm or attention `|logit|max` orders of magnitude above the first heartbeats;
* softmax entropy collapsing toward 0;
* **loss rising and staying up** — the failure that no tripwire in this project catches, because
  the gradients stay finite and `skip_pct` stays at 0;
* a completed round with no `eval/*` metrics.

The skip-rate tripwire fires only on *non-finite* gradients. A run whose gradient norm climbs to
1e12 while remaining finite reports `skip_pct: 0` and trains happily on noise. Watch the
trajectory, and add a heartbeat-level abort on the failure-mode quantity when you know it.

## 7 — Budget the run against remaining quota, not against a constant

`max_hours` is a clean-stop deadline; it only helps if it fires **before** Kaggle stops the
session. Read `get_accelerator_quota` immediately before launching and set

```
max_hours < (total_time_allowed - time_used)/3600 - one round - eval + commit margin
```

A run launched with `max_hours = 8.0` against 7.86 h of remaining quota is hard-killed 13
minutes before its own clean stop, and uncommitted artifacts are lost. Log the derived deadline
into `config.json` and W&B config so the number is auditable afterwards.

## W&B run state is not evidence that a kernel is alive or dead

Measured 2026-09-09. A resumed session had, in the W&B API:

```
state       = crashed
startedAt   = 2026-09-09T03:57:03Z
heartbeatAt = 2026-09-09T03:57:05Z     # two seconds later, then 17 minutes of silence
```

which reads exactly like a process that died immediately after `wandb.init()`. The kernel was
`RUNNING` the whole time: it was in the multi-minute resume import and data decode, with no round
finished yet and therefore nothing to log.

Two rules follow:

* **Liveness comes from the platform** — `kaggle kernels status <owner>/<slug>` — never from a W&B
  run's `state` or `heartbeatAt`. W&B is for reading metrics, not for deciding whether to relaunch.
* A `resume="allow"` run **keeps the previous session's terminal state** until the new session logs
  something. Seeing `finished` or `crashed` on a run you just resumed says nothing about the new
  session; the first metric at the next step is the real signal that the resume worked.

The corollary for a long chain: the meaningful check after a resume is not "is the run alive" but
**"did the first logged step continue the sequence, or restart at 1?"** A resumed session that
silently restarts is the failure that costs a whole session.

Seen again 2026-09-11 on two resumed runs at once, with one more detail: `heartbeatAt` kept
advancing for the first ~4 minutes of the new session and froze at the moment the training
worker processes were spawned, so the run turned `crashed` **while** the system-metrics stream
(`run.history(stream="system")`) was still delivering a row every 15 s with both GPUs busy. The
state flipped back to `running` on its own about 8 minutes after the first round was logged. So
the system stream is a third liveness signal that outranks `state`/`heartbeatAt` — and the
kernel's own live stdout (`kaggle kernels logs -f <owner>/<slug>`) outranks all of them.

# Logging a Kaggle run you cannot watch

**Kaggle publishes a notebook's output only when the kernel stops.** `kernels status` returns
a state and nothing else; `list_notebook_session_output` on a running kernel returns
`{"log": ""}`. There is no live log, no partial artifact, no way to look inside a run in
progress. Verified repeatedly.

Everything below follows from that one fact. You cannot watch, so the run has to (a) leave a
complete story behind and (b) decide for itself when to stop.

The one way to watch anyway is to stream metrics to Weights & Biases — see
[`wandb.md`](wandb.md). It removes the blindness, not the rules: W&B holds scalars, so the
self-stopping tripwires and the committed JSONL heartbeat below stay exactly as they are.

## Rule 1 — the run must be able to kill itself, fast

A tripwire that only fires at the end of a round is nearly useless when a round is 40+
minutes. Check inside the round, on a window of steps:

```python
LOG_EVERY, ABORT_AFTER, ABORT_PCT = 250, 1500, 60.0

if (s + 1) % LOG_EVERY == 0:
    ...
    if s + 1 >= ABORT_AFTER and win_skip / LOG_EVERY * 100 > ABORT_PCT:
        emit({"event": "abort", "reason": "skip rate", "step": s + 1})
        break
```

Give it a grace period (`ABORT_AFTER`) so a noisy start does not kill a healthy run. In this
project a round-level tripwire cost 3,422 s per doomed run; a windowed one would have cost
~400 s.

Tripwires worth having, cheapest first: non-finite reduced `train_loss`; a majority of steps
discarded by the non-finite-gradient guard; a monitored quantity crossing a threshold you set
in advance (here, attention `|logit|max`); and the wall-clock budget.

## Rule 2 — write a JSONL heartbeat, not just prints

`print` goes into a log you must parse out of JSON-wrapped stream records. A JSONL file lands
in the output directory as a first-class artifact, survives an ERROR exit, and is one
`pd.read_json(..., lines=True)` away from a plot.

```python
HEART = RUN_DIR / "logs" / "heartbeat.jsonl"
def emit(rec):
    with HEART.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("  " + json.dumps(rec), flush=True)
```

Always `flush=True`: a killed kernel loses whatever is sitting in the buffer.

## Rule 3 — log the quantity that explains the failure mode, not just the loss

Loss tells you *that* something went wrong, never *what*. Pick the internal quantities whose
drift is the failure you actually fear, and log them from step 0 so you have the trajectory
when it matters. For a transformer that means at minimum:

| quantity | why |
|---|---|
| `\|\|W_qkv\|\|` | attention logits grow with it; it is the leading indicator |
| `\|logit\|max` inside attention | the thing that actually saturates |
| softmax entropy | collapses to ~0 before the gradients explode |
| gradient norm | the symptom everything else predicts |
| `max\|w\|` over all parameters | catches growth outside attention |
| steps skipped in the window | silent under `GradScaler`, and it hides everything |

That last row matters more than it looks: `GradScaler` skips non-finite steps and **reports
nothing**. A run can discard most of its batches and still print a falling loss and a
plausible F1. Count the skips and log them, or you cannot tell training from theatre.

## Rule 4 — one host read per window, never per step

On XLA a host read drains the queue. Accumulate on device and convert once per window:

```python
win_loss += loss.detach();  win_ok += ok.float();  win_skip += (1.0 - ok.float())
if (s + 1) % LOG_EVERY == 0:
    l, o, k = float(win_loss), float(win_ok), float(win_skip)   # the only syncs
```

The exception is a **diagnostic** run, where catching the first bad step is the entire point.
Sync every step there, cap it at a few thousand steps, and accept the slowdown.

## Rule 5 — the results cells must survive a run that produced nothing

A run stopped inside round 0 writes no `history.csv`. Reading it unguarded raises
`FileNotFoundError`, papermill turns that into an execution error, and Kaggle marks the whole
kernel **failed** — even though the tripwire did exactly its job. Two runs in this project
were mislabelled that way.

```python
HIST = RUN_DIR / "metrics" / "history.csv"
if not HIST.exists():
    print("no completed round — nothing to plot.")
    dv = RUN_DIR / "logs" / "diverged.json"
    if dv.exists(): print(dv.read_text())
    for f in sorted((RUN_DIR / "logs").glob("*.jsonl")):
        print(f"--- {f.name} (last 20) ---")
        print("".join(f.read_text().splitlines(keepends=True)[-20:]))
else:
    ...  # the real results
```

Echo the tail of the heartbeat there too: it puts the evidence in the rendered notebook, not
only in a file you have to remember to open.

## Rule 6 — when the cause is unknown, spend a short run on measurement

After three failed training runs, the cheapest next step is not a fourth guess — it is a
20-minute run whose only job is to answer one question. A diagnostic notebook:

* trains a few thousand steps and no more, with no eval and no checkpointing;
* host-syncs every step so it catches the **first** bad step;
* at that step reports *what was already broken* — activations, loss, or only gradients —
  and names the parameters and modules involved;
* stops as soon as it has enough evidence.

Register forward hooks that keep the last output of every leaf module, so the first bad step
can be attributed to a module rather than to "somewhere in the backward".

## W&B `crashed` while the Kaggle kernel says RUNNING is not a hang

W&B flips a run to `crashed` after a few minutes without heartbeat, and a Kaggle container can
lose outbound sync for a long stretch while training continues. Measured 2026-09-14 (pFedES
100c, session 2): heartbeat froze at 00:40Z, state `crashed`, kernel RUNNING; at 01:27Z the run
returned to `running` and the missing round row arrived carrying its original `_timestamp`
(00:40Z) -- nothing had stopped. Before treating a `crashed` state as a dead kernel, wait at
least **two round-times** past the last row and require a second signal (kernel status not
RUNNING, or quota still draining with no row after that window). Cancelling on the W&B state
alone throws away a healthy session and the quota it already spent.

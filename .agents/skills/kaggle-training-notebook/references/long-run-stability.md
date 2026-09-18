# Long runs that die: guards that work, and how to find the real cause

A 50-round run has ~250,000 optimiser steps. Instabilities that are invisible in 2,000 steps
have a quarter of a million steps to compound. Three runs in this project died this way
before the cause was found, costing about 20 GPU-hours.

## Non-negotiable tripwires

```python
if not math.isfinite(train_loss):
    print(f"[diverged] round {rnd}: train_loss = {train_loss}. Stopping.", flush=True)
    (run_dir / "logs" / "diverged.json").write_text(...)
    break                      # BEFORE eval, BEFORE the checkpoint write
```

Break before saving, so `last.pt` keeps the last valid round. Without this, one run trained
20 further rounds on NaN weights: 4.1 h burned, 20 NaN checkpoints, and `last.pt` overwritten
so it could not even resume.

Add a second trigger once a skip-guard exists, or a run can stay finite while learning
nothing:

```python
if n_ok == 0 or skip_pct > 50.0:
    ...  # majority of steps discarded — stop, this round is worthless
```

That one fired in round 0 of a TPU run at 72.06% skipped and saved eight hours.

Under DDP, take `train_loss` from an `all_reduce(SUM)` so every rank holds the same value and
breaks together — no extra collective, no deadlock.

## "Skipping a step" is not just skipping `opt.step()`

CUDA's `GradScaler` skips the optimiser step when gradients are non-finite. XLA has no
equivalent; build it, on device, with no host read so the queue never drains:

```python
grads = [p.grad for p in model.parameters() if p.grad is not None]
gnorm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))   # ONE fused op
ok    = torch.isfinite(gnorm)
scale = torch.where(ok, torch.clamp(clip / (gnorm + 1e-6), max=1.0),
                    torch.zeros_like(gnorm))
for g in grads:
    g.nan_to_num_().mul_(scale)        # nan_to_num FIRST: 0 * nan is nan, not 0
```

`_foreach_norm` matters: the per-tensor version of this guard cost ~78 ms/step, 40% of the
step time. Verify the scale against `torch.nn.utils.clip_grad_norm_` — it should agree to
`max|Δgrad| = 0`.

Two leaks a test found that reading the code did not:

* **BatchNorm running statistics are written by the FORWARD**, before the gradients reveal
  the step was bad. A poisoned step moved them by 1.47e-01 with the gradient guard in place.
  Roll them back from a snapshot *only if* non-finite activations are actually observed —
  it costs ~3 ops per buffer per step, and in this project no probe ever saw a non-finite
  activation, so it was removed again.
* **AdamW's decoupled weight decay still shrinks weights when the gradient is zero**, by
  `lr*wd` relative. A sync-free guard cannot suppress it. Measure it, bound it, disclose it.

Test the guard by poisoning one gradient and asserting the parameters do not move.

## Finding the cause: replay the last clean checkpoint

Checkpoint forensics ("round 24 clean, round 25 destroyed") localises *when*, never *what*.
To get *what*, load the last clean checkpoint and run real batches through it:

1. **Measure the gradient in fp32.** If it is huge in fp32, precision is a symptom, not the
   cause. Here: grad norm 6.5 at round 23 and **1.6e9** at round 24, with loss, `max|w|` and
   logits all essentially unchanged — forward fine, backward exploding.
2. **Rank per-parameter gradient norms** across the two checkpoints. One module and its
   upstream neighbours will stand out. Here `vit.0.att.in_proj_weight` at 4,168x.
3. **Open that module and watch the quantity it owns** across rounds.

## Attention entropy collapse

The failure mode found here, worth recognising on sight:

```
round   ||Wqkv||     |logit|max   softmax entropy   max prob
    0      21.3            268           0.686       1.0000
   15     112.4          2,393           0.528       1.0000
   21     171.9        374,029           0.249       1.0000
   24     526.5    150,130,144           0.001       1.0000
```

Nothing bounds attention logits: `q·k` grows with `||W_q||·||W_k||`, Adam pushes the same
direction for 250k steps, the softmax saturates, and the backward through a softmax at that
scale produces gradients of 1e9. A pre-norm ViT does not prevent it — the *input* is
normalised, the *projection weights* are not.

Note `max prob = 1.0000 from round 0`: the attention was degenerate from the start. Report
that; it changes what can be claimed about the module.

**Fixes, least invasive first:**

1. **Raise `weight_decay`.** Measured from a mid-collapse checkpoint, 4,000 steps per arm:

   | wd | `\|\|Wqkv\|\|` | `\|logit\|max` | entropy | loss |
   |---|---|---|---|---|
   | 1e-4 | 171.9 -> 175.1 | 3.74e5 -> 4.48e5 | 0.225 | 0.0291 |
   | 1e-2 | 171.9 -> 168.0 | 3.74e5 -> 3.57e5 | 0.258 | 0.0201 |
   | 5e-2 | 171.9 -> **155.8** | 3.74e5 -> **2.47e5** | **0.329** | 0.0299 |

   Only 5e-2 reverses it, at no cost in loss. If the paper does not specify weight decay,
   this is a free knob and the right one to reach for first.
2. **QK-norm** — bounds logits by construction, but changes the architecture.
3. Lowering the LR delays; it does not prevent.

Always A/B a stability fix from a checkpoint that is **already failing**. Starting from
initialisation tells you nothing: 2,000 steps from init showed `||Wqkv||` 14.2 -> 15.0 and
0% non-finite gradients in every precision, while the real failure needed ~150,000 steps.

## Large batch is not free throughput

On dispatch-bound hardware, step time is flat and throughput *is* batch size — so a bigger
batch looks like free speed. It is not: it cuts the optimiser steps per epoch by the same
factor.

Batch 16,384 instead of 4,096 gave 2,627 steps per epoch instead of 10,509, and round-0
`f1_macro` fell from **0.83081 to 0.47856** on a round that completed cleanly. Reason about
the statistics and the hardware together, never the hardware alone.

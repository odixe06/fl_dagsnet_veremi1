# Making federated rounds fast on 2× T4

Read with [`federated-afpha.md`](federated-afpha.md) (the scientific contract) and
[`perf-2xT4.md`](perf-2xT4.md) (the hardware budget). This file covers the case those two
do not: **many short, independent single-client training jobs**, where the per-step cost is
dominated by kernel launches rather than by arithmetic. Every optimization below preserves
the FL semantics exactly — same batch, same step count, same tail batches, same math.

All numbers here were measured in this repo on **DAGSNet (395,024 parameters, input `(B, 66)`,
11 positions × 6 channels)** on an **RTX 3050 Laptop**, except §9b, which is the T4×2
confirmation. They are a *ratio* guide for the T4s, not a T4 timing. Re-measure on the target
device before quoting anything.

## 1 — Work out the step count first; it decides everything

A federated round is `Σ_i ceil(n_i / batch_i)` optimizer steps, not one epoch of one model.
Compute it from the partition sidecar before writing any code:

```python
steps = sum(math.ceil(c["rows"] / batch) for c in client_stats["clients"])
hours = steps * ms_per_step / 1000 / gpus * rounds / 3600
```

For VeReMi at 43,045,415 train rows: **84,083** steps/round at batch 512 (20 and 50 clients)
and **168,200** at batch 256 (100 clients). Over 50 rounds that is 4.2 M and 8.4 M optimizer
steps. At the naive 26.8 ms/step this is 15.6 h and 30 h of pure GPU time for two of the three
scenarios — over the weekly quota before a single evaluation runs. **A 3–4× step-time win is
not a nice-to-have here; it decides whether the experiment fits at all.**

Halving the batch does *not* halve the step cost on a launch-bound model. Measured at
`reduce-overhead`: batch 256 = 5.12 ms, batch 512 = 7.17 ms. The 100-client scenario therefore
costs ~1.4× the 20-client scenario in wall time, not 1.0×, purely because a smaller batch
doubles the step count at nearly the same per-step price.

## 2 — Diagnose launch-bound before optimizing anything

Sweep the batch size and read the throughput column, not the step column:

| batch | ms/step | samples/s |
|---:|---:|---:|
| 128 | 5.50 | 23,269 |
| 256 | 5.12 | 50,016 |
| 512 | 7.17 | 71,389 |
| 1024 | 14.06 | 72,830 |

Flat step time across 128→256 means the device is idle waiting for the CPU: **launch-bound**.
Flat *throughput* across 512→1024 means the device is now saturated: batch 512 is already the
efficient operating point and a larger batch buys nothing (see "Spare VRAM is not spare
compute" in `perf-2xT4.md`).

Two corollaries that contradict the usual advice — and note that the second one reverses
again once CUDA graphs are in place, which is the whole reason to fix the launch bottleneck
before tuning anything else:

* **The AMP verdict flips with the regime, so measure it twice.** Launch-bound and eager,
  fp32 beat fp16 AMP (19.33 vs 20.78 ms/step): AMP added a cast per tensor and the tensor
  cores had nothing to do. With CUDA graphs (§5) the order reverses and AMP wins clearly
  (7.20 vs 8.85 ms at batch 512; 4.27 vs 5.10 at batch 256). Benchmarking AMP *before*
  fixing the launch bottleneck would have produced exactly the wrong decision.
* Adding workers, `pin_memory` or a bigger `prefetch_factor` fixes nothing, because there is
  no DataLoader in the loop at all (§4).

### Diagnosing it on a run you cannot stop: read util TOGETHER with the clock

The batch sweep needs a machine you control. On a live remote job you have telemetry instead,
and **GPU utilization alone will mislead you** — it counts intervals in which any kernel was
resident, not whether the device was doing useful work. Read it against the SM clock:

| util | SM clock | meaning |
|---|---|---|
| ~100% | **below** max boost | compute-saturated; the power cap is holding the clock down |
| well under 100% | **at** max boost | **starved** — the GPU boosts precisely because it keeps going idle |

Measured in this repo, all three runs pinned at the same 67 W cap and 77 °C on 2× T4:

| scenario | steps/round | util | SM clock | round time |
|---|---:|---:|---:|---|
| 20 clients, batch 512 | 84,083 | 100% | 1290 MHz | 522 s, flat to ±3 s over 20 rounds |
| 50 clients, batch 512 | 84,098 | 100% | 1245 MHz | 530 s, flat |
| 100 clients, batch 256 | **168,200** | **58%** | **1575 / 1590** | 738 → 903 s, drifting |

Same cap, same temperature, so it is not thermal; peak VRAM was constant and the round time
was non-monotonic, so it is not a leak in the training code. Halving the batch doubles the
step count, and therefore the Python-side launch work, against a fixed **4 vCPU** allocation —
CUDA graphs remove most of the per-kernel cost but not the per-step interpreter cost.

Two practical consequences:

* **A launch-bound scenario is also the noisy one.** The starved run's per-round time wandered
  by ±15% while its siblings held within 3 s, because a CPU-bound process competes with
  everything else on the box. Do not read that variance as instability in the model.
* **Budget the small-batch scenario separately.** Here it cost ~1.7× the round time of the
  large-batch scenarios, not the 2× the step count suggests nor the 1× a naive estimate
  assumes. Measure it; do not scale from the other scenarios.

## 3 — Put the proximal term in the gradient, not in the graph

FedProx's local objective is `CE(w) + (mu/2)·||w − w_t||²`. Written literally in Python it
costs ~400 extra kernels per step over 118 parameter tensors and **9.6 ms/step — 36% of the
whole step**:

```python
loss = ce + (mu / 2) * sum(((p - g) ** 2).sum() for p, g in zip(params, w_global))   # DON'T
```

Its gradient is `mu·(w − w_t)`, a closed form. Add it directly, after `unscale_` so it is in
true gradient units and not multiplied by the GradScaler scale:

```python
scaler.scale(loss).backward()
scaler.unscale_(opt)                                    # gradients now unscaled
d = torch._foreach_sub(params, w_global)                # fp32 params, fp32 anchor
torch._foreach_add_(grads, d, alpha=mu)                 # exact FedProx, 2 kernels
torch.nn.utils.clip_grad_norm_(params, clip)            # clip sees CE + proximal, as intended
```

Verified in this repo against the autograd form with the dropout mask and BN input buffers
held identical on both paths: **max |Δgrad| = 7.45e-09, relative 3.8e-08** — fp32 rounding
only. Cost: 26.78 → 17.14 ms/step, a **1.56×** win for a mathematically exact rewrite.

**The verification trap.** A first comparison reported a 43% relative difference and looked
like a formula bug. It was not: two forward passes of a model with `Dropout(p=0.1)` in train
mode draw different masks, and BatchNorm mutates `running_mean/var` on the first pass. Any
gradient-equivalence test on a training-mode model must reseed the RNG *and* restore the BN
buffers between the two paths — restore them **after** `backward()`, since `load_state_dict`
during the forward bumps version counters and autograd then refuses to run.

Do not divide the proximal term by the parameter count, and do not apply it to BN
`running_mean`, `running_var` or `num_batches_tracked` — those are buffers, not parameters.

## 4 — Keep every client on the GPU and delete the input pipeline

Tabular FL partitions are small. 43 M rows × 66 float16 = **5.68 GB**, plus 43 MB of uint8
labels — the *entire* training set of any of the three scenarios fits in one T4 alongside the
test shard (0.71 GB), the model, Adam state and activations, at roughly 7 GB of 15 GB usable.

Load it once per session, push it to **both** GPUs, and index it. There is no DataLoader, no
worker process, no host-to-device copy in the training loop, and — because both GPUs hold
everything — the scheduler can hand any client to whichever GPU is free (§6).

```python
Xg = torch.from_numpy(X_f16).to(dev)           # (43_045_415, 66) fp16, once per session
Yg = torch.from_numpy(y_u8).to(dev).long()
lo, hi = client_span[cid]                      # clients stored as contiguous row ranges
g = torch.Generator(device=dev); g.manual_seed(hash_seed(seed, rnd, cid))
perm = lo + torch.randperm(hi - lo, generator=g, device=dev)
for i in range(0, len(perm), batch):           # NO drop_last: the tail batch is a real step
    idx = perm[i:i + batch]
    xb, yb = Xg[idx], Yg[idx]
```

Store the features as **float16**, and check the cast is safe rather than assuming it:
z-scored features are near zero-mean unit-variance, so the risk is a large outlier, not
underflow. Assert `max|x| < 65504` and log the observed maximum in the prepack step. Record
in the report that features are stored at fp16 precision — under fp16 autocast the
convolutions consume fp16 inputs anyway, but the stored value is genuinely quantized and that
is a documented choice, not a free lunch.

Fall back to per-client streaming from a host `mmap` only if measured free VRAM says the
resident copy does not fit. Never hold all clients *and* the full test set in host RAM as
float32 on a 30 GB box.

## 5 — CUDA graphs are the single biggest win

`torch.compile(model, mode="reduce-overhead")` captures the forward and backward into a CUDA
graph and replays it, removing the per-kernel launch cost that §2 identified:

| batch 512, with proximal | ms/step | vs naive |
|---|---:|---:|
| eager, proximal through autograd | 26.78 | 1.00× |
| eager, proximal added to gradients | 17.14 | 1.56× |
| `torch.compile` default mode | 13.12 | 2.04× |
| **`torch.compile(mode="reduce-overhead")`** | **7.17** | **3.77×** |

Numerics checked: compiled versus eager logits **max |Δ| = 1.65e-05, argmax agreement 100%**.

`reduce-overhead` is dismissed in `perf-2xT4.md` because CUDA graphs interact badly with DDP
and `GradScaler`. **That reasoning does not apply here**: independent FL clients never form a
process group, so the DDP objection is void. The GradScaler objection survives and is handled
by keeping `unscale_`/`step`/`update` outside the captured region — only the model's
forward/backward is compiled.

Rules that make it hold up over a 50-round run:

* **Capture once, reuse for every client.** Compile the module, not a closure over one
  client's data. `load_state_dict` writes in place with `copy_`, so parameter and BN-buffer
  addresses survive the round boundary and the graph stays valid.
* **Run the tail batch eagerly.** Every client's last batch is a different partial size and
  would trigger a recompile — 20/50/100 recompiles per round. Keep a reference to the eager
  module (`compiled._orig_mod`) and use it for that one step. Same weights, same math.
* **Prove it, then fall back.** `torch.compile` is lazy, so wrapping the call in `try` catches
  nothing. Run one real forward+backward inside the `try`, compare logits to eager, and revert
  to the eager module on any exception or on a logit difference above tolerance. sm75 Triton
  is the documented risk; the fallback is what makes it safe to enable by default.
* **Clone anything you keep.** Graph outputs live in a static pool and are overwritten by the
  next replay. Clone logits before storing them.
* Pair it with `torch.optim.Adam(..., fused=True)`, which collapses the optimizer step into
  one multi-tensor kernel.

### Four traps in the validation harness itself

Both of these rejected a working `reduce-overhead` build in this repo before they were fixed.
A validation harness that does not mirror the real step will veto the largest speedup you have.

1. **The trial must call `zero_grad(set_to_none=True)` between backwards.** Three consecutive
   `backward()` calls accumulate into CUDA-graph-owned `.grad` buffers and raise
   *"accessing gradient tensor output of CUDAGraphs that has been overwritten by a subsequent
   run"*. The real loop zeroes grads every step and is fine; only the probe was broken.
   Run the whole step in the probe — scale, unscale, clip, step, update, zero_grad.
2. **Do not require 100% argmax agreement on a freshly initialized model.** Its logits are
   nearly tied, so fp16-level noise (`max|Δ| = 4.9e-04 ≈ 2⁻¹¹`) flips ~0.2% of rows and a
   strict test rejects a correct build. Compare only rows whose top-2 gap clearly exceeds the
   observed numerical difference, and require exact agreement there:

   ```python
   top2 = ref.topk(2, dim=1).values
   decisive = (top2[:, 0] - top2[:, 1]) > max(10 * delta, 1e-3)
   agree = (got.argmax(1) == ref.argmax(1))[decisive].float().mean()   # require 1.0
   ```

   Measured on DAGSNet: `max|Δlogit| = 4.88e-04`, agreement `1.000` over 489/512 decisive rows.

3. **Never express an exact-equality acceptance test as a float mean.** `torch.mean` on
   CUDA multiplies by a rounded reciprocal, so a *perfect* match returns
   `0.9999999403953552` for many row counts — 77 of the first 600, including 41, 47, 55,
   61, 82, 83. On CPU it is always exact, so this passes locally and rejects at random on
   the GPU. An `agree < 1.0` test therefore silently falls back to eager and gives up the
   entire CUDA-graph speedup, with a log line that looks like a real numerical failure.
   Count disagreements as integers:

   ```python
   n_bad = int((got.argmax(1) != ref.argmax(1))[decisive].sum())   # require 0
   ```

4. **Turn dropout off on BOTH sides of the comparison — you cannot align Inductor's RNG.**
   This is the most expensive of the four: it disabled compile on every T4 run of this
   project for hours, and the log line it produced accused the compiler.

   Restoring `torch.get_rng_state()` before each probe makes the *eager* side repeat its
   dropout mask exactly. It does nothing for the compiled side: Inductor **functionalises
   RNG** and draws its own Philox offsets, so the two masks can never coincide. The gate then
   measures the distance between two different dropout masks and reports it as compiler error.

   | DAGSNet, batch 512 | `max\|Δlogit\|` | outcome |
   |---|---:|---|
   | `dropout=0.1` (production) | **6.07e-01** | `cannot certify` → falls back to eager |
   | `dropout=0.0` | **7.32e-04** | certifies, 0 flips |

   Note how it interacts with trap 2: the decisive-margin threshold is `10 · Δ`, so a Δ of
   0.607 demands a top-2 gap above **6.07**, which no row has. `n_decisive` collapses to 0
   and the gate refuses to certify *without ever comparing an argmax*. A wide tolerance would
   not have saved it either — the mask difference is real, it is just not the compiler's.

   ```python
   drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
   keep  = [m.p for m in drops]
   for m in drops: m.p = 0.0          # both sides now deterministic
   try:
       ref_z, ref_g = probe(model)
       got_z, got_g = probe(compiled)
   finally:
       for m, p in zip(drops, keep): m.p = p     # also on the exception path
   ```

   Warm up and capture the graph at the **production** `p` before flipping it, so restoring
   `p` reuses the entry Dynamo already compiled instead of forcing a third compile. Verify on
   a real CUDA-graph path, not only on CPU: changing `p` is a guard change and does trigger a
   recompile. The same applies to any other RNG consumer in the forward pass — this is about
   RNG, not specifically about dropout.

**Reproduce the gate on a local GPU before blaming the target architecture.** The failure
above reproduced identically on **sm_86**, which is what killed the standing "Triton is broken
on sm_75" hypothesis in one command. sm75 Triton is a real documented risk, and that made it
an attractive explanation for a gate that had in fact never worked anywhere. A gate that
cannot certify on hardware you trust is a broken gate, not a broken GPU.

**A silent fallback needs a loud field.** The fallback prints to stdout, and a *running*
Kaggle kernel's stdout cannot be downloaded — so a run can spend hours in eager with nothing
to show it. Publish the chosen backend the moment the workers are ready:

```python
wandb_run.config.update({"backend": cfg["backend"]}, allow_val_change=True)
wandb_run.summary["backend"] = cfg["backend"]
```

`wandb.init(config=CFG)` runs *before* the backend is decided, so the initial config cannot
carry it; and excluding it from the per-round `log()` dict leaves no other trace.

**Do not compare a per-step projection against a whole-round wall time.** Doing that here
produced a phantom "the T4 is 1.34× slower than local", which sent two benchmarking sessions
after the wrong thing. A round also contains the KD teacher pass and evaluation, neither of
which shrinks when the training step does. Subtract them first: `(1074 − 221 − 40) / 42041 ≈
19.3 ms/step` matched the local eager measurement almost exactly, and the real anomaly was
that compile was off — not that the GPU was slow.

### AMP skip statistics: a warm-up is not a divergence

`GradScaler` is reset per client per round, so every client walks its scale down from
65536 and skips a few steps doing so. Two consequences, both of which produced false
alarms here before they were fixed:

* **A skipped step's gradient norm is `inf` by construction** — it overflowed, which is
  why it was skipped — and its loss may be `nan`. Folding those into the round's mean
  turns routine calibration into a fake divergence signal. Accumulate over *applied*
  steps only, and use `torch.where`, not multiplication: `inf * 0` is `nan`.

  ```python
  applied = scaler._scale >= prev
  ce_acc += torch.where(applied, loss.detach(), zero)
  gn_acc += torch.where(applied, gn, zero)
  ```

* **A percentage-only skip threshold fires on healthy clients.** One warm-up skip is 33%
  of a 3-step client and 0.01% of an 11,506-step one. Allow an absolute warm-up budget as
  well: `allowed = max(abs_allowance, ceil(pct/100 * steps))`. Keep a separate, absolute
  rule for "every step was skipped" — that client returned the global weights unchanged
  and must never be averaged in as if it had trained.

## 5b — Never read a scalar from the device inside the step loop

`float(loss)` and `float(grad_norm)` each force a host sync. Measured cost of those two lines:
**8.12 vs 7.20 ms/step, a 13% tax** on every one of 4.2 M steps. Accumulate on the device and
read once when the client finishes:

```python
ce_acc = torch.zeros((), device=dev); gn_acc = torch.zeros((), device=dev)
...
ce_acc += loss.detach(); gn_acc += gn                 # stays on device
ce_sum, gn_sum = float(ce_acc), float(gn_acc)         # one sync per client, not per step
```

`GradScaler.step()` already syncs once internally (it calls `.item()` on `found_inf`), and
that sync is unavoidable with the stock scaler — it is included in every number in this file.
Counting skipped steps must not add another: watch the scale tensor instead of `get_scale()`,
which calls `.item()`.

```python
prev = scaler._scale.clone()
scaler.step(opt); scaler.update()
skip_acc += (scaler._scale < prev)      # halved scale == the step was skipped
```

`_scale` is private and created lazily on the first `scale()` call, so probe it once at
startup (after a dummy `scale()`) and fail loudly if a torch upgrade removes it, rather than
silently reporting zero skipped steps forever.

## 6 — Two GPUs: two persistent workers, dynamic scheduling, fixed-order aggregation

One process per GPU, spawned once for the whole run, each holding its resident copy of the
data and its compiled model. Per round the parent broadcasts the global `state_dict` once and
then hands out client ids from a shared queue.

* **Schedule longest-first (LPT).** Sort clients by rows descending. With one client per GPU
  at a time this bounds the idle tail; sending the largest client last can leave one GPU idle
  for its entire duration.
* **Dynamic assignment is free here** precisely because both GPUs hold all clients (§4). With
  a static split, one slow client strands a whole GPU.
* **Aggregation order must not depend on completion order.** Workers return `(client_id,
  state_dict, stats)`; the parent sorts by cluster then client id and only then accumulates.
  Otherwise floating-point addition order — and the run's reproducibility — is set by a race.
* **Never use one DDP process group across different clients.** They are different models;
  an all-reduce between them is a scientific error, not an optimization.
* Client weights are 395 K × 4 B = **1.58 MB**. Even 100 clients per round is 158 MB of queue
  traffic — do not build a shared-memory scheme for this.
* **Flatten the state_dict before it hits the queue.** A DAGSNet state_dict has 192 entries;
  `torch.multiprocessing` backs each tensor with its own shared-memory file descriptor, so
  100 clients per round is ~19k fds and the process limit is reached mid-run. One flat float
  vector plus one int vector per client is 2 fds, and it makes aggregation a weighted vector
  sum instead of a per-key Python loop. Put the parameters first in the layout so
  `vec[:n_params]` is exactly the block the proximal drift is measured over.

## 7 — Evaluation is cheap; shard it and stop worrying

Measured at batch 16384 under `inference_mode` + fp16 autocast + the compiled module:
**320,296 samples/s**, i.e. **33.6 s** for 10,761,343 test rows on one GPU, **~17 s** split
across two disjoint shards. Against a ~300 s round that is 5% — evaluate every round in full,
as the contract requires, and do not add a subsampling shortcut to "save time".

Accumulate `torch.bincount(y_true * C + y_pred, minlength=C*C)` on each GPU, sum the two
integer matrices on the host, and derive all 10 metrics from the summed confusion matrix.
Integer counts make the shard split exactly order-independent. Keep the shards disjoint and
contiguous so predictions can be written back in test order without padding or a dropped tail.

`y_true` is identical every round: write it once as `preds/y_true.npy` and store only
`preds/round_NNN_ypred.npy` (uint8) per round. That halves a 1 GB per-scenario prediction
budget to ~538 MB for no loss of information.

## 8 — What not to spend time on

* **Batching several clients into one "ensembled" model** with `vmap` or grouped convolutions.
  It is exact in principle and would help a launch-bound model, but it changes BatchNorm
  handling and per-client optimizer state in ways that are hard to audit. At 4–6 h per
  scenario the run already fits; do not buy speed with reviewability.
* **Raising the batch to fill VRAM.** The batch is a scientific parameter fixed by the user's
  specification, and §2 shows batch 512 is already saturating the device.
* **`channels_last`** (4D only) and **flash-attention kernels** (Ampere+, and there is no
  attention here).

## 8b — Resume state belongs inside the atomic commit

"Completion marker last" is not enough if the resume state is written after it. A crash in
that window leaves a marker for round N-1 and a resume state describing round N, and the next
session either asserts out or silently trains round N with the wrong `mu`. Write the resume
state **per round**, before the marker:

```
weights/round_NNN.pt  confusion/  preds/  metrics/  client_log/  history.csv
resume/round_NNN.pt          <- keyed by round, so it can never disagree with the marker
complete/round_NNN.done      <- absolutely last
```

On resume, read `resume/round_{last_complete:03d}.pt`. Files left behind by an unfinished
round are simply ignored instead of being mistaken for current state.

**Copy the previous session's output with `shutil.copyfile`, not `copy2`.** A new Kaggle
session starts with an empty `/kaggle/working` and reaches the previous run read-only under
`/kaggle/input`. `copy2` preserves mode, so every imported file lands `r--r--r--` and the
first rewrite of `config.json` or `history.csv` dies with `PermissionError` — after the
import has already reported success. `copyfile` plus an explicit `chmod(0o644)` fixes it.
This only shows up in a test that actually makes the source read-only; a normal fixture
copy passes.

**A `--require-resume` gate must fail before the decode, not after it.** That means the
fingerprint cannot depend on anything the decode produces. Row counts read from parquet
*footers* are cheap enough to compute first, so the data identity can still be part of the
fingerprint without paying for a multi-minute prepack to discover there is nothing to
resume from.

Two further things verified by simulating a crash rather than by reading the code: a half-
written round is fully redone and its corrupted artifacts replaced, and `history.csv` keyed by
round shows no duplicate row afterwards. And **a redone round is not bitwise identical** to the
attempt that crashed — cuDNN atomics and the AMP scale trajectory make it differ — so never
describe recovery as reproducing the lost round exactly.

Put the total round count in the resume fingerprint. A cosine schedule over `total_rounds`
means resuming a 3-round run as a 5-round run changes the learning rate of every round; the
fingerprint must refuse that, and in this repo it did.

## 8c — A worker killed by the OS never reports

A single long `queue.get(timeout=...)` waits the whole timeout when a worker is SIGKILLed
(the OOM killer is the usual cause) because a dead process puts nothing on the queue. Poll
in short slices and check `proc.is_alive()` / `exitcode` between them; a negative exit code
is a signal number. Measured here: 27 s to a clear error instead of a 2 h hang, with every
committed round intact.

## 9b — What the T4×2 calibration actually returned

Measured 2026-09-07 on 2× Tesla T4, torch 2.10.0+cu128, running the **production driver** on
the real 43,045,415-row 20-client partition with the full 10,761,343-row test set — not a
synthetic loop. Recorded here because it is the only entry in this file that is a T4 timing.

| batch | mode | AMP | ms/step |
|---:|---|---|---:|
| 512 | eager | on | 18.524 |
| 512 | default | on | 14.903 |
| **512** | **reduce-overhead** | **on** | **6.487** |
| 512 | reduce-overhead | off | 10.075 |
| 256 | reduce-overhead | on | 6.028 |
| 256 | reduce-overhead | off | **5.767** |

Three things worth carrying forward:

* **The ratio guide held.** The laptop predicted ~320 s per 20-client round and ~17 s for a
  two-GPU test pass; the T4s returned **319.7 s** and **15–16 s**. Extrapolating a
  launch-bound model across devices by ratio was sound. Compile is still the whole game:
  **2.86× over eager** at batch 512.
* **AMP inverts at small batch.** fp32 beat AMP at batch 256 (5.767 vs 6.028 ms) while losing
  badly at 512 (10.075 vs 6.487). Do not carry an "AMP always wins with CUDA graphs" rule
  across batch sizes — it is a per-batch measurement. Keeping AMP anyway is defensible when
  several scenarios must stay numerically comparable; say that is why, rather than implying
  it was the faster choice.
* **The real step is slower than the micro-benchmark** — 7.245 vs 6.487 ms — because the
  proximal term, the sampler and the per-client boundaries are real work the synthetic loop
  never does. Budget from the driver's own round times, never from the micro-benchmark.

Everything outside the step loop was **negligible**: aggregation 0.0 s, commit I/O 0.0 s, and
the parquet → fp16 prepack 125.9 s once per session (89.2 s train + 36.7 s test) for 43M rows.
Round 1 cost ~58 s more than steady state — compile plus the first eval — so a 3-round
calibration is the minimum that separates warm-up from the round time you budget with.


## 9 — Report the measurement, not the ratio

Every number above is one laptop GPU. On the T4s, run a short calibration first: assert two
Tesla T4s with capability `(7, 5)`, time 200 steps eager and 200 compiled at the real batch,
time one full test pass, and time the parquet → resident-tensor prepack. Write those four
numbers into `config.json` before the first real round, and derive `max_hours` from them
against live quota rather than from this file.

# PyTorch on Kaggle TPU

Everything here was measured on Kaggle in September 2026, not taken from documentation.
Where a number appears, it came from a probe kernel whose output is quoted.

## Decide whether the TPU is worth it *before* porting anything

A TPU is not a faster GPU. It is a machine that is very fast at large dense matmuls and
comparatively poor at many small operations. Porting a model that is dispatch-bound buys
nothing, and you will not find that out from documentation.

**Run this probe first.** Two numbers decide everything:

```python
# one sync per step, warm up, then time steady state — see "How to benchmark" below
for B in (512, 4096, 32768):
    bench(model, B)
```

If **ms/step is flat across that sweep**, the model is dispatch-bound, not compute-bound.
A real measurement of one such model:

| per-core batch | ms/step | samples/s/core |
|---:|---:|---:|
| 512 | 127.8 | 4,008 |
| 4,096 | 128.3 | 31,934 |
| 32,768 | 124.0 | 264,255 |

Sixty-four times the batch, the same step time. Three consequences follow, and they are
the whole decision:

1. **Throughput is set by batch size alone.** Doubling the batch nearly doubles samples/s
   until the plateau ends.
2. **Splitting a small global batch across 8 cores buys nothing.** 8 cores × 512 rows at
   127.8 ms each is 32,000 samples/s — identical to one core doing all 4,096. The cores
   are idle either way, and now you also pay an all-reduce.
3. **If the global batch is fixed by the experiment, the TPU may simply lose.** The model
   above ran at 31,934 samples/s at its required batch of 4,096, against 56,714 samples/s
   measured on 2× T4. The TPU was 1.8× *slower*, and 50 epochs did not fit the quota.

Diagnose *why* with a same-parameter-count dense MLP. If the MLP hits 5.4 M samples/s and
your model hits 264 k at the same batch, the problem is shape, not the chip: short
sequence axes (k ≈ 11), narrow channel counts (< 512), and many small parallel branches
each pay a fixed dispatch cost that the MXU never amortises.

## machine_shape lies, and only the run tells the truth

```json
{ "enable_gpu": false, "enable_tpu": true, "machine_shape": "TpuV38" }
```

`"TpuV38"` is accepted — and Kaggle allocates a **v5e-8**, not a v3-8:

```
TPU_ACCELERATOR_TYPE = v5litepod-8
```

This is the same trap as the GPU side, where every spelling but `"NvidiaTeslaT4"` was
silently coerced to 1× P100. `get_notebook_info` will not tell you either. **Print
`os.environ["TPU_ACCELERATOR_TYPE"]` in the first cell of every TPU notebook** and treat
that, not the metadata you pushed, as the truth.

Measured environment (Kaggle TPU node, Sept 2026):

| | |
|---|---|
| accelerator | `v5litepod-8` — 8 cores, 15.75 GiB HBM each |
| host | 224 vCPU, 396 GB RAM |
| torch / torch_xla | 2.8.0+cpu / 2.8.0, PJRT runtime |
| `/kaggle/temp` | **does not exist** — GPU notebooks that cache there must use `/tmp` (1 TB free) |
| session cap | shorter than the GPU's 12 h — budget for it, and only **one batch TPU session at a time** |

## The multiprocess trap

`torch_xla.launch()` / `xmp.spawn()` fork, and the children inherit an XLA computation
client if the parent already brought one up. Every child then dies:

```
F0902 03:31:24 runtime.cpp:21] Check failed: !g_computation_client_initialized
InitializeComputationClient() can only be called once.
```

Eight simultaneous `SIGABRT`s and no Python traceback. Rules:

- **Never call `torch_xla.device()`, or put a tensor on `xla`, in a notebook cell that
  will later launch multi-core work.** Merely importing `torch_xla` is fine; touching a
  device is not.
- **Launch multi-core training from a separate process**: write the script with
  `%%writefile`, then `subprocess.run([sys.executable, "train.py"])`. The runtime must
  come up fresh there.
- On this image even that was not sufficient — one process already sees all 8 devices
  (`get_xla_supported_devices()` returns `xla:0 … xla:7`), and `launch` still aborted.
  Using all 8 cores wants **SPMD** (`torch_xla.distributed.spmd`) rather than
  spawn-per-core. Confirm against a probe before relying on it.

## Writing the training loop

**bfloat16, never float16.** The TPU's reduced-precision format is bf16, which carries
fp32's exponent range. This is not a detail: an fp16 GPU run of the same model diverged to
NaN mid-training — 131/132 parameters and all 62 BatchNorm buffers destroyed in a single
epoch — precisely the dynamic-range failure bf16 does not have.

```python
with torch.autocast("xla", dtype=torch.bfloat16):
    loss = criterion(model(x), y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
xm.optimizer_step(optimizer)     # gradient all-reduce across cores
torch_xla.sync()                 # ONE sync per step
```

- **No `GradScaler`.** It is a CUDA construct and bf16 does not need loss scaling.
- **`xm.optimizer_step`** performs the cross-core all-reduce. Plain `optimizer.step()` on
  a single core skips it.
- **Static shapes only.** A changing batch or sequence length triggers a recompile, and a
  first compile costs 20–30 s. Pad the last batch instead of letting it be short.
- **Keep `.cpu()` out of the loop.** Every fetch forces a sync. Accumulate metrics in
  device tensors and read them once per epoch.

`torch_xla 2.8` renamed things; the old names still work but warn:

| old | new |
|---|---|
| `xm.xla_device()` | `torch_xla.device()` |
| `xm.mark_step()` | `torch_xla.sync()` |
| `xmp.spawn(fn)` | `torch_xla.launch(fn, args=())` |

## How to benchmark, so the number means something

Two mistakes will hand you a number that is wrong by two orders of magnitude. Both were
made on the first probe of the model above, which reported 8.5 s/step — 69× worse than
the truth.

**Sync once per step, not once per benchmark.** Running 20 training steps under a single
`mark_step()` builds one enormous graph and times its compilation. A training loop syncs
every step; benchmark what you will run.

```python
def one_step():
    ...
    torch_xla.sync()          # inside the step

for _ in range(warmup): one_step()      # first call compiles: 20-30 s, discard it
t0 = time.time()
for _ in range(steps): loss = one_step()
per_step = (time.time() - t0) / steps
```

**Chain dependencies in microbenchmarks.** `for _ in range(30): c = a @ b` is thirty
identical independent matmuls; XLA's common-subexpression elimination is entitled to
collapse them into one. Write `x = x @ b` so each result feeds the next.

Even then, treat raw FLOP/s microbenchmarks with suspicion — the chained-matmul figure on
the node above came out at 1.8 TFLOP/s bf16, two orders below a v5e's nominal rate, while
the end-to-end model benchmark was self-consistent and matched the epoch times that
followed. **Trust the whole-model measurement over the microbenchmark.**

## Cell mechanics

`%%writefile` must be the **first line of the cell**. A comment above it turns the magic
into a syntax error, and the cell fails at run time on Kaggle, not locally. Put the
explanation *inside* the file, below the magic.

## A probe costs minutes; a wrong assumption costs the quota

Four probe kernels totalling **842 s of a 20 h TPU quota** established that the port was
not worth doing. Without them the first evidence would have arrived 19 hours in. Probe
before porting, and probe the configuration you will actually run — not a convenient
approximation of it.

## Batch size is where TPU throughput comes from — and that is the catch

Step time is flat from global batch 4,096 to 65,536 on a dispatch-bound model, so throughput
*is* batch size. That makes a large batch look like free speed. It is not: it divides the
optimiser steps per epoch by the same factor.

Measured, same model, same frozen projection, same mask, both rounds completing cleanly:

| global batch | steps/epoch | round-0 `f1_macro` |
|---:|---:|---:|
| 4,096 | 10,509 | **0.83081** |
| 16,384 | 2,627 | **0.47856** |

Choose the batch from the statistics, then ask whether the hardware suits it — never the
other way round.

## XLA has no GradScaler, and you will need what it does

`GradScaler` on CUDA silently skips optimiser steps whose gradients are non-finite. Nothing
on XLA does that. A bf16 TPU run here died in round 1 where the equivalent fp16 CUDA run
lasted 24 rounds — the difference was entirely that guard.

Build it on device so the queue never drains, and use the fused norm: the per-tensor version
cost ~78 ms/step, 40% of the step.

```python
grads = [p.grad for p in model.parameters() if p.grad is not None]
gnorm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
ok    = torch.isfinite(gnorm)
scale = torch.where(ok, torch.clamp(clip / (gnorm + 1e-6), max=1.0),
                    torch.zeros_like(gnorm))
for g in grads:
    g.nan_to_num_().mul_(scale)      # nan_to_num FIRST: 0 * nan is nan
```

Then add a majority-skip tripwire, or the run stays finite while learning nothing. Ours fired
at 72.06% of round 0's steps skipped and saved eight hours of quota.

Details and the diagnosis workflow: [`long-run-stability.md`](long-run-stability.md).

## Checklist for a TPU training notebook

Ordered so that anything which can fail cheaply fails first.

**Before writing the notebook**
- [ ] Run the go/no-go probe above. Dispatch-bound models can be *slower* on a v5e-8 than on
      2×T4 at the same batch.
- [ ] Choose the batch from the statistics first — steps per epoch is what learning needs —
      then check the hardware suits it. Never the reverse.
- [ ] Answer every speed question on a local GPU first (`local-gpu-probe.md`). Quota is the
      scarce resource.

**Setup cell**
- [ ] `xr.use_spmd()` **before any device access**, then `Mesh` and `mark_sharding`. Not
      `torch_xla.launch` / `xmp.spawn` — they abort with
      `InitializeComputationClient() can only be called once`.
- [ ] Print `os.environ["TPU_ACCELERATOR_TYPE"]`. `machine_shape: "TpuV38"` yields a
      **v5litepod-8**, not a v3-8.
- [ ] Do not call `xm.get_memory_info()` — it raises under SPMD.
- [ ] Cache to `/tmp`, not `/kaggle/temp`, which does not exist on the TPU node.
- [ ] `%%writefile` must be line 0 of its cell.

**Training loop**
- [ ] bf16 autocast, no `GradScaler` (it is a CUDA construct).
- [ ] Cross-entropy in fp32, outside autocast.
- [ ] Build the non-finite-step guard by hand, on device, with `torch._foreach_norm`.
- [ ] Static shapes everywhere, eval batches padded to a constant, one
      `torch_xla.sync()` per step.
- [ ] Accumulate loss and counters on device; one host read per logging window.

**Observability** — see [`observability.md`](observability.md). Kaggle shows you nothing
until the kernel stops, so:
- [ ] JSONL heartbeat every N steps with the failure-mode quantities, `flush=True`.
- [ ] An in-round abort on skip rate, with a grace period — not just a per-round check.
- [ ] Results cells guarded against a run that completed zero rounds, or Kaggle reports the
      kernel as **failed** when the tripwire worked correctly.

**When something is wrong and you do not know what**
- [ ] Spend one short diagnostic run — a few thousand steps, no eval, no checkpoints,
      host-sync every step, forward hooks on every leaf module — instead of a fourth guess.
      Three training runs were lost here to hypotheses that measurement then refuted.

## XLA autocast is NOT CUDA autocast — this is the one that will bite you

**CUDA's autocast keeps `layer_norm` and `softmax` in fp32 and casts only linear/matmul to
bf16. XLA's autocast does not.** Measured on torch 2.13:

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    layer_norm(x).dtype   # torch.float32   <- protected by policy
    softmax(x).dtype      # torch.float32   <- protected by policy
    linear(x).dtype       # torch.bfloat16
```

Why it matters: **LayerNorm squares its input to compute the variance.** bf16 tops out at
3.39e38, so an input above ~1e19 squares to inf and the normalisation returns NaN. Measured
on a real LayerNorm:

| `\|x\|` | bf16 LayerNorm output |
|---|---|
| 1e18 | 4.56 — fine |
| 1e19 | **0.0000** — variance overflowed, all signal gone, still "finite" |
| 1e20 | **NaN** |

Note the 1e19 row: the layer silently returns zeros. No exception, no non-finite check
fires, and the model just stops learning through that path.

This cost three TPU runs here. The symptom was "66–72% of round 0's steps have non-finite
gradients", and it survived every hypothesis — bf16 in general, extreme input rows, weight
decay, the attention collapse — because the actual failure was one op running in the wrong
dtype. A diagnostic run found it in 50 minutes: at step 3469 the first non-finite activation
in the whole network was `vit.0.n2`, a LayerNorm, while every weight was still under 1.2.

**The fix is to mirror CUDA's policy explicitly.** Put the normalisation-heavy part of the
model in fp32 and leave the convolutions in bf16:

```python
def forward(self, x):
    with torch.autocast(x.device.type, enabled=False):
        feats = self.extractor(x.float())      # LayerNorm / softmax / attention
    return self.head(feats)                    # convolutions stay in bf16
```

Use `x.device.type` rather than a literal, so the same module works on CUDA and XLA.

**Rule: never assume an autocast policy transfers between backends.** Before trusting a
model under XLA autocast, print the output dtype of every normalisation and softmax in it.

## Treat "it works on CUDA" as no evidence at all about XLA

The identical configuration — same frozen init, same batch 4096, same bf16 — ran 9,000 steps
on CUDA with **zero** non-finite gradients while failing on XLA from step ~3,469. Same
PyTorch code, same numbers, different autocast policy. Verify on the backend you will run
on.

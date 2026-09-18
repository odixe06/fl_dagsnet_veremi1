# Squeezing the 2× T4 box

For federated training, first read [federated-afpha.md](federated-afpha.md) for the
scientific contract and [perf-federated.md](perf-federated.md) for the speed playbook — many
short independent client jobs are launch-bound, and several rules below invert for them.
The DDP batch multiplication, equal-step and rank-0-only evaluation examples below apply
to ranks training one logical model, not to different clients. Historical timings below
include an older feature extractor and must not be quoted as standalone DAGSNet measurements.
Preserve the user-set per-client batch and all local-epoch rows, including tails.

Budget: 2× T4 (16 GB VRAM each, Turing sm75), ~4 vCPU, ~30 GB RAM, 12 h cap. The CPU is the scarce resource, not the GPU — 4 cores feeding 2 T4s starves them by default.

## Precision

T4 has fp16 tensor cores and **no bf16, no TF32**. Use fp16 AMP with a `GradScaler`; bf16 falls back to slow emulated paths and TF32 flags are no-ops.

```python
from torch.amp import autocast, GradScaler
scaler = GradScaler("cuda")

with autocast("cuda", dtype=torch.float16):
    loss = criterion(model(x), y) / CFG.grad_accum
scaler.scale(loss).backward()
if (step + 1) % CFG.grad_accum == 0:
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.clip)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
```

AMP is not unconditional: on a launch-bound model it can be a net loss (measured fp32 19.3 ms
against fp16 20.8 ms/step on DAGSNet before CUDA graphs). Sweep the batch size first to find
out which regime you are in — [perf-federated.md](perf-federated.md) §2.

Keep the loss reduction and any softmax/log-sum-exp in fp32 — wrap them in `with autocast("cuda", enabled=False)` if a custom loss overflows. AMP roughly doubles throughput on conv and matmul heavy models; on tiny MLPs over flow features it does close to nothing, so measure before keeping the complexity.

## Feeding the GPUs

`cudnn.benchmark = True` whenever input shapes are fixed — a big win for 1D CNNs over fixed-length windows, and a loss if lengths vary batch to batch.

Worker budget is shared across ranks. Two DDP ranks on 4 vCPU:

```python
workers = max(1, ((os.cpu_count() or 4) - world_size) // world_size)   # -> 1..2 per rank
DataLoader(ds, batch_size=CFG.batch_per_gpu, sampler=sampler,
           num_workers=workers, pin_memory=True,
           persistent_workers=workers > 0, prefetch_factor=4 if workers else None,
           drop_last=True)
```

Set `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)` inside each worker process. Without it, two ranks each spawn 4 OpenMP threads onto 4 cores and thrash.

## The move that matters most for network-traffic data

Flow-feature tables are small. A 3 M × 78 float32 matrix is ~940 MB — it fits in 16 GB of VRAM with room to spare. **Put the tensors on the GPU once and index them directly; delete the DataLoader.** This removes host-to-device copies and worker processes from the loop entirely, and on tabular IDS data it is usually a larger speedup than AMP.

```python
# once, per rank, before training
X = torch.from_numpy(X_np).to(device, non_blocking=True)   # float32 or float16
Y = torch.from_numpy(y_np).to(device)
shard = torch.arange(rank, len(X), world_size, device=device)   # this rank's slice

def epoch_batches(epoch):
    g = torch.Generator(device=device); g.manual_seed(CFG.seed * 1000 + epoch)
    perm = shard[torch.randperm(len(shard), generator=g, device=device)]
    for i in range(0, len(perm) - CFG.batch_per_gpu + 1, CFG.batch_per_gpu):
        idx = perm[i:i + CFG.batch_per_gpu]
        yield X[idx], Y[idx]
```

Sanity checks before adopting it: `X.element_size() * X.nelement() / 2**30` must leave ≥6 GB free per GPU for activations; and the shard must be built the same way on every rank so the epoch length matches (otherwise DDP's gradient all-reduce deadlocks when one rank finishes early).

If it does not fit, the next best thing is `np.load(path, mmap_mode="r")` — both ranks then share one page cache instead of holding two full copies in 30 GB of RAM.

## Memory

- `opt.zero_grad(set_to_none=True)` — frees the grad buffers rather than zeroing them.
- Raise `GRAD_ACCUM` instead of `BATCH_PER_GPU` when a larger effective batch is wanted and VRAM is tight. Global batch = `batch_per_gpu × 2 × grad_accum`.
- `gradient_as_bucket_view=True` on the DDP wrapper saves a full copy of the gradients.
- Evaluate under `torch.inference_mode()` with an eval batch 2–4× the train batch — no activations are stored.
- Free the rank-0 eval tensors before the next round (`del`, `torch.cuda.empty_cache()`) so eval peak does not stack onto training peak.
- `torch.cuda.max_memory_allocated()` printed once per round tells the user how much headroom is left to raise the batch size.

## NCCL on Kaggle

```python
os.environ["NCCL_P2P_DISABLE"] = "1"     # PCIe P2P is blocked in the container
os.environ["NCCL_IB_DISABLE"]  = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
```

Set before `mp.spawn`. Without the first line, `init_process_group` hangs until the timeout instead of failing fast — the single most common way a Kaggle DDP notebook burns an hour doing nothing.

Wrap the model with `find_unused_parameters=False` (the default) unless the architecture genuinely skips branches; `True` costs a full graph traversal every step.

## Overlap the round boundary

Checkpointing and evaluation are rank-0-only, so ranks 1..n-1 sit at a barrier. That is fine and correct — but keep the barrier *after* the checkpoint write, and keep the write atomic, so a kill during eval still leaves a valid `last.pt` from this round.

## Don't bother

- `torch.compile` — Triton on sm75 is unreliable and the compile cost eats several minutes of a 12 h budget. Try it only if a round is long and the model is static; keep a `try/except` fallback to eager. It paid off twice in this project anyway: see the measured 1.34x below, and 3.77x with `mode="reduce-overhead"` on a launch-bound federated loop in [perf-federated.md](perf-federated.md).
- Flash-attention kernels — Ampere+ only.
- `channels_last` — helps 4D conv nets, irrelevant to 1D sequence models.

## Measure

Print per round: seconds, samples/sec, peak VRAM per GPU, and the ratio of data-wait to compute time. If data-wait dominates, the fix is on the CPU side (workers, mmap, resident tensors), not the model.

## torch.compile — measured 1.34x on the T4s

Verified end to end in this project: 759.7 s/round eager, **565.7 s/round compiled**, same
model, same batch, same data. The local RTX 3050 predicted 1.33x, so the ratio transfers
across GPU generations even though the absolute times do not.

Why it pays here: the extractor is ~200 tiny bandwidth-bound kernels on `(B,128,11)`
tensors — 47% of the forward — and Inductor fuses them.

```python
model = DDP(model, device_ids=[rank], ...)     # DDP FIRST
model = torch.compile(model)                    # then compile: DDPOptimizer needs this order
```

* **Default mode, not `reduce-overhead` — under DDP.** CUDA graphs bought a further 5%
  locally and interact badly with DDP + `GradScaler`. Not worth a crashed session.
  This is a DDP constraint, not a property of CUDA graphs: independent federated clients form
  no process group, and there `reduce-overhead` was the largest single win measured in this
  project (26.8 -> 7.2 ms/step). See [perf-federated.md](perf-federated.md) §5.
* **Prove it compiles, then fall back.** `torch.compile` is lazy, so a `try` around the call
  catches nothing. Run one trial forward+backward inside the `try`, identically on every
  rank, and revert to the eager module on any exception.
* **Check it does not change the answer:** here `max|logit diff| = 2.5e-05` and argmax
  agreement 100%.
* **Disable dropout on both sides while you check it.** `restore(torch.get_rng_state())`
  makes the eager side repeat its mask, but Inductor functionalises RNG and draws its own
  Philox offsets, so the compiled side never matches. The check then measures two dropout
  masks and blames the compiler: measured `max|Δlogit|` **6.07e-01** at `p=0.1` versus
  **7.32e-04** at `p=0`, on sm_86. This silently disabled compile on every T4 run of one
  project in this repo. Full recipe and the three lessons that came with it:
  [perf-federated.md](perf-federated.md) §5, trap 4.
* **Record which backend won, in a field you can read while the run is alive.** The fallback
  only prints to stdout, and a running Kaggle kernel's stdout cannot be downloaded.
* The first step pays the compile cost; eval at a different batch size recompiles once.

## Spare VRAM is not spare compute

Do not conclude a GPU is underused because peak memory is 6.2 of 14.6 GB. Sweep the batch
size instead: step time **linear** in batch with flat throughput means the device is already
saturated and a larger batch buys nothing. That was the case here at 2,048 rows/GPU, and the
opposite of what the memory headroom suggested.

## fp32 costs 1.52x

Measured on the real model: 146.2 ms/step fp32 against 96.5 ms/step fp16 AMP. Choosing fp32
"to avoid NaN" also discards `GradScaler`, whose silent skip of non-finite steps is real
protection — the run without it died in round 1 where the fp16 run lasted 24 rounds. Fix the
cause of the NaN instead; see [`long-run-stability.md`](long-run-stability.md).

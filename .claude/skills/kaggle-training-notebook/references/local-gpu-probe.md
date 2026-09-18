# Measure on the local GPU before spending Kaggle quota

Kaggle quota is the scarce resource; a laptop GPU is not. Almost every performance question
about a small model can be answered locally in minutes, and the *ratios* transfer even when
the absolute numbers do not.

Verified in this project: `torch.compile` measured **1.33x** on an RTX 3050 (sm_86) and
delivered **1.34x** on Kaggle's 2xT4 (sm_75) — 759.7 s/round eager, 565.7 s/round compiled.

## Check whether there is a local GPU at all

Do not conclude "no GPU" from the torch build. Check the hardware first.

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A CPU-only wheel on a machine that has a GPU is a five-minute fix, not a dead end:

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 "torch==<same version>+cu130"
conda install -y gcc_linux-64 gxx_linux-64          # Inductor needs a C compiler
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
```

Match the version already installed so nothing else in the environment moves.

## Load the REAL model, never a stand-in

Import the same `model.py` the notebook writes, with the same config and the same channel
mask. A toy model with the same parameter count has a different op mix and answers a
different question.

## The four questions worth measuring, in order

**1. Is throughput bounded by compute or by dispatch?** Sweep the batch size and look at the
shape, not the peak:

```
batch  512 ->  31.70 ms    16,154 samples/s
batch 1024 ->  50.50 ms    20,275 samples/s
batch 2048 ->  97.10 ms    21,092 samples/s
batch 4096 -> 190.94 ms    21,452 samples/s
batch 8192 -> 382.72 ms    21,405 samples/s
```

Step time linear in batch and throughput flat means **compute-bound: the device is already
saturated and a bigger batch buys nothing.** Step time flat across a 16x batch range means
dispatch-bound, and there throughput *is* batch size.

**Spare VRAM is not spare compute.** Concluding "only 6.2 of 14.6 GB used, so the GPU is
idle" is wrong, and it was wrong here. The batch sweep is the test; memory is not.

**2. Where does the time go?** Time each submodule separately before optimising anything.
In this project 47% of the forward was the DWT+ViT+GAT extractor — ~200 tiny bandwidth-bound
kernels on `(B,128,11)` tensors — not the four CNN branches everyone assumes are the cost.

**3. Does `torch.compile` help, and does it change the answer?** Both halves matter:

```python
ref = model(x)
compiled = torch.compile(model)          # default mode
got = compiled(x)
# max|diff| 2.5e-05, argmax agreement 100%  -> safe to ship
```

Prefer default mode over `reduce-overhead` when DDP or a GradScaler is involved: CUDA graphs
bought a further 5% here and are not worth a crashed session.

**4. Does the change survive the real precision?** fp32 was 1.52x slower than fp16 AMP
(146.2 vs 96.5 ms/step), which is what ruled it out — not an opinion about numerics.

## Watch host RAM, not just VRAM

An 8 GB box is OOM-killed (exit 137, and it can take the desktop with it) long before a 4 GB
GPU fills. When sampling a large parquet dataset:

* read **row groups**, never `pq.read_table(whole_shard)` — one shard here is 2.7M rows;
* spread the sample across many row groups: rows inside a group are contiguous, so one group
  gave 11 of 16 classes;
* put a hard `max_rows` cap in the loader and state the expected RSS in the script;
* the loader will return **fewer** rows than asked once quotas round down, so compute
  `n_batches = len(X) // B` instead of trusting the requested count.

## Never build a synthetic probe when the real data is on disk

A synthetic probe here planted one row at the observed extreme in all 66 features at once and
reported a 208x channel-scale imbalance. On real rows the same measurement gives **6.7x**,
and the gradient asymmetry it was supposed to prove is **1.5x**. That fabricated number was
written into three files before real data disproved it.

If the dataset is reachable, measure on the dataset.

# Library and runtime pinning

Every failure below cost at least one Kaggle version in this repository. They share one cause:
the library stack a notebook runs against is **not** the one it was written against, and Kaggle
reports none of the differences as errors.

## 1. Pin the image; never inherit "latest"

Kaggle rolls its default Python image forward without notice. A notebook developed against a
local install meets a different torch, CUDA and Python on the worker.

| | local dev box | Kaggle worker (2026-09-07) |
|---|---|---|
| python | 3.12 (conda `nckh`) | 3.12.13 |
| torch | 2.13.0+cu130 | 2.10.0+cu128 |
| GPU | RTX 3050 sm_86 | 2 x Tesla T4 sm_75 |

Record the digest of an image **observed to work** and pin it in `kernel-metadata.json`:

```json
"docker_image": "gcr.io/kaggle-private-byod/python@sha256:<digest>"
```

Keep the digest and its provenance in the repository (`knowledge/runtime.json`), not in a log
file that later reads as history. A generator that reads its runtime config out of a review log
cannot tell a measurement from an anecdote. Re-pin only after a run proves the new image works.

## 2. The accelerator is declared in two places, and both are load-bearing

`kernel-metadata.json` carries `enable_gpu` / `machine_shape`, but the **notebook document
itself** carries `nb.metadata["kaggle"]`. In this repository the notebooks that omitted it came
up with **zero GPUs** while the one that carried it got 2 x T4 on the same account and image.

```python
nb.metadata.update({
    "kernelspec":    {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python", "version": "3.12"},   # the IMAGE's python, not the dev box's
    "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
               "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
})
```

An invalid `machine_shape` is silently coerced to a single P100 — no error, no warning. Assert
the hardware inside the notebook *before* training, and treat the assert as the cheap failure:

```python
n = torch.cuda.device_count()
assert n == 2, f"expected 2 GPUs, got {n}. Set machine_shape=NvidiaTeslaT4."
assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(n))
```

Zero GPUs can also be **account-scoped**: identical code and metadata gave 0 GPUs on one account
and 2 x T4 on another. Before editing code to chase it, re-run unchanged on a second account.
Never "fix" this by deleting the assert or by reinstalling torch on an image that already has CUDA.

## 3. Do not reinstall torch on a Kaggle GPU image

`pip install torch` in a notebook usually pulls a CPU or mismatched-CUDA wheel over a working
CUDA build, and the symptom is `device_count() == 0` — indistinguishable from a metadata fault.
If a package is genuinely missing, install only that package and never its torch dependency.

## 4. `torch.compile` / Inductor

- `TORCHINDUCTOR_COMPILE_THREADS=1` on any box with limited RAM. Parallel compile workers are
  what took a 7.6 GB WSL machine down in this project; Kaggle's 4 vCPU gains nothing from more.
- `Not enough SMs to use max_autotune_gemm mode` on a T4 is a **benign warning**, not a failure.
- Compile must be *proved* equivalent, not assumed: compare compiled vs eager logits on real
  data and count decisive-row disagreements. Measured here: max |Δlogit| ≤ 4.5e-07, 0 disagreements.
- A compile probe must not mutate the model it probes — run it on a copy, or restore BatchNorm
  statistics and RNG afterwards.

## 5. `torch.func` vmap

Three separate failures, all silent-until-thrown:

- `called random operation while in randomness error mode` — the base module was in train mode.
  Call `.eval()` on the meta base.
- `Expected ... Float but got HalfTensor` — `autocast` does **not** reach vmap's batched conv
  weights. Cast the stacked parameters to half once and run with autocast **disabled** inside.
- `Tensor on device meta is not on the expected device` — the meta base must mirror the exact
  structure being evaluated. If BatchNorm is folded away first, fold a CPU model *then* move it
  to meta, or the folded keys go unmatched.

## 6. NumPy / serialization

- `np.save(path, arr)` appends `.npy` unless the name already ends in it, so the usual
  write-temp-then-`os.replace` pattern silently writes the wrong filename. Write through an open
  file handle, `fsync`, then `os.replace`.
- Write checkpoints that load with `weights_only=True`; store `state_dict` tensors, never a
  pickled `nn.Module`. `weights_only=False` is a compatibility crutch and a code-execution path.

## 7. Memory limits

`ulimit -v` is useless against CUDA: the driver reserves a very large virtual address space and
the process dies with `std::bad_alloc` regardless of real usage. Cap **RSS** of the process tree
and hold a floor under `MemAvailable` instead.

## 8. Local shell

`conda run -n <env> python - <<'PY' ... PY` **discards stdout**: an edit script that reports
success prints nothing, and a failed assertion looks identical to silence. Write the helper to a
file and run it, or use the system `python3` when the edit needs no environment packages.

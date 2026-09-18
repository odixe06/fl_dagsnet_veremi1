# Rebuilding a federated method on VeReMi / DAGSNet — the procedure that already works

Read this before opening `fd_ids`, `tinyproto` or `pfedes` for "how did the last project
do it". Everything below was measured on 2×T4 or verified by a test in one of those
repositories; the paths name the copy to take. It covers **any** FL method whose clients
train DAGSNet on the α = 0.5 partitions and are scored on the fixed 10,761,343-row test.

## 1. Start from `pfedes`, not from scratch

`~/nckh/veremi/pfedes/papers/pfedes-yi-2025/proj/` is the most recent generation and already
handles the harder case (per-client personalized models, partial participation). Copy and
adapt these files; the method-specific maths is confined to one module:

| file | reuse | what changes per method |
|---|---|---|
| `model.py` | verbatim (DAGSNet, `build_model`, `build_proxy`) | only if the method needs another head |
| `data.py`, `metrics.py` | verbatim | nothing |
| `<method>.py` (was `pfedes.py`) | keep `layout/flatten/unflatten_into`, `amp`, `select_clients`, the `_epoch` step loop with AMP skip accounting | the loss / update / aggregation formulas |
| `evaluate.py` | verbatim (BN folding, compiled eval template, `eval_model`) | nothing |
| `driver.py` | keep the worker protocol, gates, commit, budget | which state a worker holds (global model vs per-client table), what `check_updates` rejects |
| `ckpt.py` | keep everything; edit `FINGERPRINT_KEYS` and the weights-file schema | the keys that define the method |
| `verify.py` | keep the structure | per-artifact checks for the new schema |
| `scripts/gen_notebook.py`, `validate_notebooks.py`, `run_local_checked.py` | copy; edit the CFG block, `MODULES`, expected CFG values, `run_name` regex | |
| `tests/` | copy `test_units`, `test_smoke_real`, `test_two_workers`, `test_compile_gate`, `test_gpu_local`, `test_ckpt_verify`, `test_notebook_sim` and rewrite the method-specific assertions | |

For a **global-model** method (FedAvg-like, no personalization) `~/nckh/veremi/fd_ids/papers/fd-ids-2025/proj/`
is the smaller template: one flat global vector broadcast per round, eval of one model
sharded across GPUs. Same worker protocol.

## 2. The parts that must not be re-derived

* **Data path.** Parquet → resident fp16 tensors once per session (`data.load_clients` /
  `load_test`, cache under `/kaggle/temp`, manifest written LAST, `cache_ok` re-validated).
  train/ is already z-scored, test/ is raw and gets `scaler.json` exactly once. Both GPUs
  hold the whole train (5.29 GiB) and test (1.32 GiB). List parquet parts explicitly:
  the centralized test ships `.stats.json` sidecars that kill `ds.dataset(dir)`.
* **Mounts.** Never hard-code `/kaggle/input/<slug>`; resolve by sentinel
  (`find_root("train/client_id=000")`, `find_root("upload/test")`). Measured layout is
  `/kaggle/input/datasets/<owner>/<slug>/...`.
* **Workers.** One persistent process per GPU, spawned once; tasks by id over
  `mp.Queue`; payloads as **numpy** (torch tensors on the queue go through `/dev/shm`);
  LPT dispatch (longest client first); aggregation re-sorted by client id; every client
  re-seeds from `(seed, round, client)` so 1 worker ≡ 2 workers bit for bit
  (`tests/test_two_workers.py`). Never a process group between clients.
* **Compile.** `torch.compile(mode="reduce-overhead")` on the module, loss and optimizer
  outside the graph, tail batch on the eager module, weights swapped with `copy_`. Gate
  it on real rows with dropout off on BOTH sides and a decisive-row argmax rule
  (`driver._compile_train` / `_compile_eval`); fall back to eager loudly and publish the
  backend to W&B summary. Two traps unique to multi-module steps (pFedES): call
  `torch.compiler.cudagraph_mark_step_begin()` at the top of every step when more than
  one captured graph runs per step (else an illegal memory access that only appears
  without `CUDA_LAUNCH_BLOCKING`), and raise `torch._dynamo.config.recompile_limit`
  when several module instances share one `forward` code object (9 variants > default 8
  ⇒ the 9th silently runs eager).
* **Eval.** Fold BatchNorm into the convs (exact), load each client into ONE compiled
  template, accumulate `bincount(y*C + pred)` on device, count non-finite logits
  explicitly. Per-client models: split clients across GPUs by index, and **cache the
  confusion matrix of any client whose weights did not change** (unselected clients),
  re-evaluating everyone every N rounds and at the end to prove the cache
  (`cache_mismatch == 0`). That cut 50c/100c eval from 15/30 min to 3 min per round.
  **Pin client c to worker c % W for eval.** The eval is bit-exact within one worker but
  the two workers (separate processes, cuDNN benchmark picks per process) disagree on a
  handful of fp16 rows: splitting eval by list position made the re-check compare GPU 0
  with GPU 1, and the 100c run flagged exactly the clients whose previous eval had run
  on the other GPU (28 of 90 at round 10; ≤ 23 rows of 10.7 M, |Δmetric| ≤ 2e-5). The
  verifier tolerates ≤ `CACHE_TOL_ROWS = 100` rows for such deviations and demands the
  json count match; pinned runs must show 0.
* **Artifacts.** Weights only, `weights_only=True`, one file per round holding every
  model the method keeps (state_dict tensors); resume bundle keyed by round; marker
  absolutely last; history CSVs derived and rebuilt from the per-round JSON; y_true copied
  into the run; `verify.py` recomputes everything from the confusion matrices and rejects
  the 13 tamper cases in `tests/test_ckpt_verify.py`.
* **Budget.** `T0 = time.monotonic()` in the first cell; driver stops when
  `elapsed + 1.15 × worst_round + reserve > max_seconds`; `max_hours` sized to the
  account's free quota, not to the 12 h cap. Read quota with `kaggle_account.py quota`
  right before pushing; reservations are returned when a session is cancelled.

## 3. Numbers to budget with (2×T4, torch 2.10, image digest in `knowledge/runtime.json`)

| quantity | value | source |
|---|---|---|
| plain DAGSNet step, compiled+AMP | 6.2 ms @512, 5.7 ms @256 | tinyproto calibration |
| pFedES phase-step (F on [x̂;x] then G through frozen F), compiled | **12.06 ms** @512 (eager 42.7); compile+gate 264 s | pfedes probe 2026-09-11 |
| eval, one model, full test | **26 s/GPU compiled-folded @16384 (414k rows/s)**; eager-folded 296k; compiled @8192 only 76k (re-record) — use 16384 | pfedes probe |
| pFedES 20c round (20 clients train + eval, 2 GPUs) | 1256 s = train 1000 + eval 256; startup 15.7 min; VRAM 7.3 GiB/GPU | pfedes probe |
| prepack 43 M rows | 90–130 s; startup incl. spawn + compile 12–17 min | fd_ids |
| 100c at batch 256 | CPU-starved on 4 vCPU: util 58 %, round drifts +20 % — budget it separately | fd_ids |
| eager fallback cost | ×2.9 on train | fd_ids |
| weights per round | 1.6 MB per DAGSNet; per-client methods: N × 1.6 MB | |

## 4. Operating sequence (copy-paste)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
H=.agents/skills/kaggle-training-notebook/scripts
python $H/kaggle_account.py quota                      # who can pay; refresh time
python scripts/gen_notebook.py --owner <acct> --clients 20 --probe --max-hours 2   # 2-round probe + T4 micro-benchmark
python scripts/validate_notebooks.py
python $H/embed_wandb_key.py <nb.ipynb> --metadata <kernel-metadata.json> && python scripts/validate_notebooks.py
python $H/kaggle_as.py <acct> -- kaggle kernels push -p papers/<slug>/notebook/20c_probe/   # no account switch
python $H/kaggle_as.py <acct> -- kaggle kernels status <acct>/<kernel-slug>
# watch W&B: summary.backend, cal_*, history rows; Kaggle stdout is invisible while running
python $H/kaggle_as.py <acct> -- kaggle kernels output <acct>/<slug> -p papers/<slug>/runs/pulls/<name> --page-size 200
python scripts/verify_run.py <pulled run dir> --require-rounds 50
```

Continuation on the SAME account: `--session 2 --require-resume --kernel-source <acct>/<slug>`
— its own slug `<slug>-s2` and dir `<K>c_s2`, same `run_name` (fingerprint, W&B id); a
notebook attaching its own output is unverified on Kaggle, so never reuse the slug. On ANOTHER account: upload the pulled run tree as a dataset owned by the new
account (`kaggle datasets create -p <dir> -r zip -t`), CPU-probe the import, then
`--require-resume --dataset-source <acct>/<ds>` — see [multi-account.md](multi-account.md).

## 5. Local test discipline (8 GB WSL, 4 GB GPU)

Every test through `scripts/run_local_checked.py`, one at a time. Fixtures: 2–4 clients of
the 100-client partition, 2–12k rows each, CPU, `compile=False`. Two CPU workers fit only
with lazy `pyarrow` imports and batch ≤ 128; Inductor fits only in a single process
(`tests/test_gpu_local.py`), never parent + CUDA worker. The notebook simulation runs the
generated cells against a fake `/kaggle/input` tree (with sidecars) — CPU for training,
`--gpu` for the calibration cell. sm_86 passing certifies the graph scheme, not sm_75;
the probe on Kaggle is what certifies the T4.

## 6. Personalized models on the GLOBAL test peak after ~2 local epochs (measured 2026-09-13)

Every per-client DAGSNet on this partition (pFedES F_k, TinyProto `clf`) reaches its global-test
ceiling after **one to two local epochs** and then erodes: pFedES v1 50c, f1 by number of times a
client was trained: 0,376 (1) → **0,404** (2) → 0,380 → 0,385 (5) → 0,366 (8) → 0,313 (14);
accuracy 0,47 → 0,50 → 0,39. The cause is the protocol, not a bug: F_k is supervised only by
local labels (every class with per-client f1 < 0,05 had < 1,2 % local prior; corr(log prior,
per-class f1) = 0,50), and the fixed test set has the global prior. Consequences for any
per-client method at C = 100 % (50 local epochs per client):

* **Expect the W&B curve to fall from round ~2** under a constant rate. Say so in the plan
  before the owner watches it; "accuracy decreasing since round 1" is the predicted shape.
* **A per-round cosine schedule stops the drift but does not recover the peak.** Local scan
  (6 clients of 100c, B = 512, 1/10 test subsample): constant 1e-3 f1 0,298 → 0,299 → 0,294 →
  0,290 → 0,284 (r1–r5); cosine 1e-3 → 1e-5 over T = 10 flattens at 0,289 from r3. The plateau
  sits where the constant rate is at the same **cumulative** LR (≈ Σ lr_t / 1e-3 local
  epoch-equivalents), so a 50-round cosine from 1e-3 (Σ ≈ 25) still descends until ~round 25.
  A lower peak (Σ ≈ 2–5) is the lever for a rising curve; it was not measured.
* What does NOT help (probe weights, 4 clients): recalibrating BN on x-only (−0,09 f1: the
  concatenated [x̂; x] BatchNorm is self-consistent), post-hoc logit adjustment by the local
  prior (±0). What helps but changes the protocol: F_k(G(x)) inference (+0,06 accuracy),
  local-prior-weighted metrics from the same confusion (acc 0,80) — owner declined both.
* Implementation: `proj/pfedes.py::lr_at(cfg, rnd)` (`constant` | `cosine`, afpha's formula),
  AdamW built per client per round at that rate, `lr` logged per client and per round,
  verifier checks it. **`rounds` must enter the fingerprint** once a schedule spans the run;
  a continuation session must plan the same T, and a smoke test that wants "3 rounds then
  resume" keeps T fixed and deletes the last round to emulate the session boundary
  (`tests/test_smoke_real.py`: the re-run must be bit-identical).
* Local LR scans that fit the 8 GB box: ONE process (no spawned worker: the driver's parent +
  CUDA worker is ~2,7 GB and the watchdog ceiling is `min(3000, MemAvailable − 2048)` MiB,
  ~2,6 GB when VSCode holds 2,5 GB), 6 clients of the 100c partition prepacked without torch,
  a stride-10 test subsample (matches the stored per-client metrics to 1e-3), eager, B = 512:
  4,6 min/round. cuDNN benchmark noise between identical runs ≈ ±0,005 f1 — do not read
  differences smaller than that.

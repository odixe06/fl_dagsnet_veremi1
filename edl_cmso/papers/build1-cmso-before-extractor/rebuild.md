# Rebuild — build1-cmso-before-extractor

**Paper** · [paper.md](paper.md)   **Run** · `edl_cmso_r50_b4096`
**Notebook** · [notebook/edl_cmso_veremi.ipynb](notebook/edl_cmso_veremi.ipynb)   **Kernel** · `odixe0502/edl-cmso-veremi`
**Status** · complete — 50/50 rounds, results in [report.md](report.md)

## What is being rebuilt

Stages **3–5** of the paper — DWT → ViT → GAT → fusion → CMSO feature selection →
DAGSNet — trained and evaluated on the user's VeReMi NextGen dataset.

Explicitly **out of scope**, by the user's instruction:

- Stage 1, the IoVCipherGuard cryptographic front end (HE, SMPC, AES-256, §4.3–4.6).
- Stage 2, preprocessing (median imputation + Z-score, §4.7) — the dataset already
  ships imputed and standardised, see below.
- Stage 6, AFPHA federated aggregation (§4.6) — this is a **centralized** rebuild,
  one single model, confirmed by the user.

Dropping stage 1 costs no accuracy: the paper's own Table 2 reports encryption
only as a timing column, and no reported metric depends on it.

## Our dataset

**VeReMi NextGen — 16-class centralized**, Kaggle `odixe0502/veremi-nextgen2026-centralized`,
derived from Zenodo record `19665762` (DOI `10.5281/zenodo.19665762`).
One row = one received CAM message, seen from the receiving vehicle's side.

| split | files | rows | size | feature state |
|---|---:|---:|---:|---|
| `train/` | 15 | 43,045,415 | 7.3 GiB | imputed **and standardised** — use as is |
| `test/` | 4 | 10,761,343 | 1.7 GiB | imputed, **not standardised** — apply `scaler.json` |

- **Task** · 16-class, `label ∈ 0..15`, `0` = benign. Class imbalance 41:1.
- **Features** · exactly the 66 columns whose name starts with `f_` (float32),
  grouped as `raw` 15, `position` 4, `geometry` 20, `session` 16, `rate` 5,
  `profile` 6. Everything else — 9 identifiers, 9 raw-context columns,
  `attack_type`, `scenario` — is **leakage** and must never reach the model.
- **Split policy** · by *simulation time*, 64 separate cut points, one per
  (class × scenario); train share within `[0.799945, 0.800083]`. Not random, so
  no message leaks from a session's future into its past. Given, not created by us.
- **No NaN, no Inf** in either split.

Per-class counts, feature dtypes and schema agreement are pending the Step-3 audit
(runs on Kaggle — the data is not on this machine).

### Caveats this dataset forces us to report

Carried straight from the dataset's own "Reporting results" section; every number
we publish is stated alongside these:

1. The split is by **time**, not by vehicle.
2. The benign class comes from an **attack-free stream**, which makes the `rate`
   feature group unusually strong.
3. **3,338,358 ambiguous rows were dropped** upstream (manipulated but unflagged).
4. Sybil activity is partly visible through `f_first_in_session`: **100.00%** of
   `trafficCongestionSybil` rows sit in a flow where every row is a first-in-session
   row (next highest class is `feignedBraking` at 0.20%; `dosAttack` at 0.00%).
   **⚠ By the user's decision no ablation is run** — the dataset is kept exactly as
   shipped, all 66 `f_*` features offered to CMSO. The consequence must travel with
   the result: **a near-perfect `trafficCongestionSybil` F1 in our table is
   substantially attributable to the `session` feature group, not to the model.**
   It is not evidence that DAGSNet detects Sybil behaviour. Quantifying the split
   would need the session-group-removed run, which was not performed.
5. `timeDelayAttack` is the hardest class.
6. **88% of test flows are one message long**, almost all Sybil. We keep them:
   the paper's DAGSNet is a per-row classifier, so no flow is ever assembled and
   nothing is filtered. Filtering short flows here would delete most of a class.

## Deviations from the paper

Tagged **forced** (our data leaves no choice), **hardware** (2× T4), **user**
(the user set it), or **judgement** (the paper allowed either; we chose).

| # | Paper | This build | Why | Kind |
|---|---|---|---|---|
| 1 | CIC-IDS 2017 / CAN / CICIoV2024 | VeReMi NextGen 16-class | the user's dataset | forced |
| 2 | binary (normal / anomaly) reporting | 16-class softmax, `C=16` | our label taxonomy; Eq. (50) already permits it | forced |
| 3 | HE + SMPC + AES-256 front end | omitted | user instruction; affects no reported metric | user |
| 4 | median imputation (§4.7) | omitted — dataset pre-imputed with constant `0.0` before the split | user instruction; a constant fill needs no statistic so it cannot leak | user / forced |
| 5 | Z-score on the whole set (§4.7) | omitted for `train/` (already standardised); `scaler.json` (train-only, `ddof=0`) applied to `test/` at load | user instruction; **never** fit any statistic on test | user / forced |
| 6 | AFPHA federated aggregation | omitted — one centralized model | user decision (confirmed twice) | user |
| 7 | batch size 64 | **4096** global (2048/GPU × 2 T4) | user set it, raised from 2048 to halve the optimizer-step count; LR stays at the paper's 1e-3 (decision #4), so a round now performs half as many updates | user |
| 8 | 100 epochs | **50 rounds × 1 epoch** over the full 43,045,415 train rows | user set it; a round here is one full pass + full-test-set eval + checkpoint | user |
| 9 | 1× RTX 3090 | 2× T4 DDP, fp16 AMP, `NCCL_P2P_DISABLE=1` | Kaggle hardware; T4 is Turing — no bf16, no TF32 | hardware |
| 10 | LR 0.001 "with adaptive decay", decay `unstated` | 0.001, cosine decay over the 50 rounds, 1 warmup round | paper does not state the schedule | judgement |
| 11 | optimizer listed as "CMSO" (Table 1) but §4.9.2 says Adam | **Adam** trains the weights; **CMSO** selects features. The two roles are separated | the paper never reconciles them; only this reading is implementable | judgement |
| 12 | DWT family/level `unstated` | Haar (`db1`), level 1, as a fixed non-learnable Conv1d so it runs on-GPU | smallest choice that satisfies Eq. (19); keeps the pipeline one graph | judgement |
| 13 | `p × p` patches over 2-D input | 66 features → DWT → 66 coeffs → **11 patches × length 6** (66 = 11×6 exactly, no padding). CMSO subsets floored at 12 features so k ≥ 2 | the input is a flat tabular row; the paper never says how it becomes 2-D. k=1 makes the GAT softmax a constant, starving `a_src`/`a_dst` of gradient and hanging DDP's all-reduce | judgement |
| 14 | ViT depth/heads/`d` `unstated` | `d=128`, 2 layers, 4 heads, FFN ×2, dropout 0.1, pre-norm | must be small: 21,018 steps/round × 50 rounds | judgement |
| 15 | GAT neighbourhood `N(i)` `unstated` | fully connected over the 12 patch nodes, 1 layer, 4 heads, LeakyReLU slope 0.2 | Eq. (26)'s all-pairs numerator implies it; 12 nodes makes it cheap | judgement |
| 16 | Eq. (28) concatenates three tensors of different rank | concatenated on the **channel** axis at each of the k token positions → `F ∈ R^{11×384}`; the `X_wavelet` term is its patch embedding | Eq. (28) as printed does not type-check | judgement |
| 17 | backbone depths `unstated`, 1-D vs 2-D `unstated` | four **1-D** backbones over `F` (length 12, 384 ch): DenseNet 3 dense layers growth 32; GoogleNet 2 inception modules; AlexNet 3 conv+pool; SqueezeNet 3 fire modules. Global-pooled, concatenated (Eq. 47), FC → 16 logits | tabular data has no spatial 2-D structure; sizes chosen for the step budget | judgement |
| 18 | CMSO applied after feature extraction (§4.9 follows §4.8) | CMSO applied to the **66 input features** (binary mask, `dim=66`), once, before the 50 rounds | §4.8's features come from a *learned* ViT/GAT that does not exist before training — the paper's ordering is circular. Selecting on inputs is also the only ordering under which its stated benefit ("reduces computational overhead") is real | judgement |
| 19 | CMSO `N`, `T`, `C₁`, `μ`, `σ`, mutation rate, fitness all `unstated` | `N=20`, `T=50`, `C₁=0.2`, `μ=25`, `σ=3`, mutation 0.05; fitness = macro-F1 of a fast surrogate MLP on a 2 M stratified subsample, minus `0.01 × |S|/66` | none are given; the size penalty encodes the paper's "minimizing redundancy" | judgement |
| 20 | Eqs. (31)–(35) are continuous; feature selection is binary | S-shaped transfer `σ(x)` + threshold 0.5 to binarise each candidate | the paper gives no binarisation rule | judgement |
| 21 | 10 binary metrics (Eqs. 51–59): accuracy, precision, sensitivity, specificity, F-measure, MCC, NPV, FPR, FNR, recall | the user's **10 multi-class metrics**: accuracy, precision {macro, micro, weighted}, recall {macro, micro, weighted}, F1 {macro, micro, weighted} | the user set the metric set. The paper's are binary-shaped (specificity, NPV, MCC, FPR, FNR all need a TN that 16 classes do not define without an averaging convention) | user |
| 22 | pruning / quantization proposed | not applied | the paper applies them to no reported number either | judgement |
| 23 | Table 3 contrasts *with* vs *without* feature selection | **one scenario only: with CMSO feature selection** (the paper's proposed method). No without-FS run, no session-group ablation | the user's decision; the compute budget buys one 50-round run | user |
| 24 | — (paper stores nothing per epoch) | `y_pred` persisted every round; **`y_prob` only for the best-`f1_macro` round and the final round** | 50 × 10,761,343 × 16 fp16 ≈ 17 GB would breach Kaggle's 20 GB output cap. Every other metric stays recomputable from `preds/` without retraining | forced |
| 25 | 100 epochs run to completion on one owned GPU | training **stops cleanly at 11 h of session wall clock** and resumes next session from `last.pt`; a round is only started if `elapsed + 1.15 × last_round > deadline` is false | Kaggle hard-kills a batch session at 12 h. Stopping first lets the closing cells run and the output commit normally, instead of relying on a hard kill committing anything. The decision is made by `all_reduce(MAX)` so both ranks break together — a per-rank clock check would deadlock the next collective | hardware |
| 26 | — (single GPU, no collectives) | NCCL `init_process_group(timeout=60 min)`; `torch.cuda.set_device(rank)` moved **before** init | PyTorch's NCCL default timeout is 10 min, and rank 1 sits in the round barrier for the whole of rank 0's 10.7 M-row eval + artifact write. Exceeding it aborts the run with no traceback. Setting the device after init makes every rank build its communicator on `cuda:0` | hardware |
| 27 | — (paper reports no throughput) | **one** parquet decode pass instead of two: row counts read from the parquet footers, then audit statistics and the fp16 memmap written from the same decoded block | `/kaggle/temp` is wiped every session, so the second decode was paid again on every resume. No statistic changes — the audit is computed on the same float32 block, before the fp16 cast | judgement |
| 28 | — | rank 0 holds the **whole test set resident on its GPU** as fp16 (≤ 1.42 GB) and slices it; `eval_batch` 8192 → 32768; softmax probabilities accumulate in a GPU buffer and reach the host only on the rounds actually saved; `y_prob` written uncompressed | evaluation was host-bound, not model-bound: it fancy-indexed a memmap and copied host→device once per batch, every round, on 4 vCPU shared with the other rank. Rank 1 waits in the barrier for all of it | judgement |
| 30 | — | a superseded `y_prob` file is deleted when a new best replaces it, leaving the two deviation-24 promises exactly: best and final | the implementation wrote one 344 MB file per *improving* round and never removed the old one, while its own docstring and deviation 24 said "best + final". f1_macro improves on most early rounds, so 50 rounds would have left up to 16 GB of superseded copies. Found mid-run; the running attempt 3 keeps them (bounded by the round count at 16.7 GB of 19.5 GB free, so not a failure risk) and the redundant files are simply not pulled |
| 29 | — | the 10 metrics are computed **once** per round and passed into the artifact writer | they were computed twice: once for the `is_best` decision, once inside `save_round_artifacts` — three redundant sklearn passes over 10,761,343 rows every round | judgement |

Nothing in the paper's method has been silently changed. Rows 12–20 exist only
because the paper leaves the value unstated — every one of them is listed in
[paper.md § Gaps](paper.md#gaps-in-the-paper).

## Round definition

**Shape: epoch block.** `EPOCHS_PER_ROUND = 1`, `ROUNDS = 50`.

One round = one full pass over all 43,045,415 train rows at global batch 2048
(10,509 optimizer steps), then rank 0 evaluates the **entire** 10,761,343-row test
set, writes all metrics, and checkpoints. The checkpoint holds the single model —
there is no aggregation step, because this build is centralized.

Fifty rounds is therefore equivalent to 50 epochs, with a full-test-set evaluation
after each.

## Config

*(filled in verbatim from the notebook's `CFG` block once the notebook is written)*

## Results

*(filled after each pull)*

One scenario: **CMSO feature selection enabled**, 16-class, full test set.

| Metric | round 49 — final, delivered | round 2 — peak, **selected on the test set** |
|---|---:|---:|
| accuracy | 0.795341 | 0.842970 |
| precision_macro | 0.791254 | 0.842282 |
| precision_micro | 0.795341 | 0.842970 |
| precision_weighted | 0.810798 | 0.844544 |
| recall_macro | 0.810740 | 0.809816 |
| recall_micro | 0.795341 | 0.842970 |
| recall_weighted | 0.795341 | 0.842970 |
| f1_macro | 0.793345 | 0.816758 |
| f1_micro | 0.795341 | 0.842970 |
| f1_weighted | 0.788977 | 0.832646 |

In single-label multi-class classification every prediction is exactly one class,
so $\sum_c FP_c = \sum_c FN_c$ and therefore
$P_{\text{micro}} = R_{\text{micro}} = F1_{\text{micro}} = \text{Accuracy}$;
`recall_weighted` equals accuracy for the same reason. Five of the ten columns
collapse to one number — that is correct, not a bug. The two that actually
separate models on 41:1-imbalanced traffic are **f1_macro** (every attack class
counts equally) and **f1_weighted** (reliability under the real class mix).

The paper's own headline numbers (Table 3, accuracy 0.99125 / F-measure 0.98455)
are **binary** classification on CIC-IDS 2017 / CAN. Ours are 16-class on a
time-split V2X dataset with a different metric convention. They are not
comparable and are deliberately not placed in the same table.

Raw `y_true` / `y_pred` / `y_prob` are persisted per round, so any further metric
— the paper's binary-style set included — is recomputable without retraining.

## Pre-flight verification

Run locally in the `nckh` conda env against `torch 2.13.0+cpu` before any Kaggle push —
GPU quota is too scarce to debug shapes on. Both suites pass.

`smoke.py` — component level:

- Haar DWT preserves energy (orthonormal) and pads odd lengths; Eq. (19) shape holds.
- `PatchEmbed` gives k=11, pad=0 at 66 features.
- GAT attention rows sum to 1 across all heads — Eq. (26) normalisation is correct.
- Full forward+backward at n_features ∈ {66, 45, 33, 20, 12}: logits `(B,16)`, ~824 k
  parameters, **every** parameter receives gradient (no dead branch → DDP can keep
  `find_unused_parameters=False`).
- fp16/bf16 autocast path, with softmax forced back to fp32.
- The metric collapse identity holds exactly, and a class the model never predicts still
  occupies its macro column.

`smoke2.py` — end-to-end rehearsal on synthetic data:

- CMSO on a problem with a **planted** answer (only features 0–17 carry signal): fitness
  rises monotonically 0.497 → 0.928 over 50 iterations, recovers **100%** of the planted
  features at 75% purity, and honours the 12-feature floor.
- One full round: shard load → 156 steps → full-test-set eval → all 9 artifacts →
  checkpoint. Loss stays finite and sits at ln 16 = 2.773 on random labels, which is the
  correct value for label-free data.
- Eval covers every test row with no dropped tail; softmax rows sum to 1.
- Resume lands on round 1 with weights bit-identical to the saved model, and the
  fingerprint guard refuses a resume after `n_selected` changes.

Two real bugs were caught and fixed here rather than on Kaggle:

1. `PatchEmbed` was sized on `n_features` while `HaarDWT` returns `n + (n % 2)` — any
   odd-sized CMSO subset would have crashed the reshape mid-run.
2. A subset below `2 × patch_len` collapses the token axis to k=1, making the GAT softmax
   constant and leaving `a_src`/`a_dst` gradient-free — a silent DDP all-reduce hang.
   Fixed with a `MIN_FEATURES = 12` floor in `binarise`, plus an assert in the launch cell.

## Run history

| Attempt | Date | Rounds done | Outcome | Fix applied |
|---|---|---|---|---|
| 0 | 2026-08-31 | — | pre-push hardening + throughput work, no GPU spent | accelerator resolved to `machine_shape: "NvidiaTeslaT4"` (2× T4 sm_75) after 6 spellings measured; deviations 25–26 added |
| 1 | 2026-08-31 | 0 | **failed**, cell 4, ~2 min GPU | `assert cands` fired: the notebook globbed `/kaggle/input/*/train` and `/kaggle/input/*/*/train`, but an attached dataset mounts at `/kaggle/input/datasets/<owner>/<slug>/`, and this one keeps the uploader's `upload/` prefix — four levels down. Replaced with a recursive glob requiring parquet inside, and the cell now prints the `/kaggle/input` tree *before* asserting. Accelerator confirmed here: `GPUs: 2 / Tesla T4 sm_75` |
| 2 | 2026-08-31 | 0 | **failed**, cell 6, ~3 min GPU | dataset found. `int()` on `label_mapping.json`: its top level holds three redundant views (`label_to_class`, `classes`, `class_to_label`), not a flat `{index: name}`. Now reads `label_to_class` and asserts the other two agree. Same local check caught a silent one: the schema's group key is `groups`, not `feature_groups` — guarded by an `if`, so the per-group CMSO report was being skipped with nothing to show it. Both found by downloading the sidecar JSON and parsing it locally, not by another run |
| 3 | 2026-08-31 | **50/50** | **complete**, 9.74 h wall, one session | prep 16.5 min (one decode pass: train 111.6 s + test 31.6 s; CMSO 845 s). Audit clean — 41.4:1 both splits, 0 NaN/Inf, train:test 0.8000:0.2000. CMSO kept **41/66** at fitness 0.75488 — `session` 13/16, `geometry` 11/20, `raw` 10/15, `position` 3/4, `profile` 3/6, `rate` 1/5. 701.5 s/round across all 50 (spread under 2 s), 61,362 samples/s, peak 4.57 GB of 14.6 GB. Final round 49 f1_macro 0.79335; peak round 2 f1_macro 0.81676. Train loss 0.2610 → 0.0553 while every test metric fell from round 2 on — overfitting under the time-based split, and the cosine decay to LR 0 did not recover it. Only rounds 0 and 2 ever set a new best, so deviation 30's unpruned `y_prob` files never accumulated: 3 files, not 50. All 51 checkpoints pulled and verified loadable |

## Decisions taken

Settled by the user, recorded so they are not relitigated:

1. **Metrics** · the 10 multi-class metrics above only.
2. **Ablations** · none. Dataset kept exactly as shipped; one feature-selection
   scenario. The Sybil/`session` leak is disclosed in caveat 4 instead of measured.
3. **Compute** · ~25 h across ~3 Kaggle sessions accepted. Parquet + on-GPU
   resident tensors may beat the estimate; actual per-round wall time is reported
   from round 1 and this document updated with the measured figure.

# Rebuild — build1-cmso-before-extractor — build 2, the paper's pipeline order

**Paper** · [paper.md](paper.md)   **Build 1** · [../build1-cmso-before-extractor/rebuild.md](../build1-cmso-before-extractor/rebuild.md)
**Notebook** · [notebook/edl_cmso_veremi_v2.ipynb](notebook/edl_cmso_veremi_v2.ipynb)   **Kernel** · `odixe0502/edl-cmso-veremi-v2`
**Run** · `edl_cmso_v2_r50_b4096`
**Status** · 🔴 attempt 1 **diverged at round 25**. Rounds 0–24 valid, peak f1_macro
**0.83081** at round 0. NaN tripwire added. Re-run planned on **TPU v3-8** in bf16 — the
final report aggregates that full run, not this truncated one.

## What is being rebuilt

Stages **3–5** of the paper — DWT → ViT → GAT → fusion → CMSO feature selection →
DAGSNet — on the user's VeReMi NextGen dataset, **in the paper's own order**.

```
build 1   66 raw ──CMSO(41)──► DWT ─► ViT ─► GAT ─► fusion ─► DAGSNet
build 2   66 raw ─► DWT ─► ViT ─► GAT ─► fusion(Eq.28) ──CMSO──► DAGSNet
paper     identical to build 2  (§4.8 → §4.9 → §4.10)
```

Build 1 inverted the order — its deviation 18 — because the paper's ordering is
circular: §4.8's features come from a ViT/GAT that has not been trained when §4.9 is
supposed to run. Build 2 restores the paper's order and resolves the circularity
explicitly. **Everything else the user controls is held at build 1's value**, so the
position of CMSO (together with the Eq. 28 reading, below) is what differs between the
two runs.

Explicitly **out of scope**, unchanged from build 1 and by the user's instruction:
stage 1 (IoVCipherGuard, §4.3–4.6), stage 2 (preprocessing, §4.7 — the dataset ships
imputed and standardised), stage 6 (AFPHA federated aggregation — this is a
**centralized** build, one model).

## Decisions that define this build

Settled by the user on 2026-09-01, before any code was written.

| # | Question | Decision |
|---|---|---|
| A | How the extractor/selection circularity is broken | **CMSO runs on the extractor at initialisation** — no warm-up phase the paper does not describe |
| B | Training budget | **batch 4096, 50 rounds** — identical to build 1 |
| C | Eq. (28)'s `X_wavelet` term | **raw Haar coefficients**, the literal reading; fused = 6 + 128 + 128 = **262** |
| D | Scenarios | **one: with CMSO**. The Table 3 without-FS pair is not run |

### A, and the contract it forces

The user chose the literal reading of the paper's feed-forward order: §4.8 emits `F`,
§4.9 selects among its channels, §4.10 classifies — with no pretraining stage invented to
make the features meaningful first.

That choice is only coherent under one condition, which the notebook enforces. A mask
chosen against one random projection and then applied to a *different* one would describe
nothing at all. So:

1. the probe network is built under `CFG.seed`;
2. its §4.8 parameters (33 tensors, 284,164 values) are frozen to `extractor_init.pt`;
3. CMSO searches the fused map **that** projection emits;
4. `train_worker` loads that exact state at round 0 — verified bit-identical in
   [smoke.py](smoke.py) check 6 — and `ckpt.fingerprint` refuses a resume across a
   changed `n_selected` or `n_features`.

**What this buys.** Because the mask is fixed before training, the extractor spends all 50
rounds learning to route information through the channels that survived. The mask acts as
a structured width reduction of the fusion layer that the network adapts to, not a pruning
applied after the fact.

**What it costs, and what must therefore never be claimed.** At selection time the 262
channels were random projections. The mask is **not** evidence about which learned
features matter, and no statement of that kind may be made from it — not from the
wavelet/ViT/GAT survival counts, not from the fitness value. The measurement that would
support such a claim is selection against a *trained* extractor, which this build does not
perform. This was raised with the user before the decision and accepted.

## Our dataset

Identical to build 1 — **VeReMi NextGen 16-class centralized**, Kaggle
`odixe0502/veremi-nextgen2026-centralized`, from Zenodo `19665762`
(DOI `10.5281/zenodo.19665762`). 43,045,415 train rows / 10,761,343 test rows, 66 `f_*`
features, 16 classes, imbalance 41:1, split by simulation time.

The caveats in [build 1 § Our dataset](../build1-cmso-before-extractor/rebuild.md#our-dataset) apply
here without change, **with one that build 2 makes worse**:

> **⚠ Sybil.** 100.00% of `trafficCongestionSybil` rows sit in a flow where every row is a
> first-in-session row (next class: `feignedBraking` 0.20%). In build 1, CMSO at least
> *could* have dropped `session` columns at the input — it kept 13 of 16. In build 2 it
> cannot: selection happens after fusion, so **all 66 inputs always reach the extractor**.
> A high Sybil F1 here is attributable to the `session` group at least as much as in build
> 1, and this build has even less ability to separate the two. Unchanged from build 1: no
> ablation is run (decision D), so the split is disclosed, not measured.

## Deviations from the paper

Carried from build 1. Rows in **bold** changed; row 18 no longer is a deviation.

| # | Paper | This build | Why | Kind |
|---|---|---|---|---|
| 1 | CIC-IDS 2017 / CAN / CICIoV2024 | VeReMi NextGen 16-class | the user's dataset | forced |
| 2 | binary reporting | 16-class softmax, `C=16` | our taxonomy; Eq. (50) permits it | forced |
| 3 | HE + SMPC + AES-256 | omitted | user instruction; affects no reported metric | user |
| 4 | median imputation (§4.7) | omitted — pre-imputed with constant `0.0` before the split | a constant fill needs no statistic, so cannot leak | user / forced |
| 5 | Z-score on the whole set | `train/` already standardised; `scaler.json` (train-only, `ddof=0`) applied to `test/` at load | never fit a statistic on test | user / forced |
| 6 | AFPHA federated aggregation | omitted — one centralized model | user decision | user |
| 7 | batch size 64 | **4096 global** (2048/GPU × 2 T4) | 64 is 672,584 steps/epoch ≈ 187 h of T4 against a 30 h weekly quota. Held at build 1's value so it is not a second changed variable | hardware / user |
| 8 | 100 epochs | 50 rounds × 1 epoch over all 43,045,415 rows | user set it; held at build 1's value | user |
| 9 | 1× RTX 3090 | 2× T4 DDP, fp16 AMP, `NCCL_P2P_DISABLE=1` | Kaggle hardware; Turing has no bf16/TF32 | hardware |
| 10 | LR 1e-3 "adaptive decay", schedule unstated | 1e-3, 1 warmup round then cosine to 0 | the paper does not state the schedule | judgement |
| 11 | "Optimizer: CMSO" vs "Adam" (§4.9.2) | Adam trains weights; CMSO selects features | the paper never reconciles them | judgement |
| 12 | DWT family/level unstated | Haar (`db1`), level 1, fixed non-learnable stride-2 conv | smallest choice satisfying Eq. (19); stays on-GPU | judgement |
| **13** | `p × p` patches over 2-D input | all 66 features → DWT → 66 coeffs → **11 patches × 6, exact, no padding, every run** | the input is a flat tabular row. CMSO no longer touches the inputs, so `k` is constant at 11 and build 1's `k=1` degeneracy **cannot occur** | judgement |
| 14 | ViT depth/heads/`d` unstated | `d=128`, 2 layers, 4 heads, FFN ×2, dropout 0.1, pre-norm | must be small: 10,509 steps/round × 50 | judgement |
| 15 | GAT `N(i)` unstated | fully connected over the 11 patch nodes, 1 layer, 4 heads, slope 0.2 | Eq. (26)'s all-pairs numerator implies it | judgement |
| **16** | Eq. (28) concatenates tensors of different rank | **raw wavelet patch ‖ z_final ‖ z_new** per token → `F ∈ R^{11×262}` | Eq. (28) does not type-check as printed; this is its literal reading. **Build 1 used the patch embedding (384)** — decision C | judgement |
| 17 | backbone depths / 1-D vs 2-D unstated | four 1-D backbones over the masked `F`, 1×1 stem to 96 ch each: DenseNet 3 layers growth 32, GoogleNet 2 inception, AlexNet 3 conv+pool, SqueezeNet 3 fire; global-avg-pooled, concatenated (Eq. 47), FC → 16 | tabular rows have no 2-D structure | judgement |
| **18** | CMSO applied after feature extraction (§4.9 follows §4.8) | **CMSO applied to the 262 fused channels of Eq. (28)** | **no longer a deviation — this is what build 2 fixes.** The circularity is broken as set out under decision A | — |
| **19** | CMSO `N`,`T`,`C₁`,`μ`,`σ`, mutation, fitness unstated | `N=20`, `T=50`, `C₁=0.2`, `μ=25`, `σ=3`, mutation 0.05; fitness = macro-F1 of a surrogate **1-D CNN** (a miniature DAGSNet) on a train-only subsample − `0.01·|S|/262` | none are given. `N`/`T` held at build 1's values so the optimiser budget is not a second changed variable. **Limitation:** 1000 evaluations now cover 2²⁶² rather than 2⁶⁶ | judgement |
| **20** | Eq. (31)–(35) continuous, selection binary | sigmoid transfer + threshold 0.5; floors of **16 channels overall and 2 per Eq. (28) term** | no binarisation rule is given. The per-term floor keeps `F` a three-term concatenation — see "the floor" below | judgement |
| 21 | 10 binary metrics (Eq. 51–59) | the user's 10 multi-class metrics: accuracy, P/R/F1 × {macro, micro, weighted} | the user set the set; specificity/NPV/MCC/FPR/FNR need a TN 16 classes do not define without an averaging convention | user |
| 22 | pruning / quantization proposed | not applied | the paper applies them to no reported number either | judgement |
| 23 | Table 3 contrasts with/without FS | one scenario: with CMSO | decision D; the remaining quota buys one run | user |
| 24 | — | `y_prob` for the best and final round only; `y_pred` every round | 50 × 10.7 M × 16 fp16 ≈ 17 GB would breach Kaggle's 20 GB output cap | forced |
| 25 | 100 epochs on one owned GPU | stops cleanly at 11 h of session wall clock, resumes from `last.pt`; decision via `all_reduce(MAX)` | Kaggle hard-kills at 12 h. A per-rank clock check would deadlock the next collective | hardware |
| 26 | — | NCCL `timeout=60 min`; `set_device(rank)` **before** `init_process_group` | the 10 min default expires while rank 1 waits out rank 0's 10.7 M-row eval; setting the device after init builds every communicator on `cuda:0` | hardware |
| 27 | — | one parquet decode pass: counts from footers, audit and fp16 memmap from the same block | `/kaggle/temp` is wiped each session, so a second decode is paid on every resume | judgement |
| 28 | — | test set resident on rank 0's GPU as fp16, `eval_batch` 16384, probabilities accumulated on-GPU | evaluation was host-bound, not model-bound | judgement |
| 29 | — | the 10 metrics computed once per round and passed to the artifact writer | they were computed twice: three redundant sklearn passes over 10.7 M rows per round | judgement |
| 32 | — | `torch.compile` (default mode) on the DDP module in run 3 | 1.33x measured on the real model locally (96.5 -> 72.5 ms/step, batch 2048, sm_86); logits agree with eager to max 2.5e-05 and argmax agrees on 100% of rows. `fuse` is ~200 tiny bandwidth-bound kernels on (B,128,11), which is what Inductor fuses. A trial fwd+bwd proves the path compiles, and it falls back to eager otherwise | forced — 50 rounds do not fit 8.29 h of quota otherwise |
| 33 | — | cross-entropy computed in fp32, outside `autocast`, in run 3 | a softmax over 16 half-precision logits can round a probability to exactly 0, and log(0) is -inf. Costs nothing on a (B,16) tensor | judgement |
| 34 | Eq. (28) | the raw Haar patch is concatenated **unnormalised**, and stays that way | `z_final` and `z_new` are both LayerNorm'd; the patch is not. **An earlier version of this row claimed a 207.8x scale gap and named it the root of the divergence — that was wrong on both counts.** The 207.8x came from a synthetic probe that planted one row at the observed extreme in all 66 features at once; on real rows spread over the 15 shards the gap is **6.7x**, and the stem-gradient asymmetry it was supposed to cause is **1.5x**, because the BatchNorm immediately after the stem conv absorbs it. The real cause is upstream of Eq. (28) entirely — attention entropy collapse in `vit.0`, see CONTEXT.md 1i. The unnormalised patch is kept because normalising it would abandon the literal reading of Eq. (28), which is the reading this build exists to test | **owner's decision, 2026-09-02** — "the option closest to the original paper". Disclosed, and no longer suspected |

| 30 | — | a superseded `y_prob` file is deleted when a new best replaces it | leaves exactly deviation 24's promise: best and final | judgement |
| **31** | — (paper trains once, on one GPU) | the §4.8 projection CMSO selected against is frozen to `extractor_init.pt` and **reloaded at round 0** | without it the mask describes a random projection the trained network never had. This is what makes decision A coherent | forced by A |

### The per-term floor (deviation 20), and a claim that was checked rather than assumed

CMSO must keep ≥ 2 channels from each of Eq. (28)'s three terms.

The first reason written for this was **wrong** and is recorded here because the correction
matters: it was assumed that a mask keeping no GAT channel would leave `gat.W`/`a_src`/
`a_dst` without gradient and deadlock DDP under `find_unused_parameters=False` — the same
class of failure build 1 hit from the other direction. Measured on the actual graph
([smoke.py](smoke.py) check 5), that does not happen: `cat` and `index_select` keep all
three terms in the autograd graph, so those parameters receive a **zero** gradient, not
`None`. DDP is safe for any mask.

The real consequence, and the actual reason for the floor: such a mask would freeze the
entire GAT branch at its random initialisation for all 50 rounds while still paying its
forward cost every step, and would reduce Eq. (28) to a two-term concatenation. The floor
costs at most 6 of 262 channels of CMSO's discretion.

## Round definition

**Shape: epoch block.** `EPOCHS_PER_ROUND = 1`, `ROUNDS = 50`, identical to build 1.

One round = one full pass over all 43,045,415 train rows at global batch 4096 (10,509
optimizer steps, 5,255 per rank), then rank 0 evaluates the **entire** 10,761,343-row test
set, writes all metrics, and checkpoints. No aggregation step — this build is centralized.

**Expected cost, and why it exceeds build 1's.** Build 1's CMSO kept 41 inputs → 42 DWT
coefficients → `k = 7` tokens. Build 2 always has `k = 11`. ViT, GAT and all four DAGSNet
backbones scale with `k`, so a round should take roughly 1.4–1.6× build 1's 701.5 s —
about 1000–1100 s. Fifty rounds plus CMSO plus prep is therefore **≈ 15–16 h**, against
19.87 h of quota remaining (refresh 2026-09-05). It will **not** fit one 11 h session: the
run stops cleanly on the wall-clock budget after roughly round 31 and resumes next session
with its own output attached. This is the designed path, not a failure mode.

## Config

*(filled verbatim from the notebook's `CFG` block after the first push)*

Fixed by construction: `n_features = 66` always (the extractor sees every input column),
`fused_dim = patch_len + 2·d_model = 262`, `k = 11`, CMSO subsample 400,000 fit +
100,000 val rows — train-only, down from build 1's 2.3 M because a row of `F` is
262 × 11 fp16 = 5.6 KB against 132 B for 66 raw columns, keeping the resident tensor at
2.9 GB either way.

## Results

*(filled after each pull)*

One scenario: **CMSO feature selection on the fused map**, 16-class, full test set.

### Attempt 1 (GPU, truncated) — superseded, kept for the record

**Not the reported result.** Valid rounds are **0–24 only**; the run went NaN at round 25
(see Run history). The final report aggregates the full TPU run instead.

| Metric | round 0 — peak, and best of the valid range | round 24 — last valid |
|---|---:|---:|
| accuracy | 0.865448 | 0.822820 |
| precision_macro | 0.857528 | — |
| precision_micro | 0.865448 | 0.822820 |
| precision_weighted | 0.858874 | — |
| recall_macro | 0.815819 | — |
| recall_micro | 0.865448 | 0.822820 |
| recall_weighted | 0.865448 | 0.822820 |
| **f1_macro** | **0.830807** | 0.786230 |
| f1_micro | 0.865448 | 0.822820 |
| f1_weighted | 0.856521 | 0.818010 |

Per-class at round 0 — the two classes the dataset card singles out behave exactly as it
warns: `timeDelayAttack` F1 **0.1948** (recall 0.1356, the hardest class by far) and
`trafficCongestionSybil` F1 **0.9669** at precision 0.9800, which is the `session`-group
signature, not evidence that DAGSNet detects Sybil behaviour. `dosAttack` is again the
cleanest at F1 **0.9950** on 1.57 M rows. `positionMirroring` (0.5475) and
`suddenConstantSpeed` (0.6270) are the other weak classes.

**CMSO kept 133/262 fused channels** — wavelet 3/6, ViT 69/128, GAT 61/128, best fitness
0.56789, having moved only 0.55859 → 0.56789 across all 50 iterations. That near-flat
search is the predicted consequence of decision A: selecting against a random projection
gives the optimiser very little to climb. Per the limit stated under decision A, this
distribution is **not** evidence about which learned features matter.

**Measured cost:** 759.7 s/round (min 729.4, max 791.0), 56,714 samples/s, peak 6.21 GB.
That is **1.08×** build 1's 701.5 s, not the 1.4–1.6× predicted here. The prediction was
wrong because it scaled only with the token axis (k: 7 → 11) and ignored that DAGSNet —
which dominates the cost — got *narrower* at the same time (133 input channels against
build 1's 384), nearly cancelling it.

In single-label multi-class classification every prediction is exactly one class, so
$\sum_c FP_c = \sum_c FN_c$ and therefore
$P_{\text{micro}} = R_{\text{micro}} = F1_{\text{micro}} = \text{Accuracy}$;
`recall_weighted` equals accuracy for the same reason. Five of the ten columns collapse to
one number — correct, not a bug. The two that separate models on 41:1 traffic are
**f1_macro** and **f1_weighted**.

**Against build 1.** The two runs share every training hyperparameter, the dataset, the
split, the metric set and the CMSO population. They differ in **two** things — where CMSO
sits, and the Eq. (28) reading — so a per-round `f1_macro` comparison is meaningful but
must be attributed to both, not to the reordering alone.

**Not against the paper.** Table 3 (accuracy 0.99125, F-measure 0.98455) is *binary*
classification on CIC-IDS 2017 / CAN under a metric convention 16 classes do not define.
Not comparable, and deliberately not tabulated together.

## Pre-flight verification

Run locally in the `nckh` conda env against `torch 2.13.0+cpu` before any Kaggle push —
GPU quota is too scarce to debug shapes on.

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
python papers/build2-paper-order/smoke.py
```

**40 checks, all passing.** `smoke.py` extracts the five `%%writefile` modules straight out
of the notebook, so it tests the code that will actually run, not a copy of it.

1. **Eq. (19)** — Haar DWT preserves energy (orthonormal) and pads odd lengths.
2. **Eq. (20)–(21) and decision C** — `k = 11`, `pad = 0` at 66 features; `PatchEmbed`
   returns the raw coefficient patch, asserted equal to `xw.view(B,11,6)`, not a projection.
3. **Eq. (28)** — `fuse()` is `(B, 262, 11)`; channels `0:6` are asserted identical to the
   raw wavelet term.
4. **Masked forward/backward** at `|S| ∈ {262, 131, 40, 17}` — logits `(B,16)`, and all
   131 parameter tensors receive gradient, so `find_unused_parameters=False` is valid.
5. **Frozen-branch guard** — a GAT-free mask leaves `gat.*` with a gradient that is not
   `None` (DDP safe) but exactly zero (branch frozen); no other parameter is affected; the
   `binarise` floor rescues the worst possible candidate; 500 random candidates all honour
   both floors.
6. **The build-2 contract** — `extractor_state` covers only §4.8; a differently-seeded
   network's `fuse()` differs by 6.0 in max-abs; after `load_extractor_state` it is
   **bit-identical** to the probe's; a foreign key is rejected.
7. **CMSO on a planted answer** — fitness 0.679 → 0.888 over 50 iterations, monotone,
   recovers **8/8** planted channels across all three terms at `|S| = 87/262`, and the best
   mask honours the per-term floor.
8. **SurrogateCNN** shapes at `cin ∈ {16, 131, 262}`.
9. **Autocast** — logits finite, softmax rows sum to 1 in fp32.
10. **Launch-cell invariants** — `k = 11`, `fused = 262`, `MIN_CHANNELS ≥ 3 × MIN_PER_TERM`.
11. **Cell 16's chunked extraction** — the 16k-row chunk loop agrees with a single pass to
    fp16 rounding (4.9e-04), while a deliberate one-row shift is 6.29 and would fail the
    same tolerance; the fit/val slices partition the subsample exactly.
12. **Static scan of the notebook** — no top-level cell reads a name nothing has bound.

Two real problems were caught here rather than on Kaggle:

- **A `NameError` that would have cost a whole GPU session.** Cell 16 used `autocast`
  without importing it; every other cell gets the symbol from inside a `%%writefile`
  module, so nothing at top level bound it. It would have fired *after* the ~17 min
  parquet prep. Fixed, and check 12 exists so the class of bug cannot recur — build 1
  lost two sessions to failures of exactly this shape (`assert cands`, then `int()` on
  `label_mapping.json`).
- **An incorrect assumption in the design rationale** — the DDP-deadlock claim behind the
  per-term floor, disproved by measurement and corrected in place. See "The per-term
  floor" above.

## Run history

| Attempt | Date | Rounds done | Outcome | Fix applied |
|---|---|---|---|---|
| 3 | 2026-09-02 | **running** | 🟡 run 3, `edl-cmso-veremi-c50`. fp16 AMP + `torch.compile`, batch 4096, 50 rounds, 8.05 h budget. Inherits run 1's `channel_mask.json` and `extractor_init.pt`, never its checkpoints. **This is the run the final report is written from** | fp32 abandoned (deviation 32 rationale): measured 1.52x slower than fp16, which puts 50 rounds at ~16 h against 8.29 h of quota, and it discards `GradScaler`, whose skip-on-non-finite-gradient is the only reason run 1 lasted 24 rounds |
| 2 | 2026-09-02 | 0 | ⚫ **cancelled by the owner** while still in prep. `edl-cmso-veremi-fp32`, fp32 batch 4096. Superseded before it produced a round | fp32 was chosen to avoid the NaN; measurement showed it neither addresses the cause nor fits the quota |
| 1 | 2026-09-02 | **45 run, 0–24 valid** | 🔴 **diverged.** 10.89 h GPU, stopped on the 11 h budget after round 44. `train_loss` went NaN in round 25 and never recovered; checkpoint forensics show round 24 completely clean (0/132 params, 0/62 BN buffers, 0 optimizer tensors non-finite) and round 25 completely destroyed (131/132, 62/62, 262). Loss had been rising for three rounds first — 0.0506 → 0.0514 → 0.0559 → 0.0771 → NaN — with `max\|w\|` at 17.9. I recorded this as an fp16 dynamic-range blow-up; **that diagnosis was wrong** — see deviation 34 and attempt 3, where a bf16 run died at round 1 and the real 208x scale imbalance in Eq. (28) was measured. **The operational failure was mine:** the loop never checked that `train_loss` was finite, so it trained 20 more rounds on NaN weights — 4.1 h of GPU wasted, 20 NaN checkpoints written, and `last.pt` overwritten with NaN so the run could not even be resumed | **NaN tripwire** added to `train_worker.py`: breaks the moment `train_loss` is non-finite, *before* eval and *before* `save_round`, and writes `logs/diverged.json`. `train_loss` comes from `all_reduce(SUM)` so both ranks hold the same value and break together — no extra collective, no deadlock risk. Next attempt moves to **TPU v3-8 in bf16**, whose exponent range is fp32's and therefore does not have the failure mode that killed this run |
| 0 | 2026-09-01 | — | notebook written from build 1's, decisions A–D applied, 40/40 pre-flight checks pass | `autocast` used but never imported in cell 16 — a `NameError` that would have surfaced after the 17 min prep and cost the session; caught by a new static scan (check 12) that now guards the whole notebook. Assumed DDP deadlock behind the per-term floor disproved by measurement and the justification corrected. fp16 tolerance in the extraction test set where noise ends, with a shifted control to prove it has teeth |

## Decisions taken

1. **Metrics** · the 10 multi-class metrics only. Unchanged from build 1.
2. **Ablations** · none (decision D). The Sybil/`session` leak is disclosed, not measured —
   and build 2 removes even the incidental partial mitigation build 1 had.
3. **Circularity** · decision A, with the `extractor_init.pt` contract that makes it
   coherent, and with the explicit limit on what the mask may be claimed to show.
4. **Compute** · ~15–16 h across 2 sessions, of 19.87 h remaining (refresh 2026-09-05).

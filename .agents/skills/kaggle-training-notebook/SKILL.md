---
name: kaggle-training-notebook
description: Build, validate, resume, and operate PyTorch training notebooks on Kaggle; retrieve artifacts and write training reports with all 10 classification metrics. Use for paper reproduction, Kaggle training/evaluation, or reporting its results. Do not use for ordinary local notebooks or data exploration without training.
---

# Kaggle training notebooks

Read the nearest AGENTS.md, current paper, architecture specification and dataset documentation.
Use `conda run -n nckh` for Python and Kaggle CLI in this repository. State assumptions and
verifiable success criteria. User decisions in this session govern the work.
Read current `CONTEXT.md` for approved settings, account policy and unresolved review findings.
This `.agents/skills` tree is the maintained Codex source; `.claude/skills` entrypoints link here.
Paths under `references/` and `scripts/` resolve from the skill directory unless labelled
repository-root paths. Keep mirrored references/helpers synchronized when modifying them.

Historical build numbers, TPU results, accounts, W&B credentials and exceptions in supporting
references are evidence from other runs, not authorization or defaults for this project.
Check referenced paths/scripts exist. Never infer completed training from copied documentation.

## Route

1. Read [kaggle-mcp.md](references/kaggle-mcp.md), discover callable tools and verify an
   account-scoped read such as quota. Public dataset metadata does not authenticate the account.
   Record MCP and CLI results separately. Missing MCP does not prevent local audits or skill
   edits; respect a user requirement to verify MCP before building models. Never print tokens.
   Before requesting credentials, inspect the user-designated ~/.kaggle/ directory:
   credentials.json identifies the active CLI account; accounts/<username>.mcp-token
   stores its MCP bearer. Follow [saved credentials](references/kaggle-credentials.md).
   Never print tokens, silently switch accounts, or overwrite OAuth credentials with a
   bearer. The current Codex header helper does not require an environment export.
2. Extract the method and unresolved choices into `papers/<slug>/paper.md` and `rebuild.md`;
   see [paper-dossier.md](references/paper-dossier.md). Do not invent missing equations,
   optimizer settings or adaptive rules and attribute them to the paper. Resolve scientific
   choices with the user before implementing dependent training code when requested.
3. For any FL method on VeReMi/DAGSNet read [per-client-fl-veremi.md](references/per-client-fl-veremi.md)
   FIRST: it names the module set to copy from `~/nckh/veremi/pfedes` (or `fd_ids` for a
   global-model method), the measured T4 numbers, the two CUDA-graph traps of multi-module
   steps, the exact eval cache for unchanged clients, and the push/pull command sequence, so
   no sibling project has to be re-read.
   For FL read [federated-afpha.md](references/federated-afpha.md) before using any DDP template,
   and [perf-federated.md](references/perf-federated.md) before optimizing its speed: measure
   whether the loop is launch-bound before applying centralized AMP/DataLoader advice, and keep
   each speed change within justified numerical tolerances against an eager reference.
   Record tolerances and preserve the approved algorithm; AMP/compile need not be bit-identical.
   Audit actual data using [dataset-audit.md](references/dataset-audit.md) and
   [veremi-dataset-layout.md](references/veremi-dataset-layout.md). Distinguish full scans,
   samples and historical sidecar counts. Freeze feature order, class mapping and train-fitted
   preprocessing. Never standardize already standardized training data a second time.
4. Read [library-runtime.md](references/library-runtime.md) BEFORE writing or pushing a
   notebook: pin `docker_image` by digest, declare the accelerator in the notebook document as
   well as in `kernel-metadata.json`, and never reinstall torch on a Kaggle GPU image. The
   library stack on the worker is not the one you developed against, and Kaggle reports none of
   the differences as an error.
   Build using [notebook-template.md](references/notebook-template.md),
   [metrics.md](references/metrics.md), [perf-2xT4.md](references/perf-2xT4.md),
   [perf-federated.md](references/perf-federated.md) for federated loops,
   [local-gpu-probe.md](references/local-gpu-probe.md),
   [long-run-stability.md](references/long-run-stability.md), and
   [observability.md](references/observability.md) as relevant.
   Historical speedups are not measurements of the new model. W&B is optional unless requested;
   read [wandb.md](references/wandb.md) only when using it. The owner of this environment has
   a standing authorization to embed the W&B key in their own `is_private: true` notebooks
   (wandb.md §1); state the permanence cost when you use it, and §1.1 when asked to undo it.
5. Test the exact notebook's generated modules: real-data forward/backward, aggregation,
   all metrics/artifacts, strict model reconstruction, interrupted-round recovery and next-round
   resume. Read [verifying-artifacts.md](references/verifying-artifacts.md) before trusting any
   verifier: checks built on freshly generated random inputs test the formula, not the run, and
   pass on tampered artifacts. It also carries the crash-safe write ordering, the atomic-import
   rules and the data-identity fingerprint that recovery depends on. Small fixtures can verify arithmetic/crash behavior; benchmarks use real data.
   State limits of one-GPU local checks for two-GPU execution.
6. Verify notebook metadata: correct code_file, kernel_type notebook, owner, dataset sources,
   and explicit is_private true. Request machine_shape NvidiaTeslaT4 and enable_gpu true;
   assert exactly two Tesla T4 devices with capability (7, 5) inside the notebook before train.
   Also set `nb.metadata["kaggle"]` in the notebook document and a pinned `docker_image`
   (see [library-runtime.md](references/library-runtime.md) §1-§2): notebooks that carried only
   the JSON metadata came up with zero GPUs. Zero GPUs can be account-scoped -- re-run unchanged
   on another account before editing code, and never respond by removing the assert.
   Read current quota/session limits. Inspect saved accounts with
   [kaggle_account.py](scripts/kaggle_account.py) `list`, `quota`, `health` and `plan`, none of
   which switch accounts. Size the whole training plan with `plan --gpu-hours`, not one session
   at a time, and read [multi-account.md](references/multi-account.md) before `ensure` or `use`.
   Under the **2026-09-10** policy the agent runs those two with `--confirm` itself, without
   asking; announce which account and why. The user's only account action is reconnecting
   the MCP client when a reconnect is actually needed. Budget free hours after used and
   reserved time, including all planned simultaneous runs. Reconnect MCP after a switch and verify
   alignment; use selected CLI OAuth while MCP identity remains uncertain. An account with quota
   is not an account with GPUs -- zero-GPU allocation has been account-scoped here before, and
   only a run proves it. Script creation does not itself authorize launching runs.
7. For operation/recovery read [pull-outputs.md](references/pull-outputs.md),
   [stop-session.md](references/stop-session.md), and only if needed
   [multi-account.md](references/multi-account.md). Distinguish latest batch-version status,
   active session and fresh heartbeat. A replacement push does not prove cancellation.
   Recheck dated API limitations before asserting they still apply.
8. Build reports from verified artifacts with [reporting.md](references/reporting.md).
   A verifier certifies only the rounds that exist: require its explicit completion gate before
   any "the scenario finished" claim (see [verifying-artifacts.md](references/verifying-artifacts.md) §6).

Local tooling note: `conda run -n <env> python - <<'PY' ... PY` **discards the script's stdout**,
so an edit script that reports success prints nothing and a failed assertion looks like silence.
Write helper scripts to a file and run that file, or use the system `python3` for edits that need
no environment packages.
   Include all 10 metrics for every reported round; never fabricate results.

## Scientific and artifact contracts

- Confirm which model(s) are evaluated. Shared client/server architecture does not mean shared
  post-training weights. Global test scores are not personalized performance.
- Unless the user requests another schema, use this ordered METRIC_KEYS list:
  accuracy, precision_macro, precision_micro, precision_weighted, recall_macro,
  recall_micro, recall_weighted, f1_macro, f1_micro, f1_weighted.
  This differs from the paper's binary metric list. Resolve ambiguity explicitly.
- Evaluate the entire fixed test after every completed round without padding/dropped tails or
  test-driven adaptation. Compute metrics from full confusion counts, not batch metric averages.
  Save JSON, CSV, confusion matrices, per-class results, y_true and y_pred. Probabilities are
  optional and require a disk estimate. Keep all 10 columns even when values coincide.
- **A per-round checkpoint stores weights, not a model.** Write a `state_dict` of tensors
  including BatchNorm buffers; never a pickled `nn.Module`, and never the optimizer, scaler or
  scheduler state in the same file. Keep them as `weights/round_NNN.pt` and put the resume
  bundle in a separate `resume/round_NNN.pt`, so retaining every round costs weights alone.
  For DAGSNet that is 1.5 MB a round instead of 4.7 MB — and the weights file stays the thing
  a reader can load without trusting a pickle.
- **Weights are only a checkpoint if current code can rebuild the model at them.** Ship the
  `build_model(cfg)` used by the run, persist the architecture config, feature order, class
  names and scaler beside the weights, and make the rebuild path prove what it restored:
  `torch.load(..., weights_only=True)`, `load_state_dict(strict=True)`, an explicit
  parameter-count assert, and the config fingerprint. `strict=True` is what catches a filtered
  `running_mean`/`running_var`, which otherwise leaves `eval()` normalizing by 0/1 and reports
  no error at all. Exercise that path against a file you actually wrote, in the same test run
  that wrote it; a checkpoint no code reconstructs is not a checkpoint.
- **`weights_only=True` holds only if the file contains tensors and plain types.** One numpy
  RNG tuple, `Path`, or custom object anywhere in the dict forces every future reader back to
  `weights_only=False` — a code-execution path. Keep those in the resume bundle. Verify by
  loading with `weights_only=True` in the test, not by inspecting the dict.
- Resume needs the round, algorithm state, RNG and any persistent optimizer/scaler/scheduler
  states. If local optimizers deliberately reset every round, document this and omit their
  discarded state. Weights alone do not support arbitrary exact resume.
- Commit complete rounds atomically, completion marker last -- and the resume state must be
  inside that atomic unit, keyed by round and written before the marker. A latest-only
  resume_state written after the marker disagrees with it whenever a crash lands in between.
  Resume from the last complete round.
  Keep weights and metrics for every completed round. A crash during eval must not advance last
  or leave duplicate history rows. Estimate storage before launch.
- Persist effective config.json and a fingerprint covering model, preprocessing and scientific
  settings. Validate on resume. Search working output then unique attached outputs; require_resume
  must fail if absent. New sessions wipe working files. Notebook outputs use kernel_sources;
  datasets use dataset_sources.
- Flush logs and guard finite values/AMP skipped steps. Reject invalid client updates before
  aggregation. Finite weights are not enough: a client whose AMP steps were all skipped
  returns the global weights unchanged and finite. Report applied steps separately, compute
  loss/gradient means over applied steps only, and allow an absolute warm-up skip budget so a
  healthy client is not failed for the scaler calibrating itself. Do not silently discard
  failed clients or overwrite valid weights with NaNs -- dropping one changes the
  participation rate the specification fixed. Stop before the measured session budget,
  leaving time to finalize artifacts, and budget on the worst round observed including
  commit I/O rather than the last one.
- A worker killed by the OS puts nothing on the queue. Poll the result queue in short slices
  and check process liveness between them, or a single long timeout turns an OOM kill into a
  multi-hour hang.
- Cross-session resume needs an explicit import step and a require-resume gate that fails
  *before* the data decode; copy the previous output with `copyfile`, since a read-only
  mount's permissions otherwise carry across and break the first rewrite.
- Centralized DDP rules apply when ranks optimize one logical model: equal collective counts,
  device selection before process-group setup, appropriate NCCL settings. Different FL clients
  must not synchronize gradients with one another.

## Completion evidence

Report separately: account-authenticated MCP, measured dataset coverage, local verification,
remote two-GPU verification, training completion and pulled results. One does not prove another.
Name remaining manual actions with exact steps and observed failure. Complete independent
authorized work before handing back; do not describe a partial scaffold as ready to train.

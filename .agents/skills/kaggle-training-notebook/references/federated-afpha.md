# AFPHA with DAGSNet

Read for the 20/50/100-client task. User decisions override proposals and old examples.

## Sources

- Paper: s41598-025-94445-9.md, section 4.6 describes AFPHA as FedAvg + FedProx + HFL.
  It does not specify proximal coefficient, clusters, participation, synchronization schedule
  or an adaptive equation. Table 1 lists CMSO and LR 0.001 with adaptive decay; it does not
  supply an executable FL optimizer. Label mathematical completions as implementation choices.
- Model: architecture/ARCHITECTURE.md, standalone DAGSNet, 66 features, 16 logits,
  395,024 learnable parameters and PyTorch default initialization. The user confirmed fresh
  initialization, not the supplied trained round-5 weights.
- Train: /home/odixe/nckh/veremi/dataset/fl_client/alpha05/{20,50,100}_client/train/client_id=NNN/.
  User confirmed fl_client on 2026-09-06.
- Kaggle: odixe0502/veremi-fl-{20,50,100}client, version 1 observed 2026-09-06;
  relative roots {20,50,100}_client/. Resolve unique runtime sentinels, not a fixed mount prefix.
  Read manifest/partition_config.json and client_stats.json.
- Fixed test: /home/odixe/nckh/veremi/dataset/centralized/test; Kaggle
  odixe0502/veremi-nextgen2026-centralized, upload/test/.
- Train is standardized; test is raw. Reuse matching scaler.json, feature order and labels;
  verify against architecture/meta.json. Only f_* columns enter the model.
- Sidecars report 43,045,415 train rows per scenario and 10,761,343 test rows, 16 classes,
  Dirichlet alpha 0.5. Individual clients may lack classes: check the train UNION against test.
  Disclose global train-scaler access, receiver-unit clients and dataset caveats in reports.

## Confirmed training contract

50 communication rounds, 1 complete local epoch per selected client per round, batch per
client 512 for 20/50 clients and 256 for 100 clients. Fresh DAGSNet initialization.
Evaluate only the post-aggregation global model on the full fixed test using the skill's
10 metrics. Record participation, clustering, proximal anchor/coefficient, adaptive rule,
optimizer/LR schedule, optimizer persistence and BatchNorm aggregation before training.
The user requested a concrete proposed specification for approval before implementation.

## Two GPUs

Prefer two persistent spawned processes, one independent client task per GPU at a time.
Reuse model allocations and schedule large clients early. Reset/load states as the FL spec
requires. Never use one DDP process group to synchronize different clients' gradients.
Per-client batch does not double because two GPUs exist.

Keep a client's features resident on its GPU if measured headroom allows; otherwise stream
bounded batches. Avoid host-RAM copies of all clients plus test. Global eval may use disjoint
deterministic test shards on both GPUs: sum integer confusion counts and restore prediction
order. Rank-0-only eval in the centralized template is not a scientific FL requirement.

Use fp16 AMP with scaling on T4, proximal reductions in fp32. Validate eager versus compile
on this model; preserve weights, BN buffers and RNG around probes. Do not transfer historical
timing claims to this model or treat spare VRAM as proof of idle compute.

With one sample-weighted cluster average and one sample-weighted server average per round,
the result equals sample-weighted FedAvg over all participating clients. State this
equivalence: hierarchy alone does not define an adaptive algorithm. Never use test metrics
for adaptation or let GPU task completion order determine floating-point aggregation order.

Keep global state_dict weights every round. Keep client/cluster states only when the chosen
algorithm requires persistence or the user requests them. If local optimizers reset each
round, boundary resume may omit their discarded state, but still needs round/RNG/adaptive state.

## Approved specification (user decision, 2026-09-06)

The user approved Proposal A in `papers/khan-2025-afpha/rebuild.md` unchanged. It is the
executable specification for this project. Every quantity below completes a gap the paper
leaves open — report them as implementation choices, never as the authors' hyperparameters,
and never call the result an exact reproduction of AFPHA.

| item | value |
|---|---|
| participation | 100% of clients every round |
| clusters | fixed, 5 clients each -> 4 / 10 / 20 clusters; permutation from `default_rng(42)` |
| aggregation | sample-weighted within cluster, then sample-weighted across clusters |
| local objective | `CE + (mu_i/2)·\|\|w - w_t\|\|²` over learnable parameters only |
| mu, round 1 | 0.01 for every client |
| mu, round t+1 | `0.01·(1 + d_i/(d_i + d̄ + 1e-12))`, `d_i = \|\|w_i,t+1 - w_t+1\|\|₂`, `d̄ = Σ (n_i/N)·d_i` |
| optimizer | Adam, betas (0.9, 0.999), eps 1e-8, weight_decay 0; reset per client per round |
| LR | `1e-5 + (1e-3 - 1e-5)/2 · (1 + cos(π(t-1)/49))`, fixed within a round |
| gradient clipping | global norm 1.0, applied after unscale and after the proximal gradient |
| BatchNorm | `running_mean`/`running_var` averaged with the same `n_i/N` weights; `num_batches_tracked` = max |
| class balancing | none — no class weights, no resampling |

Two consequences to state in any write-up. One sample-weighted cluster average followed by one
sample-weighted server average is **algebraically identical to sample-weighted FedAvg** over
all participating clients; the hierarchy is a system description, not an optimization
difference. And averaging `running_var` is a buffer-merge convention, not a pooled variance.

Monitoring is required: stream progress to W&B (see `wandb.md`) alongside the committed JSONL
heartbeat. W&B holds scalars only — every reported number still needs a pulled artifact.

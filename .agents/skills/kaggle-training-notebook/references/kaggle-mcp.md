# Kaggle MCP in Codex: configure, verify, operate

## Codex connection contract

**For this project, first read [saved credentials](kaggle-credentials.md).**
The user stores tokens in ~/.kaggle/accounts and Codex now uses http_headers_helper.
That current workflow supersedes the environment-export/restart examples below.
Check saved tokens and active account before asking for new credentials.
The following environment-backed setup is an alternative, not the current configuration.

Codex CLI and the IDE extension share MCP configuration in `config.toml`. Use the project-scoped `.codex/config.toml` only for a trusted project; otherwise use the user's `~/.codex/config.toml`. Claude/VS Code files such as `.mcp.json` and `.vscode/mcp.json` are migration inputs, not Codex configuration.

An alternative environment-backed configuration uses the official Streamable HTTP endpoint:

```toml
[mcp_servers.kaggle]
url = "https://www.kaggle.com/mcp"
env_http_headers = { Authorization = "KAGGLE_MCP_AUTHORIZATION" }
enabled = true
required = false
startup_timeout_sec = 30
tool_timeout_sec = 600
default_tools_approval_mode = "writes"

[mcp_servers.kaggle.tools.get_dataset_info]
approval_mode = "approve"
```

Kaggle's current MCP schemas do not consistently mark read operations as read-only. Add explicit
`approval_mode = "approve"` overrides only for the metadata/status/list operations actually needed.
Leave `save_notebook`, `create_notebook_session`, `cancel_notebook_session`, uploads, submissions,
and large downloads on the server default so Codex still requests approval.

`agents/openai.yaml` declares the skill's MCP dependency, but neither that declaration nor editing `config.toml` proves tools are callable in an already running session. Only the alternative environment-backed setup needs `KAGGLE_MCP_AUTHORIZATION` inherited by the host. The current file-backed helper does not.

Never print, paste into a command argument, or commit the authorization value. If a legacy `.mcp.json` already contains it, the user can launch the IDE from a terminal without displaying it:

```bash
export KAGGLE_MCP_AUTHORIZATION="$(jq -r '.mcpServers.kaggle.headers.Authorization' .mcp.json)"
code .
```

This is a manual launch step because an agent editing the repository cannot change the environment of an already running IDE process.

## Gate

1. Inspect the current tool inventory for tools from the `kaggle` MCP server. Do not assume a fixed callable prefix; the host may expose MCP tools eagerly or through tool discovery.
2. If absent, inspect `.codex/config.toml` without printing secret values, run `codex mcp list`, and confirm the project is trusted. A listed server proves configuration discovery only.
3. Check the effective credential mechanism. The current header helper needs no export;
   reload the relevant MCP client. Only an environment-backed header requires an inherited
   environment variable.
4. In the refreshed client, call an account-scoped read such as `get_accelerator_quota`.
   Public `get_dataset_info` and file listings can succeed without authentication: on
   2026-09-06 all four requested datasets returned metadata while quota returned
   `Unauthenticated`. Those public reads prove transport only, not account access.

For diagnostics outside the extension, a direct JSON-RPC `initialize`, `tools/list`, and one read-only `tools/call` against the endpoint is valid evidence that the endpoint and credential work. Report that separately from extension tool availability. Do not hand-roll the protocol for routine work once the native MCP tools are loaded.

When MCP authorization is unavailable but Kaggle CLI OAuth is active, read-only status and quota
checks remain available without handling a token directly:

```bash
conda run -n nckh kaggle kernels status <owner>/<kernel-slug>
conda run -n nckh kaggle quota
```

This proves the remote state through CLI OAuth, not an in-client MCP connection; keep that
distinction explicit.

| Observation | Meaning | Action |
|---|---|---|
| Authenticated payload from a Kaggle tool | Connected in this client | Proceed |
| Direct protocol probe succeeds, tools absent | Endpoint works; current extension has stale tool inventory | Restart/new session, then recheck |
| Server listed but environment header missing | Config loaded without credential | Export the variable in the IDE parent process |
| 401/403 | Authentication or permission failure; cause not yet established | Check effective credential source and native host result before requesting replacement |
| Timeout/transport closed | Server unreachable or client transport failed | Retry once, then report the exact failure |

If the gate still fails, use the smallest action supported by the observed failure.
For the current saved-token workflow:

```text
MANUAL ACTION REQUIRED
1. Check the native host call as well as any direct probe. A probe-only failure is
   recorded in this project and does not by itself require replacing a token.
2. If host configuration is stale, sync static configs and reconnect the relevant client.
3. Request token replacement only if evidence supports a credential problem. The user
   writes it to ~/.kaggle/accounts/<username>.mcp-token, never into chat; then sync/retest.
```

Do not loop, reveal credentials, or claim that a local config edit changed the current tool inventory.

**Mid-run drop.** Report the last confirmed round and treat newer state as unknown. Do not restart from round 0. Reconnect, inspect the remote state, and pull committed outputs before deciding how to resume.

## Push

Prefer the Kaggle CLI in the project conda environment for notebook pushes:

```bash
conda run -n nckh kaggle kernels push -p papers/<slug>/notebook/
```

This reads the notebook and `kernel-metadata.json` from disk byte-for-byte. Use MCP `save_notebook` only when the task specifically benefits from it and the full notebook text is available. Either path is an external write that starts a run; require authorization from the user's current request.

Metadata that must be set every time:

- `is_private: true` — explicit on every push; do not rely on account or prior-version defaults.
- `enable_gpu: true` **and** `machine_shape: "NvidiaTeslaT4"` — see *The accelerator field* below. Without it you get one P100 and the `world_size = 2` assumption breaks.
- `enable_internet: true` when the run requires package/weight downloads or online W&B monitoring.
- `dataset_sources` — input datasets; `kernel_sources` — previous notebook outputs for resume.
- `kernel_type: notebook`, and a stable slug so re-pushes version the same kernel instead of forking new ones.
- The slug is the **full** `USERNAME/KERNEL-SLUG`, not the bare kernel name. A bare name is
  rejected with `Invalid slug`, which costs a whole re-send of the notebook body — get it right
  the first time.

For W&B, the safe default is never to paste a local `.env` key into the notebook before pushing.
A private kernel still retains source in its remote version history. Enable Internet and retrieve
the attached Kaggle secret at runtime with `UserSecretsClient().get_secret("wandb_key")`; fail
before expensive work if it is unavailable.

There is an important API boundary: an attachment selected under **Add-ons > Secrets** belongs to
an editor-launched notebook version and is not expressible in `kernel-metadata.json`. Therefore a
CLI/API `kernels push` batch run cannot inherit that selection. Once the intended private source is
uploaded, the owner must open that notebook, enable the secret, and choose **Save Version > Save &
Run All**. A second CLI push only repeats the missing-secret failure.

If the repository records the owner's explicit choice to trade that version-history risk for fully
automated private runs, use `scripts/embed_wandb_key.py NOTEBOOK --metadata METADATA`. It reads the
credential from `~/.netrc`, refuses non-private metadata, updates the notebook without printing the
value, and leaves the credential embedded as requested. Run it as a separate command before the
security/preflight checks; never interpolate the key into a shell command. If privacy is later
disabled or collaborators are added, stop and require W&B key rotation before proceeding.

### The accelerator field — measured, not guessed

**Set `machine_shape` to the exact string `"NvidiaTeslaT4"`.** That one value yields
**2× Tesla T4, sm_75** — Kaggle's T4 offering *is* the two-GPU machine, so the name carries
no `x2` suffix. Verified by a probe notebook printing `torch.cuda.device_count()` → `2`.

It is the same field under three names, so any of the three routes works:

| Route | Field |
|---|---|
| Kaggle MCP `save_notebook` | `machineShape: "NvidiaTeslaT4"` (+ `hasMachineShape: true`) |
| `kernel-metadata.json` | `"machine_shape": "NvidiaTeslaT4"` |
| CLI flag | `kaggle kernels push --accelerator NvidiaTeslaT4` |

The CLI proves they are one field:

```python
# kaggle/api/kaggle_api_extended.py
# The allowed names are in an enum that is not currently included in kagglesdk.
request.machine_shape = acc if acc else self.get_or_default(meta_data, "machine_shape", None)
```

The enum lives only on the server, so there is nothing local to check a value against.

**Two traps that make a wrong value expensive:**

1. **An invalid value is silently coerced** to the default single **P100**. No error, no warning.
   Measured rejects: `GpuT4x2`, `GpuT4X2`, `T4x2`, `GpuTeslaT4x2`, `nvidia-t4-x2`, and pure
   garbage — every one came back as one P100. Worse, Kaggle's PyTorch build supports
   sm_70–sm_120 only, so a coerced P100 (sm_60) fails with a baffling *"Tesla P100 with CUDA
   capability sm_60 is not compatible with the current PyTorch installation"* rather than
   anything mentioning accelerators. Read that message as **"your `machine_shape` was invalid"**.
2. **`get_notebook_info` cannot confirm it.** Its `machine_shape` reply is the coarse category
   (`"Gpu"`) and reads `"Gpu"` for a valid T4 request and an invalid one alike. The push
   response carries no accelerator field either.

So confirmation comes from **the run itself, before any round trains** — the notebook's first
cell prints `torch.cuda.device_count()` and asserts `== 2` and `sm_major == 7`. That assert
firing within the first two minutes is the cheap failure you want; letting a run proceed on an
unconfirmed accelerator is not, because the quota it burns is not refundable.

To probe a value without spending a round, push a one-cell notebook that prints
`torch.cuda.device_count()` and the device names. Roughly 20 s of GPU quota per probe, and
**at most 2 batch GPU sessions run at once** — a third push is rejected outright with
`Maximum batch GPU session count of 2 reached.`, so probe in pairs.

### Re-pushing costs the whole notebook, every time

`save_notebook` with a slug but no `text` is rejected outright — there is no "re-run what is
already there" call. Every push re-sends the entire notebook body inline, which for a real
training notebook is ~85 KB. On a task facing many push cycles (crash fixes, plus one push per
session for a run longer than the 12 h cap), install the CLI instead and push from the shell,
where the body is read from disk: byte-exact and free.

```bash
conda run -n nckh kaggle kernels push -p papers/<slug>/notebook/
```

CLI auth is `conda run -n nckh kaggle auth login` (OAuth, no token to mint or paste). Keep using the MCP server for
status, logs and output — only the push is worth moving.

## Babysit

Poll status → `running` → `complete` / `error`. Between polls, pull the log/output tool and read it rather than waiting blind.

On the first sign of failure, stop and diagnose from the traceback. Common Kaggle 2×T4 failures:

| Symptom | Cause | Fix |
|---|---|---|
| NCCL hang at `init_process_group`, or `unhandled system error` | PCIe P2P blocked in the container | `os.environ["NCCL_P2P_DISABLE"]="1"`, `NCCL_IB_DISABLE="1"` before spawn |
| `Address already in use` on re-run | stale `MASTER_PORT` from the killed run | pick a free port at runtime; never hard-code 29500 |
| `CUDA out of memory` on one rank | per-GPU batch too large, or eval running on all ranks | lower `BATCH_PER_GPU`, raise `GRAD_ACCUM`, keep eval on rank 0 |
| `ProcessRaisedException` with a child traceback | the real error is *inside* the traceback | read the child traceback, not the spawn wrapper |
| Loss becomes NaN under AMP | fp16 overflow | check `GradScaler` is stepping, clip grads, lower LR |
| Session ends at 12 h mid-training | Kaggle hard cap | expected — resume from `last.pt` on the next push |
| `DataLoader worker killed` | 30 GB RAM exceeded by N ranks × N workers | drop workers, switch to `mmap_mode='r'` |

After each fix, re-push. The resume contract means the run continues from the last completed round — say so explicitly in the report so the user knows nothing was retrained.

## Cross-session resume (the part people get wrong)

`/kaggle/working` is **wiped when a new session starts**. Checkpoints survive only as the *committed output* of the finished run.

To resume a run that hit the 12 h cap or died:

1. The dead run's output is available as a data source under the notebook's slug.
2. Attach prior notebook outputs via `kernel_sources`; use `dataset_sources` only for datasets.
3. It mounts read-only somewhere under `/kaggle/input/`; the exact nesting depends on source and attach path, so do not assume `/kaggle/input/<slug>/`. Measured 2026-09-04, the mount is nested by kind and owner — `/kaggle/input/notebooks/<owner>/<slug>/runs/<run_name>/checkpoints/last.pt`, and `/kaggle/input/datasets/<owner>/<slug>/...` — three levels before the run's own tree begins, which is why a glob with any fixed number of `*` segments finds nothing.
4. Implement recursive discovery of a unique compatible resume bundle and import it into the new working output. Match the current driver's artifact layout; `last.pt` is a historical example. Multiple matches are a stop condition, not a reason to pick the first. AFPHA currently lacks this import path (CONTEXT.md §12); attaching outputs alone is insufficient.

Attaching that source is usually a **MANUAL ACTION** if the MCP server cannot add notebook-output sources programmatically. Ask for it explicitly rather than letting the run silently restart at round 0.

**And enforce that in the notebook, not only in the ask.** A push whose only purpose is to continue another run should set `require_resume` and die in cell 4 when no checkpoint is found. Measured here: a resume push whose glob missed the checkpoint trained from round 0 through its whole 1.9 h budget, redoing work that already existed up to round 45. The glob was one bug; treating a missing checkpoint as a warning rather than an error was the expensive one.

An alternative worth proposing for long multi-day runs: after each round, rank 0 pushes `checkpoints/` to a private Kaggle **Dataset** via the API and the next session attaches that dataset. Costs one upload per round; buys resume that does not depend on the previous kernel's output being intact.

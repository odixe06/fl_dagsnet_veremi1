---
name: kaggle-training-notebook
description: Build, validate, resume, and operate PyTorch training notebooks on Kaggle; retrieve artifacts and report all 10 classification metrics. Use for paper reproduction or Kaggle training/evaluation, not ordinary local data exploration.
---

# Kaggle training notebooks

Read the maintained [shared skill](../../../.agents/skills/kaggle-training-notebook/SKILL.md).
Resolve its relative paths from its own directory. Read `CONTEXT.md` for current decisions and
outstanding review findings; historical examples are not current run status or authorization.

Use this host's callable tools and MCP configuration: Claude uses `.mcp.json`, while Codex
uses `.codex/config.toml`. The Claude `SessionStart` hook does not run in Codex.

For a new FL method on VeReMi/DAGSNet, read
[per-client-fl-veremi.md](references/per-client-fl-veremi.md) first: it is the procedure
distilled from `fd_ids`, `tinyproto` and `pfedes`, so those projects need not be reopened.

For account work, read [multi-account.md](references/multi-account.md) and
[saved credentials](references/kaggle-credentials.md). Never display credentials.

The account policy the user set on **2026-09-10**, replacing the 2026-09-08 and 2026-09-09
versions: **the agent switches accounts on its own, without asking.** `ensure`/`use` may be run
with `--confirm` directly. The user's only action is to **reconnect the MCP client when a
reconnect is actually needed** (Claude: `/mcp`) — the CLI switches immediately, but a live MCP
client keeps its old bearer and will silently answer as the previous account. Say which account
you switched to and why; a silent switch is still a defect, because the user must be able to
tell which identity produced which artifact. Two things are still outside this permission:
`kaggle auth login --force`, which destroys an unrecoverable refresh token, and any account the
user has restricted for a specific piece of work. A dead refresh token needing a browser login
is the one step only the user can perform; `SessionStart` reports it.

W&B monitoring is required for this project. A Kaggle API/CLI push cannot attach an
editor-selected secret, so the owner gave a **standing authorization** on 2026-09-07 covering
all of their own projects: the generator injects the key from `~/.netrc` into `is_private: true`
notebooks and still prefers an attached secret at runtime. The key is then permanent in Kaggle's
version history and the generated `.ipynb` files must not be committed or shared — say so each
time you use it. To take a key back, follow the ordered procedure in
[wandb.md](references/wandb.md) §1.1: revoke at W&B first, because a pushed Kaggle version
cannot be scrubbed and rebuilding locally removes nothing from Kaggle.

Before building or pushing any notebook, read
[library-runtime.md](references/library-runtime.md): the worker's torch/CUDA/Python stack is not
the one you developed against, the accelerator must be declared in the notebook document as well
as in `kernel-metadata.json` (notebooks missing that block came up with zero GPUs), `docker_image`
must be pinned by digest, and torch must never be reinstalled on a Kaggle GPU image. It also
carries the vmap/autocast, Inductor RAM, `np.save` and `weights_only` failures this project hit.

Before trusting a verifier or a recovery path, read
[verifying-artifacts.md](references/verifying-artifacts.md): random-input checks test the formula
and not the run, completion markers must be written last, and a configuration fingerprint is not
a run identity.

The `.agents/skills` tree is maintained; this directory retains mirrored references and helpers
for existing Claude callers. Sync edited copies instead of maintaining a second workflow here.

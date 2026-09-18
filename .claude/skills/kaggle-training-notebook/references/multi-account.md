# Kaggle multi-account operation

Use this reference when checking several saved accounts, selecting an authorized account for
launch, or handing a run to another owner. Read the current account policy in `CONTEXT.md` and
later user messages first. A saved credential is not by itself authorization to launch as its owner.

## Inspect without switching

From the repository root:

```bash
KAGGLE_ACCOUNT_HELPER=.agents/skills/kaggle-training-notebook/scripts/kaggle_account.py
conda run -n nckh python "$KAGGLE_ACCOUNT_HELPER" list
conda run -n nckh python "$KAGGLE_ACCOUNT_HELPER" quota
conda run -n nckh python "$KAGGLE_ACCOUNT_HELPER" health
conda run -n nckh python "$KAGGLE_ACCOUNT_HELPER" plan --gpu-hours 36.2 --session-hours 11
```

`list` discovers snapshots and MCP-token files in `~/.kaggle/accounts/`. Do not hardcode the
number of accounts or treat an old inventory as current. `quota` reads each OAuth snapshot via
an explicit SDK file path, generates an access token in memory, introspects its username, and
queries that account. It does not overwrite the live login or saved snapshots. An account with
only an MCP token can appear in `list`, but cannot supply OAuth quota or be selected by `ensure`.
Failures are reported by account; a missing response means unknown quota, not zero usage.

`health` asks only whether each snapshot's refresh token still mints an access token and whether
the server agrees with the filename. It is the cheap early warning that a rotation target is dead,
and it runs from the Claude `SessionStart` hook. A single failure is not proof of revocation:
a network or service error reads the same way, so re-run before asking for a browser login.

`plan --gpu-hours N` answers "which accounts, in how many sessions" for a training plan, and
changes nothing. It ranks the active account first, then by free hours, allocates greedily, and
splits each allocation by `--session-hours` (at most the service's 12 h). `--exclude USER`
keeps an account out of the plan — use it for an account with work you must not disturb.
It exits 2 on a shortfall and prints the earliest quota refresh. The allocation is arithmetic
over a dated quota read, not a reservation.

Budget from the live response: `max(0, total_time_allowed - time_used - time_reserved)`.
Quota, refresh time and concurrent-session limits belong to each account; do not promise a
pooled budget or assume all accounts have 30 hours. For several planned launches, reserve their
combined budgets locally as well: quota checks do not atomically reserve hours and two callers
can observe the same free balance. Serialize account switches and launch preparation.

Each run's estimate includes preparation, compilation, rounds, evaluation, and artifact writes.
Account quota and a session's duration limit are separate constraints. Validate them against the
current service before launch; extrapolation from a different GPU is not a measured budget.

## Credentials and host identity

| Consumer | Credential/configuration in this project |
|---|---|
| Kaggle CLI | `~/.kaggle/credentials.json`, one active OAuth login |
| Saved CLI accounts | `~/.kaggle/accounts/<user>.credentials.json` |
| Saved MCP bearers | `~/.kaggle/accounts/<user>.mcp-token` |
| Codex | `.codex/config.toml` → repository `scripts/kaggle_mcp_headers.py`, using `http_headers_helper` |
| Claude Code | `.mcp.json`, local static header |
| VS Code MCP | `.vscode/mcp.json`, local static header |

The MCP bearer and CLI OAuth are separate credentials. This helper uses OAuth; do not put a
`KGAT_` bearer or its suffix into an OAuth snapshot. Earlier tests rejecting legacy `kaggle.json`
do not establish that every CLI version supports only OAuth.

The Codex helper reads the active username and corresponding token from disk. It needs no
`KAGGLE_MCP_AUTHORIZATION` export. Never run it with visible stdout: its JSON output is secret.
See [saved credentials](kaggle-credentials.md) for synchronization and reload commands.

`list` compares the saved `.mcp.json` header to saved tokens; it cannot identify the credential
already used by a live host connection. After switching, reconnect that host and verify an
account-scoped native MCP read. A quota response proves authentication, but not a unique username.
`has_ever_run` flags and similar quota values are not reliable account identifiers. Record the
configured account and server-confirmed identity separately; use snapshot OAuth introspection
for an explicit username. Until alignment is established, use the selected CLI for account-scoped
operations and do not use stale MCP quota to budget a launch.

## Select an account within the user's authorization

`use <user>` is an explicit switch. `ensure --gpu-hours N [--tpu-hours N]` keeps the active
account if sufficient, otherwise selects a saved account with enough free hours and both
credentials. These commands mutate local login/configuration; `quota`, `health` and `plan` are
the inspection commands.

**Both take `--confirm`, and the agent may pass it itself.** Without `--confirm` they print the
exact change and exit 3, which stays useful for dry runs. The account policy the user set on
**2026-09-10** — replacing 2026-09-08 and 2026-09-09 — is: **the agent switches accounts on
its own, without asking.** No approval round-trip. The user's only account action is to
**reconnect the MCP client when a reconnect is actually needed**; everything else is the
agent's to do.

What "appropriate and necessary" rules out, and what still holds:

- **Announce every switch** in the reply: which account, why, and switch back when done if the
  session had a prior active account. A silent switch is still a defect — the user must be able
  to tell which identity produced which artifact.
- **`kaggle auth login --force` is NOT covered.** It overwrites the refresh token and the old
  one is unrecoverable. That still needs an explicit human decision, and `save` must snapshot
  the active account first. `use`/`ensure` only swap saved files and are safe to repeat.
- **Run `health` before a rotation depends on a snapshot.** A dead refresh token discovered
  mid-handoff costs more than the check.
- **Identity comes from the server**, never from a config file: `verify_kaggle_identity.py
  <expected_user>` after the switch. `list` reports what is *saved*, not who the API answers as.
- **Reconnect the host MCP client after a switch** (Claude: `/mcp`). The CLI changes at once;
  a live MCP client keeps its old bearer until reconnected, so MCP reads can silently answer as
  the previous account.
- If the user has authorized only one account for a piece of work, that restriction still wins
  over this general permission.

```bash
# Propose (changes nothing, exit 3 when a switch is needed):
conda run -n nckh python "$KAGGLE_ACCOUNT_HELPER" ensure --gpu-hours 6
# Perform the switch (the agent runs these directly under the 2026-09-10 policy):
conda run -n nckh python "$KAGGLE_ACCOUNT_HELPER" ensure --gpu-hours 6 --confirm
conda run -n nckh python "$KAGGLE_ACCOUNT_HELPER" use USERNAME --confirm
```

`ensure` exits 0 when the selected account is ready, 1 when quota/authentication/synchronization
fails, 2 when no switchable account meets the budget, and 3 when a switch is needed but not yet
confirmed. A partial quota survey is not proof that inaccessible accounts have no hours.
Never launch after a nonzero result.

Concurrent switches are refused rather than interleaved: both commands hold an exclusive lock on
`~/.kaggle/accounts/.switch.lock` while they write. A second caller exits with a message instead
of leaving the CLI on one account and another account's bearer in the host configs.

`use` requires an MCP token and a correctly labelled OAuth snapshot. It refuses to discard an
active account with no snapshot, refreshes the selected CLI login, rewrites `.mcp.json`, and
invokes the repository `scripts/sync_kaggle_mcp.py` when present to update all host configs.
A refresh or sync failure returns nonzero; CLI or some config files may already have changed.
Inspect `list`, repair the failed stage, and synchronize before proceeding. There is no automatic
rollback and no guarantee that a running MCP client has reloaded. Do not run two switches at once.

Before launch, verify the selected account can read every input and update the notebook
`id` to the authorized `owner/slug`. Membership in a group is not proof of per-item access.
Public inputs need no owner switch. For private prior outputs, inspect access as the account
that will run; arranging sharing or uploading another private copy requires the applicable
user authorization. Account selection does not itself authorize starting a training run.

## Save or refresh an account

Preserve the active login before browser authorization overwrites it. `add-account` performs that
ordering — snapshot the current login, hand the terminal to the browser flow, verify the account
that actually logged in, snapshot it — and refuses to run without a real terminal, because the
verification step is a human action an agent tool call cannot complete:

```bash
conda run -n nckh python "$KAGGLE_ACCOUNT_HELPER" add-account NEW_USERNAME
```

The equivalent by hand, when you want the steps separated:

```bash
conda run -n nckh python "$KAGGLE_ACCOUNT_HELPER" save CURRENT_USERNAME
conda run -n nckh kaggle auth login --force
conda run -n nckh python "$KAGGLE_ACCOUNT_HELPER" save NEW_USERNAME
```

The browser step is manual and must be performed as the intended owner. Use the CLI's interactive
terminal flow; do not put verification codes into repository files, logs, or command arguments.
Store the MCP token directly in the account file, mode 0600, without sending it in chat.
The accounts directory is mode 0700. `save` refuses a username that differs from the live login.
`add-account` leaves the new account ACTIVE; returning to the previous one is a confirmed switch.

An MCP bearer alone does not make an account usable. `list` shows it, but `quota`, `health`,
`plan`, `ensure` and `use` all need the OAuth snapshot, which only a browser login creates.

For an existing login with an expired access token:

```bash
conda run -n nckh python "$KAGGLE_ACCOUNT_HELPER" refresh
```

`refresh` updates the live OAuth credentials and an existing matching snapshot; it does not
change account or MCP bearer. A refresh failure may be network or authentication related;
do not diagnose token revocation solely from a generic permission error. Request another
browser login only when the evidence requires it. Exact SDK expiry behavior is version-specific.

What rotation depends on is the **refresh** token in each snapshot, not the access token beside
it. An access token expires within hours and every command that reads another account mints a new
one in memory; an expired one in a snapshot is normal and harmless. A refresh token ends only by
revocation, account deletion, or another `--force` login overwriting it — which is why `save`
comes first, and why a switch immediately refreshes and re-snapshots the account it moved to.
Idle accounts are the risk: nothing exercises their refresh token until the day a rotation needs
it, so run `health` (the session hook does) rather than discovering it at launch time.

`.claude/settings.json` has a `SessionStart` refresh hook. That is a Claude host hook, not a
Codex skill feature, and Codex does not automatically execute it. In Codex, call `refresh` when
CLI authentication needs it; do not assume a window restart refreshed OAuth.

## Runbook: the active account runs out of quota

Under the 2026-09-10 policy the agent does all of this without asking and without stopping.
Steps 4-5 are a report and an action, not a gate:

1. `plan --gpu-hours <total the training plan needs> --session-hours <cap>`. The total is the
   whole plan, not one session: a 36 h scenario is four accounts-worth of decisions, not four
   separate surprises. Add `--exclude` for any account whose work must not be disturbed.
2. `health`, if the session hook has not just run it. An account the plan depends on whose
   refresh token is dead needs a browser login, and that is the user's action — surface it now,
   not after they approve a switch.
3. `ensure --gpu-hours <this session's need>` without `--confirm`. Exit 0 means the active
   account pays and nothing needs saying. Exit 3 carries the proposal.
4. Re-run the identical `ensure` with `--confirm`, then verify with
   `scripts/verify_kaggle_identity.py <user>` before pushing anything.
5. Say in the reply which account you switched to and why, and what the switch touched (CLI
   login, `.mcp.json`, and the other host configs through `sync_kaggle_mcp.py`). Announcing is
   not asking, but it is not optional either: without it the user cannot tell which identity
   produced which artifact.
6. Reconnect the MCP client of the host you are using, and record which owner paid for which
   rounds in the run metadata and the report.

Exit 2 at step 3 means no saved account can pay, and that is where the agent does stop:
report the shortfall, the earliest refresh time from `plan`, and which accounts need a browser
login. Do not launch a smaller run to "use up" the remainder unless the user asks. The
switching permission covers choosing an identity, not spending quota the plan cannot cover.

An MCP client that is already connected keeps the old bearer until the host reconnects, so a
rotation is complete for the CLI immediately and for MCP only after that reconnect. Codex reads
its bearer through the header helper at connection time, so it needs no config edit — only the
reconnect. Prefer the CLI for anything account-scoped while a reconnect is outstanding.

## Cross-account resume

**TinyProto status, 2026-09-08:** `scripts/gen_notebooks.py` writes `kernel_sources` as
`f"{a.owner}/{sweep_slug}"`, so `--owner` moves the notebook *and* every attachment to the new
account at once. Splitting one scenario across accounts therefore needs the previous owner named
separately — the generator has no flag for that yet. Until it does, a multi-session plan whose
boundaries fall inside one account is the supported shape; a boundary between accounts is an
unimplemented change, not a configuration choice. The datasets are `odixe0502/veremi-fl-*`, so
every other account also needs read access to them; the shared group is the mechanism, and
per-item access still has to be verified as the account that will run.

**AFPHA status, 2026-09-06:** the current driver resumes only in the same output directory.
Attaching another notebook's output alone does not import checkpoints. See `CONTEXT.md` §12;
implement and test new-session resume before attempting a handoff. Historical successes with
`last.pt`, round 0–49, or 4.7 MB checkpoints belonged to a different implementation.

Once the actual driver's resume contract is verified:

1. Record the source owner, notebook version, last committed round, effective config, and
   artifact hashes. Confirm that the target account can read the full required resume bundle.
2. Attach notebook outputs using `kernel_sources` (checkpoint datasets use `dataset_sources`).
   Resolve a unique source recursively under `/kaggle/input`; support nested owner prefixes.
3. Require resume explicitly. Missing, ambiguous, incompatible or incomplete artifacts must
   fail before expensive preparation. Inference weights alone may omit mu, RNG or history.
4. Restore the last committed round and prove the next round is its successor. Compare all
   overlapping history and committed artifact hashes; matching a single F1 value is insufficient.
5. Preserve the authorized W&B run identity only for a verified continuation with monotonic
   steps. W&B `resume="allow"` does not load model weights. A deliberate new run or unverified
   restart needs its own identity and provenance; do not silently overwrite the original history.
6. Record which owner/session paid for which rounds and budget each continuation separately.

Monitor the exact session/version and fresh W&B step progress. Account-wide `time_reserved`
can corroborate accelerator activity but cannot tell which of several runs is alive or stopped.

## Handing a resumed run to another account

**`kernel_sources` does not cross accounts.** Measured 2026-09-09 with a dedicated CPU probe
(`minhtrit06` attaching a private notebook output owned by `khanhmay0304`):

```
The following are not valid kernel sources and could not be added to the kernel: [...]
Kernel version 1 successfully pushed.
```

Kaggle prints one warning line, **reports the push as successful**, and starts the kernel with an
**empty `/kaggle/input`**. There is no exception, no non-zero exit, nothing an exit-code check
would see. A multi-session chain that crosses an account boundary this way looks correct right up
until the session starts training from round 1 — or, with a resume guard, dies after the queue
wait for no visible reason.

Two consequences for planning:

* Decide the account boundary from *where the payload is smallest*, not from where the quota runs
  out. Moving the boundary one session earlier can cut the handoff from 43 rounds to 16.
* Verify the boundary with a **CPU probe before** the chain reaches it. GPU quota is the expensive
  resource; a probe that mounts the source and reads a few bytes costs none of it and answers the
  question hours or days before the chain gets there.

### The route that does work: a checkpoint dataset

Upload the previous session's run tree as a dataset **owned by the account that will run the next
session** — then nothing has to be shared or made public, and the run keeps its private weights
private. `find_import_source`-style discovery (`rglob("config.json")` filtered by run name and
fingerprint) does not care whether the tree arrived from a kernel output or a dataset mount.

Two defaults of `kaggle datasets create` will silently corrupt such a dataset:

| Default | What it does | Fix |
|---|---|---|
| `--dir-mode skip` | **ignores every directory**, uploading nothing but loose top-level files | `-r zip` |
| tabular conversion is ON | rewrites `*.csv` (and other tabular files) into Kaggle's own CSV | `-t` / `--keep-tabular` |

The second is the dangerous one: a rewritten `client_log/round_007.csv` still looks like a CSV, so
the upload and the mount both succeed, and the failure surfaces much later as an integrity-hash
mismatch mid-resume. Pass `-t` and then **compare sizes against the local originals** before
trusting the dataset.

Kaggle re-extracts the uploaded zip on its side and **drops one directory level**: a local
`runs/<run_name>/…` arrives as `<run_name>/…`. Harmless for rglob-based discovery; not harmless
for any code that pins the parent directory's name. The mechanism (seen 2026-09-14, pFedES 100c):
`-r zip` turns each top-level directory of `-p <dir>` into `<that-dir>.zip` and Kaggle unpacks the
zip's *contents* at the dataset root — so the top-level directory's own name is what disappears.
Uploading `<dir>/<run_name>/…` therefore lands `weights/`, `complete/` … at the root with **no
`<run_name>` anywhere in the mount path**, and a discovery filter such as pFedES's
`run_name in p.parts` finds nothing. Always stage one level deeper (`<dir>/runs/<run_name>/…`)
and check `kaggle datasets files` for the run name before pushing a GPU kernel. A wrong layout is
fixed with `kaggle datasets version -p <dir> -r zip -t -m "..."` on the same slug (new version
replaces the file set); a CPU probe of the gate is generated by the project's
`scripts/gen_ckpt_probe.py` (pFedES) — it embeds `proj/ckpt.py` and asserts the imported round.

**Measured end to end, 2026-09-11** (two run trees of 510 MB and 308 MB, `-r zip -t`): the
trees compressed about 5:1, the datasets were `ready` about a minute after upload, every file
matched the local tree by name *and* size (compare **all pages** of `datasets files`, not the
first; the CLI pages at 200), and the mount path was
`/kaggle/input/datasets/<owner>/<slug>/<run_name>/…` — note the extra `datasets/<owner>/`
level compared with the older `/kaggle/input/<slug>/`. Code that pins either layout breaks;
`rglob` for the per-round completion marker with the run name in the path survives both.

**Probe the gate itself, and assert the round, not just "found something".** A resume gate that
only checks `last is not None` has a hole: a dataset with one unreadable round in the middle
imports up to it, passes the gate, and the GPU session *retrains* the missing rounds under the
same run name — W&B drops the now non-monotonic steps silently, and only a later byte-level
merge notices. The CPU probe is where that is caught for free: run the production import
against the mounted dataset with the cfg from the run's own manifest, and **assert
`last == the round the previous session committed`** before pushing the GPU kernel. Keep the
probe as a generated notebook next to the training notebook's generator, embedding the same
checkpoint module the training notebook uses, so it cannot drift from the gate it tests. It
costs about half a minute per kernel and no GPU quota.

### The cheaper route: a handoff bundle (last round + sha chain)

The full-tree dataset above re-uploads every round the owner already holds locally, and the
next session's output then re-carries them, so every hop costs one upload and one download
of the whole history (NILM-FL 100c: 2.2 GB after 14 rounds, ~8 GB by round 50). Measured
2026-09-16: a full-tree upload ran at ~1.4 MB/s from the owner's machine.

What a continuation actually needs is one round: the weights it resumes from, that round's
resume/metrics/confusion/marker, and proof that the round is the product of the history it
claims. NILM-FL (`papers/nilm-li-2024/proj/ckpt.py`, `HANDOFF`) carries that proof as
`reports/handoff.json`: the sha256 of `weights/round_001..r.pt` taken from the full tree,
which the owner verifies locally (`last_complete_round` walks the `prev_sha` chain from
round 1) *before* exporting. The resume gate then anchors round r two ways — its bytes must
hash to `chain[r]` and its `prev_sha` must equal `chain[r-1]` — and refuses a bundle whose
chain, round or fingerprint does not match, or that would be spliced onto a tree with its own
rounds. Every session's output still holds a contiguous, verifiable tree; it just starts at r.

```
scripts/stage_ckpt_dataset.py <full run dir> --owner A --slug S --out DIR --last-only
kaggle_as.py A -- kaggle datasets create -p DIR -r zip -t        # ~35 MB (20c) / 170 MB (100c)
scripts/gen_ckpt_probe.py --owner A --dataset A/S --run-name R --expect-round r --out ...
gen_notebook.py --owner A ... --session N --require-resume --dataset-source A/S
```

Nothing is lost for the final record: the owner pulls every session and
`scripts/merge_sessions.py` unions the per-round files (byte-identical on every overlap, or it
refuses), rebuilds `history.csv`/`clients.csv` keyed by round, keeps each session's
`handoff.json` under `logs/sessions/<n>/`, and the merged directory reads as one run —
`verify_run --require-rounds 50` re-walks the whole chain there. Test:
`tests/test_ckpt_verify.py` cases 7–10 (bundle import, tampered chain, splice refusal, merge).

The same-account `--kernel-source` route keeps importing the whole attached tree; a project
that wants the smaller commit there too can stage a bundle for itself the same way.

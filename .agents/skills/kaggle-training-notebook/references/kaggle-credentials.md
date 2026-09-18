# Saved Kaggle credentials and client reload

The user designated `/home/odixe/.kaggle` as the credential source on 2026-09-06. Inspect
available files before requesting another token, environment export or browser login.

- `credentials.json`: active CLI OAuth login, including username.
- `accounts/<username>.credentials.json`: saved OAuth snapshots.
- `accounts/<username>.mcp-token`: saved MCP bearer, raw `KGAT_` or `Bearer`-prefixed.
- `accounts/.switch.lock`: held while a switch writes; not a credential.

The two files are independent. An account with only `.mcp-token` is half-installed: `list` shows
it, but it has no OAuth and cannot be inspected, planned with, or switched to until a browser
login creates its snapshot. `health` names those accounts every session.

Use the skill's `scripts/kaggle_account.py list` for current account inventory. Read
[multi-account.md](multi-account.md) for quota, authorized selection, refresh and handoff.
Do not select a dataset owner's account merely because its datasets are attached. Never print
credential files or token values; report only filenames, account names and verification results.

## Current host configuration

As of 2026-09-08 this repository has its own copies of `scripts/kaggle_mcp_headers.py`,
`scripts/sync_kaggle_mcp.py` and `scripts/probe_kaggle_mcp.py`, its own `.mcp.json`, and a
`.codex/config.toml` helper path pointing inside this repository. Before that date the Codex
helper path pointed at the AFPHA checkout and there was no `.mcp.json` here at all, which made
`use` fail rather than switch. `.mcp.json` and `.vscode/mcp.json` carry a bearer, are mode 0600
and are in `.gitignore`.

Codex uses `http_headers_helper` in `.codex/config.toml` to invoke the repository-root
`scripts/kaggle_mcp_headers.py` with the `nckh` interpreter. It reads the active CLI username
and matching saved MCP token. Its stdout is secret JSON consumed by the MCP client; do not run
it visibly. This configuration does not need an IDE-parent environment export.

From the repository root, when credentials/configuration need synchronization:

```bash
conda run -n nckh python scripts/sync_kaggle_mcp.py
```

Sync updates `.codex/config.toml` (helper, no bearer), `.mcp.json` and `.vscode/mcp.json`
(local static headers, mode 0600). It does not switch accounts. The account helper's `use`
command invokes this sync script when present; do not repeat sync after a successful switch
unless new evidence requires it. Never display or commit the two secret-bearing JSON files.

## A `KGAT_` token is also a CLI API token

Measured 2026-09-11 with Kaggle CLI 2.2.4 / kagglesdk: the same `KGAT_` value the MCP server
takes as a bearer authenticates the CLI and SDK when supplied as `KAGGLE_API_TOKEN`, and the
SDK accepts a **file path** in that variable and reads the token from the file — so the value
never has to appear on a command line. A token-only account (no OAuth snapshot, no browser
login yet) can therefore push kernels, create datasets and pull outputs immediately; what it
still cannot do is be managed by the OAuth-based helper (`use`, `ensure`, `health`, `quota`).

The reliable way to verify any saved bearer is the server's own `IntrospectToken`
(`api.security.oauth_client.introspect_token`): it returns `active` and the `username`, which
is a server-confirmed identity. The request must carry the token in its body as well as in the
auth header — build `IntrospectTokenRequest()` and set `request.token = <token>`; an empty
request returns HTTP 400 (measured 2026-09-17). Read the value from the `.mcp-token` file in
memory; never paste it on a command line:

```python
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.security.types.oauth_service import IntrospectTokenRequest
tok = (Path.home() / ".kaggle/accounts/<user>.mcp-token").read_text().strip()
api = KaggleApi.build_kaggle_client_with_params(args=[], api_token=tok)
req = IntrospectTokenRequest(); req.token = tok
with api:
    r = api.security.oauth_client.introspect_token(req)   # r.active, r.username
```

**Token-only account added 2026-09-17: `odixeuit`.** Its `.mcp-token` introspected active with
username `odixeuit` the same minute it was saved. Until the owner runs the browser login
(`kaggle_account.py add-account odixeuit`, see [multi-account.md](multi-account.md)), `list`
shows it as `CLI creds MISSING` and `health` names it every session; it can already push kernels,
create datasets and pull outputs through `kaggle_as.py odixeuit -- ...`. The owner stated its
GPU quota is guaranteed (fresh account, phone-verified); it is NOT yet part of any planned
100c hop chain — the owner chose to keep the s4→s8 plan in `CONTEXT.md` §11 unchanged. On 2026-09-11 all five saved bearers introspected as active,
including the four that `probe_kaggle_mcp.py` had reported `Unauthenticated` days earlier —
which is the measured form of the warning above: the direct probe cannot reject a token.

Running one command as another saved account without switching the live login — minting an
OAuth access token in memory (and introspecting it) or passing the token file path — is what
the skill's `scripts/kaggle_as.py <user> -- <command>` does; pulling one run as its owner
while pushing another as a different account then needs no switch and no MCP reconnect.
Setting `KAGGLE_API_TOKEN` globally, or creating `~/.kaggle/access_token`, would instead
override the OAuth login for every command: keep it per-process.

## Reload and verify

The Kaggle MCP server is remote; reconnect the client in the host supplying the tools.
Codex IDE supports restarting its extension after configuration changes. Restarting VS Code's
separate MCP client does not necessarily reload Codex. A full WSL/window restart is usually
unnecessary. Verify the smallest applicable reconnect by an account-scoped native read such as
`get_accelerator_quota`; public dataset metadata only proves transport/public access.

An authenticated quota response does not return a unique username. Record the configured
account separately from a server-confirmed identity. `has_ever_run` flags are not identity proof.
For explicit usernames, the multi-account helper introspects each OAuth snapshot in memory.

The following is an optional independent diagnostic, not the native client's status:

```bash
conda run -n nckh python scripts/probe_kaggle_mcp.py --source codex
```

Probe sources include `codex`, `claude`, `vscode`, `saved`, and `oauth`. `--account` chooses a
saved token for diagnosis without switching the CLI. The `oauth` probe does not refresh the
CLI token. The probe captures headers in-process and does not print secrets.

On 2026-09-06 direct probes returned `Unauthenticated`, while a later native host call after
restart returned authenticated quota. The probe/native discrepancy remains unexplained.
Re-measured 2026-09-08: a direct probe returned `Unauthenticated` for **all four** saved bearers,
including two whose accounts were verifiably working through OAuth the same minute. So the probe
cannot be used to accept or reject a newly added MCP token; only a native host call can, and only
account-scoped tools count.
A direct probe failure alone does not establish token expiry/revocation and must not override a
successful native call or trigger unnecessary token replacement. If the native call also fails,
inspect transport/configuration and report the actual error before requesting a manual action.

In this Codex session, native quota succeeded after the user's restart: GPU allowed 108000s,
used/reserved 0s, refresh 2026-09-12T00:00:00Z. These are dated observations, not future balances.
No new token is currently required. Do not publish credential fingerprints in handoff notes.

The `SessionStart` OAuth refresh hook in `.claude/settings.json` belongs to Claude. Codex does
not execute it as part of loading a skill. Refresh CLI OAuth when needed with the account helper.

Sources: [Codex MCP configuration](https://learn.chatgpt.com/docs/extend/mcp?surface=cli),
[header helper configuration](https://learn.chatgpt.com/docs/config-file/config-reference).

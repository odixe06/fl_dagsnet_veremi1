#!/usr/bin/env python3
"""Inspect saved Kaggle accounts and perform authorized CLI/MCP account switches.

CLI OAuth lives in ~/.kaggle/credentials.json; MCP has a separate saved bearer.
Secrets live in ~/.kaggle/accounts/ outside the repo and are never printed.
Quota reads do not switch accounts. `ensure` and `use` can change the active login;
use them only under the account policy authorized for the current project/session.

    kaggle_account.py list                 which accounts are saved, which is active
    kaggle_account.py quota                remaining GPU/TPU per account, without switching
    kaggle_account.py health               is each saved refresh token still usable?
    kaggle_account.py plan --gpu-hours N   split a budget into sessions across accounts
    kaggle_account.py ensure --gpu-hours N propose (or, with --confirm, perform) a switch
    kaggle_account.py save <username>      snapshot the CURRENT login under that name
    kaggle_account.py add-account <user>   snapshot current, browser-login, snapshot new
    kaggle_account.py use  <username>      make that account active (CLI + MCP)
    kaggle_account.py refresh              mint a fresh access token for the active account

`use` and `ensure` do not switch unless `--confirm` is passed; without it they print
exactly what would change and exit 3, which is the dry-run form. Under the account policy the
user set on 2026-09-09 (replacing the stricter 2026-09-08 one) the agent may pass `--confirm`
itself whenever the switch is appropriate and necessary, provided it announces the switch.
`kaggle auth login --force` is NOT covered: it destroys a refresh token and still needs a human.

`save` immediately after `kaggle auth login --force` is what captures a new account.
Snapshot the active account BEFORE logging in as another one: `--force` overwrites
credentials.json and the previous refresh token is gone. `add-account` does that
ordering for you; the browser step inside it is still performed by a human.

Refresh tokens, not access tokens, are what make a saved account reusable. An access
token expires within hours and is regenerated on demand; a refresh token dies only by
revocation, account deletion, or another `--force` login overwriting it. `health` is
the cheap way to learn a snapshot is dead before a rotation depends on it.

After `use`, reconnect the host MCP client and verify an authenticated read. Updating
the project host configurations does not establish which bearer a live client uses.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ACCOUNTS = Path.home() / ".kaggle" / "accounts"
LIVE = Path.home() / ".kaggle" / "credentials.json"
KAGGLE_SESSION_HOURS = 12.0     # service cap per GPU session, 2026-09-08
_lock_depth = 0


@contextlib.contextmanager
def switch_lock():
    """Serialize everything that writes credentials.json.

    Two concurrent switches interleave a copy, a refresh and a config rewrite, and the
    loser leaves the CLI on one account with another account's bearer in the configs.
    Re-entrant: `use` takes the lock and then calls `refresh`, which takes it again.
    """
    global _lock_depth
    if _lock_depth:
        _lock_depth += 1
        try:
            yield
        finally:
            _lock_depth -= 1
        return
    ACCOUNTS.mkdir(parents=True, exist_ok=True)
    os.chmod(ACCOUNTS, 0o700)
    handle = os.open(ACCOUNTS / ".switch.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit("another account operation is in progress; not switching")
        _lock_depth = 1
        try:
            yield
        finally:
            _lock_depth = 0
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)
# scripts/ -> kaggle-training-notebook/ -> skills/ -> .claude/ -> project root
DEFAULT_MCP = Path(__file__).resolve().parents[4] / ".mcp.json"


def creds_path(user: str) -> Path:
    return ACCOUNTS / f"{user}.credentials.json"


def token_path(user: str) -> Path:
    return ACCOUNTS / f"{user}.mcp-token"


def active_user() -> str | None:
    try:
        return json.loads(LIVE.read_text())["username"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def known_users() -> list[str]:
    if not ACCOUNTS.is_dir():
        return []
    return sorted({p.name.removesuffix(suffix)
                   for suffix in (".credentials.json", ".mcp-token")
                   for p in ACCOUNTS.glob(f"*{suffix}") if p.is_file()})


def cmd_list(args: argparse.Namespace) -> int:
    act = active_user()
    users = known_users()
    if not users:
        print(f"no accounts saved in {ACCOUNTS}")
        return 1
    print(f"{'account':<16} {'CLI creds':<10} {'MCP token':<10} active")
    for u in users:
        print(f"{u:<16} {'yes' if creds_path(u).is_file() else 'MISSING':<10} "
              f"{'yes' if token_path(u).is_file() else 'MISSING':<10} "
              f"{'<-- ACTIVE' if u == act else ''}")
    if act and act not in users:
        print(f"\nactive login is {act}, which has no snapshot — run: save {act}")
    if not args.mcp_config.is_file():
        print(f"\n{args.mcp_config} does not exist; nothing carries an MCP bearer for this host.")
        return 0
    mcp_user = mcp_account(args.mcp_config)
    print(f"\n.mcp.json ({args.mcp_config}) carries the token of: {mcp_user or 'unknown'}")
    print("This checks the saved file, not the identity of an already connected MCP client.")
    if act and mcp_user and act != mcp_user:
        print("WARNING: CLI and MCP point at different accounts. Quota read through MCP "
              "will not describe the account the CLI pushes with.")
    return 0


def mcp_account(mcp_config: Path) -> str | None:
    """Which saved account owns the bearer currently in .mcp.json (by comparison, no print)."""
    try:
        hdr = json.loads(mcp_config.read_text())["mcpServers"]["kaggle"]["headers"]["Authorization"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    tok = hdr.removeprefix("Bearer ").strip()
    for u in known_users():
        p = token_path(u)
        if p.is_file() and p.read_text().strip().removeprefix("Bearer ").strip() == tok:
            return u
    return None


def cmd_save(args: argparse.Namespace) -> int:
    act = active_user()
    if act is None:
        print("no active login to save; run: kaggle auth login", file=sys.stderr)
        return 1
    if act != args.username:
        print(f"the active login is '{act}', not '{args.username}'. Refusing to mislabel it.",
              file=sys.stderr)
        return 1
    ACCOUNTS.mkdir(parents=True, exist_ok=True)
    os.chmod(ACCOUNTS, 0o700)
    dst = creds_path(args.username)
    with switch_lock():
        shutil.copy2(LIVE, dst)
    os.chmod(dst, 0o600)
    print(f"saved CLI credentials for {args.username}")
    if not token_path(args.username).is_file():
        print(f"note: no MCP token stored for {args.username}. Write it to "
              f"{token_path(args.username)} (mode 600) to rotate MCP too.")
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    """Refresh the active OAuth login and its existing snapshot, without switching.

    Explicit refresh can recover an expired access token. Authentication failures
    do not alone distinguish network errors, permissions, or a revoked grant.
    """
    user = active_user()
    if user is None:
        print("no active login; run: kaggle auth login", file=sys.stderr)
        return 1
    with switch_lock():
        return _refresh_locked(user)


def _refresh_locked(user: str) -> int:
    try:
        from kagglesdk.kaggle_client import KaggleClient
        from kagglesdk.kaggle_creds import KaggleCredentials
    except ImportError:
        print("kagglesdk not importable — run inside the conda env that has the Kaggle CLI",
              file=sys.stderr)
        return 1

    client = KaggleClient()
    creds = KaggleCredentials.load(client)
    if creds is None:
        print(f"{LIVE} has no usable refresh token; `kaggle auth login --force` is needed.",
              file=sys.stderr)
        return 1
    try:
        with client:
            creds.refresh_access_token()          # writes credentials.json in place
    except Exception as exc:                      # a revoked refresh token surfaces here
        print(f"refresh failed for {user} ({type(exc).__name__}); credential details suppressed. "
              "Check connectivity and authentication; this alone does not prove revocation.",
              file=sys.stderr)
        return 1

    os.chmod(LIVE, 0o600)
    exp = json.loads(LIVE.read_text()).get("access_token_expiration", "")
    print(f"access token refreshed for {user}; valid until {exp or 'unknown'}")

    # Persist the refreshed credential so a later switch uses the current snapshot.
    if creds_path(user).is_file():
        shutil.copy2(LIVE, creds_path(user))
        os.chmod(creds_path(user), 0o600)
        print(f"snapshot {creds_path(user).name} updated")
    else:
        print(f"note: no snapshot for {user} yet — run: {Path(__file__).name} save {user}")
    return 0


def account_quota(user: str) -> dict:
    """Remaining GPU/TPU hours for one saved account, without disturbing the active login.

    `KAGGLE_CONFIG_DIR` does not help here: it only relocates the legacy `kaggle.json`, while
    OAuth credentials are hardcoded to `~/.kaggle/credentials.json` in kagglesdk. Reading
    another account's quota by pointing the CLI at a copied snapshot therefore reports the
    ACTIVE account's quota — a silent wrong answer, and the reason rotation decisions must not
    be made from `kaggle quota` output alone.

    What is safe: `generate_access_token()` mints a token from a snapshot's refresh token and
    writes nothing (only `refresh_access_token()` saves, and it saves to the default path).
    The minted token is then introspected, so the account the numbers describe is confirmed by
    the server rather than by the snapshot's filename.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kaggle_client import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetAcceleratorQuotaStatisticsRequest,
    )

    client = KaggleClient()
    with client:
        creds = KaggleCredentials.load(client, file_path=str(creds_path(user)))
        if creds is None:
            raise RuntimeError("no usable refresh token in the snapshot")
        creds._access_token = creds.generate_access_token().token
        confirmed = creds.introspect()
    if confirmed != user:
        raise RuntimeError(f"snapshot is labelled {user} but the server says {confirmed}")

    api = KaggleApi.build_kaggle_client_with_params(args=[], api_token=creds._access_token)
    with api:
        r = api.kernels.kernels_api_client.get_accelerator_quota_statistics(
            ApiGetAcceleratorQuotaStatisticsRequest())

    def hours(q) -> tuple[float, float, float]:
        if q is None:
            return (0.0, 0.0, 0.0)
        total = q.total_time_allowed.total_seconds() / 3600
        used = q.time_used.total_seconds() / 3600
        reserved = q.time_reserved.total_seconds() / 3600
        return (max(0.0, total - used - reserved), total, reserved)

    gpu_left, gpu_total, gpu_reserved = hours(r.gpu_quota)
    tpu_left, tpu_total, _ = hours(r.tpu_quota)
    return {"user": confirmed, "gpu_left": gpu_left, "gpu_total": gpu_total,
            "gpu_reserved": gpu_reserved,
            "tpu_left": tpu_left, "tpu_total": tpu_total,
            "refresh_at": r.quota_refresh_time.isoformat() if r.quota_refresh_time else ""}


def survey() -> tuple[dict[str, dict], dict[str, str]]:
    """Quota for every account holding CLI credentials; failures reported, never raised."""
    quotas, failures = {}, {}
    for u in known_users():
        if not creds_path(u).is_file():
            failures[u] = "no CLI credentials — needs a browser login (see multi-account.md)"
            continue
        try:
            quotas[u] = account_quota(u)
        except Exception as exc:                # a revoked refresh token surfaces here
            failures[u] = f"{type(exc).__name__}: quota unavailable; credential details suppressed"
    return quotas, failures


def print_survey(quotas: dict[str, dict], failures: dict[str, str], active: str | None) -> None:
    print(f"{'account':<16} {'GPU left':>10} {'TPU left':>10}  refreshes at")
    for u, q in sorted(quotas.items(), key=lambda kv: -kv[1]["gpu_left"]):
        mark = "  <-- ACTIVE" if u == active else ""
        print(f"{u:<16} {q['gpu_left']:>9.2f}h {q['tpu_left']:>9.2f}h  {q['refresh_at']}{mark}")
    for u, why in failures.items():
        print(f"{u:<16} {'?':>10} {'?':>10}  {why}")


def cmd_quota(args: argparse.Namespace) -> int:
    quotas, failures = survey()
    if not quotas and not failures:
        print(f"no accounts saved in {ACCOUNTS}")
        return 1
    print_survey(quotas, failures, active_user())
    return 0 if quotas else 1


def snapshot_identity(user: str) -> str:
    """Server-confirmed username for a snapshot, minting an access token in memory.

    Proves the refresh token is still honoured. Writes nothing: only
    `refresh_access_token()` saves, and it saves to the live path.
    """
    from kagglesdk.kaggle_client import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials

    client = KaggleClient()
    with client:
        creds = KaggleCredentials.load(client, file_path=str(creds_path(user)))
        if creds is None:
            raise RuntimeError("no usable refresh token in the snapshot")
        creds._access_token = creds.generate_access_token().token
        return creds.introspect()


def cmd_health(args: argparse.Namespace) -> int:
    """Can each saved refresh token still mint an access token, and is it labelled right?

    Cheaper than `quota` and answers the question a rotation actually depends on. A dead
    snapshot here is not proof of revocation on its own -- a network or service failure
    reads the same way -- so re-run before asking for a browser login.
    """
    users = known_users()
    if not users:
        print(f"no accounts saved in {ACCOUNTS}")
        return 1
    problems = 0
    rows = []
    for u in users:
        if not creds_path(u).is_file():
            rows.append((u, "NO OAUTH", "browser login needed before this account can be used"))
            problems += 1
            continue
        try:
            confirmed = snapshot_identity(u)
        except Exception as exc:
            rows.append((u, "DEAD", f"{type(exc).__name__}: refresh token did not mint a token"))
            problems += 1
            continue
        if confirmed != u:
            rows.append((u, "MISLABELLED", f"server says this snapshot belongs to {confirmed}"))
            problems += 1
            continue
        note = "" if token_path(u).is_file() else "OAuth fine, but no MCP token: `use` will refuse"
        rows.append((u, "ok", note))
    if args.quiet:
        rows = [r for r in rows if r[1] != "ok"]
        if not rows:
            print(f"all {len(users)} saved accounts healthy")
            return 0
    print(f"{'account':<16} {'refresh token':<13} note")
    for user, state, note in rows:
        print(f"{user:<16} {state:<13} {note}")
    return 1 if problems else 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Split a GPU budget into sessions across accounts, without changing anything.

    Answers "which accounts, in how many sessions, does this training plan need", which is
    the decision the user asked to drive rotation. It is arithmetic over a dated quota read,
    not a reservation: quota is not held, and another launch can consume the same hours.
    """
    session = min(args.session_hours, KAGGLE_SESSION_HOURS)
    quotas, failures = survey()
    active = active_user()
    excluded = set(args.exclude or [])
    if not quotas:
        print_survey(quotas, failures, active)
        print("No usable quota response; cannot plan.", file=sys.stderr)
        return 1

    def rank(item: tuple[str, dict]) -> tuple[int, float]:
        # Prefer staying put when it is equally good: a switch costs an MCP reload and
        # makes it harder to say which account paid for which rounds.
        return (0 if item[0] == active else 1, -item[1]["gpu_left"])

    need, rows, skipped = args.gpu_hours, [], []
    for user, q in sorted(quotas.items(), key=rank):
        if user in excluded:
            skipped.append((user, q, "excluded by --exclude"))
            continue
        if not token_path(user).is_file():
            skipped.append((user, q, "no MCP token: `use` refuses to switch to it"))
            continue
        if need <= 0:
            skipped.append((user, q, "not needed by this plan"))
            continue
        take = min(need, q["gpu_left"])
        if take <= 0:
            skipped.append((user, q, "no free GPU hours"))
            continue
        count = max(1, math.ceil(take / session))
        rows.append((user, q, take, count))
        need -= take

    print(f"Plan for {args.gpu_hours:.2f} h GPU, session cap {session:.2f} h "
          f"(service cap {KAGGLE_SESSION_HOURS:.0f} h)\n")
    print(f"{'account':<16} {'GPU left':>9} {'allocate':>9} {'sessions':>9}  note")
    for user, q, take, count in rows:
        note = "ACTIVE" if user == active else "needs a confirmed switch"
        if q.get("gpu_reserved", 0) > 0:
            note += f"; {q['gpu_reserved']:.2f}h reserved — a session may be live"
        print(f"{user:<16} {q['gpu_left']:>8.2f}h {take:>8.2f}h {count:>9}  {note}")
    for user, q, why in skipped:
        print(f"{user:<16} {q['gpu_left']:>8.2f}h {'-':>9} {'-':>9}  {why}")
    for user, why in failures.items():
        print(f"{user:<16} {'?':>9} {'-':>9} {'-':>9}  {why}")

    total_sessions = sum(c for _, _, _, c in rows)
    covered = args.gpu_hours - max(0.0, need)
    print(f"\ncovered {covered:.2f} h of {args.gpu_hours:.2f} h in {total_sessions} session(s) "
          f"across {len(rows)} account(s)")
    if len(rows) > 1:
        print("Session boundaries inside one account resume by attaching that account's previous "
              "output. A boundary BETWEEN accounts additionally needs the next owner to read the "
              "previous owner's private output, and the notebook's kernel_sources must name the "
              "PREVIOUS owner. Verify that read as the account that will run: group membership is "
              "not per-item access.")
    print("At most 2 concurrent GPU sessions per account; every switch needs the user's "
          "confirmation under this project's account policy.")
    if need > 0:
        soonest = min((q["refresh_at"] for q in quotas.values() if q["refresh_at"]), default="")
        print(f"\nSHORTFALL {need:.2f} h. Earliest quota refresh: {soonest or 'unknown'}",
              file=sys.stderr)
        return 2
    return 0


def cmd_ensure(args: argparse.Namespace) -> int:
    """Make sure the active account can pay for a run of --gpu-hours, switching only if not.

    Sticky by design: an account that still has enough hours is kept, because every switch
    costs an MCP client reload and makes it harder to say which account a run belongs to.
    """
    quotas, failures = survey()
    active = active_user()
    if not quotas:
        print_survey(quotas, failures, active)
        print("No usable quota response; no account changed.", file=sys.stderr)
        return 1

    def sufficient(q: dict) -> bool:
        return q["gpu_left"] >= args.gpu_hours and q["tpu_left"] >= args.tpu_hours

    need = f"{args.gpu_hours}h GPU" + (f" + {args.tpu_hours}h TPU" if args.tpu_hours else "")
    if active in quotas and sufficient(quotas[active]):
        q = quotas[active]
        print(f"keeping {active}: {q['gpu_left']:.2f}h GPU / {q['tpu_left']:.2f}h TPU left, "
              f"enough for {need}")
        return 0

    candidates = sorted((q for u, q in quotas.items()
                         if u != active and sufficient(q) and token_path(u).is_file()),
                        key=lambda q: -q["gpu_left"])
    if not candidates:
        print(f"no switchable account has {need} available (both credentials required).", file=sys.stderr)
        print_survey(quotas, failures, active)
        soonest = min((q["refresh_at"] for q in quotas.values() if q["refresh_at"]), default="")
        if soonest:
            print(f"\nEarliest quota refresh: {soonest}", file=sys.stderr)
        return 2

    chosen = candidates[0]["user"]
    args.username = chosen
    if not args.confirm:
        print(f"{active or 'the active account'} cannot pay for {need}.")
        print(f"proposed switch: {active or '(none)'} -> {chosen} "
              f"({candidates[0]['gpu_left']:.2f}h GPU / {candidates[0]['tpu_left']:.2f}h TPU left)")
        others = ", ".join(f"{c['user']} ({c['gpu_left']:.2f}h)" for c in candidates[1:])
        print(f"other accounts that could pay: {others or 'none'}")
        print_survey(quotas, failures, active)
        print(f"\nnothing changed. Ask the user, then re-run with --confirm.")
        return 3
    print(f"{active or 'the active account'} cannot pay for {need}; switching to {chosen} "
          f"({candidates[0]['gpu_left']:.2f}h GPU left)")
    return cmd_use(args)


def cmd_use(args: argparse.Namespace) -> int:
    user = args.username
    src, tok = creds_path(user), token_path(user)
    if not src.is_file():
        print(f"no saved credentials for {user}. Run `kaggle auth login --force` as that "
              f"account, then `{Path(__file__).name} save {user}`.", file=sys.stderr)
        return 1

    if not tok.is_file():
        print(f"no MCP token saved for {user}; no account changed.", file=sys.stderr)
        return 1
    token = tok.read_text().strip().removeprefix("Bearer ").strip()
    if not token.startswith("KGAT_") or any(c.isspace() for c in token):
        print("Invalid saved MCP token format; no account changed.", file=sys.stderr)
        return 1
    if json.loads(src.read_text()).get("username") != user:
        print("Saved credential username does not match; no account changed.", file=sys.stderr)
        return 1
    cfg = None
    if args.mcp_config.is_file():
        cfg = json.loads(args.mcp_config.read_text())
        cfg["mcpServers"]["kaggle"].setdefault("headers", {})

    prev = active_user()
    if prev and prev != user and not creds_path(prev).is_file():
        print(f"refusing: the active account '{prev}' has no snapshot and would be lost. "
              f"Run `save {prev}` first.", file=sys.stderr)
        return 1

    sync = args.mcp_config.parent / "scripts" / "sync_kaggle_mcp.py"
    if not args.confirm:
        print(f"would switch the CLI login: {prev or '(none)'} -> {user}")
        print(f"would rewrite {args.mcp_config}" if cfg is not None
              else f"would leave MCP headers alone: {args.mcp_config} does not exist")
        if sync.is_file():
            print(f"would run {sync.name}, updating the other host configs to {user}")
        print("nothing changed. This project requires the user's confirmation for every "
              "account switch; re-run with --confirm once they agree.")
        return 3

    with switch_lock():
        return _use_locked(args, user, src, token, cfg, prev)


def _use_locked(args: argparse.Namespace, user: str, src: Path, token: str,
                cfg: dict | None, prev: str | None) -> int:
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, LIVE)
    os.chmod(LIVE, 0o600)

    # Refresh before config-view, which may otherwise use an expired snapshot token.
    if cmd_refresh(args) != 0:
        print(f"CLI snapshot is now {user}, but refresh failed; host configs were not changed. "
              "Do not launch until authentication and account alignment are verified.", file=sys.stderr)
        return 1
    if cfg is None:
        mcp_note = "(this host has no .mcp.json to update)"
    else:
        headers = cfg["mcpServers"]["kaggle"]["headers"]
        headers.pop("authorization", None)
        headers["Authorization"] = f"Bearer {token}"
        args.mcp_config.write_text(json.dumps(cfg, indent=2) + "\n")
        os.chmod(args.mcp_config, 0o600)
        mcp_note = f"and {args.mcp_config.name}"

    # The project's sync script rewrites the other hosts' configs (.codex/config.toml,
    # .vscode/mcp.json) from the now-active account. Without it a rotation reaches the CLI
    # and Claude only, and Codex keeps naming the previous account.
    sync = args.mcp_config.parent / "scripts" / "sync_kaggle_mcp.py"
    if sync.is_file():
        r = subprocess.run([sys.executable, str(sync)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"CLI is now {user}, but {sync.name} failed; host configs may disagree. "
                  "Do not launch until synchronization succeeds.", file=sys.stderr)
            return 1
        mcp_note += f" and host configs via {sync.name}"

    print(f"active account is now {user} {mcp_note}")
    r = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "username" in line or "auth_method" in line:
            print("  " + line.strip())
    # CLI 2.2.x prints each value as "- username: NAME"; older versions had no bullet. Match the
    # field exactly either way (the exact-match test below is what stops "bob" accepting "bobby").
    if r.returncode != 0 or f"username: {user}" not in [line.strip().lstrip("- ")
                                                         for line in r.stdout.splitlines()]:
        print("WARNING: `kaggle config view` does not report this account. The saved "
              "refresh token may have been revoked — log in again and re-save.",
              file=sys.stderr)
        return 1
    print("\nRestart the session so the MCP server picks up the new header.")
    return 0


def cmd_add_account(args: argparse.Namespace) -> int:
    """Capture a new account without destroying the refresh token of the current one.

    `kaggle auth login --force` overwrites credentials.json, and the refresh token it
    replaces cannot be recovered. This snapshots the active account first, hands the
    terminal to the CLI's browser flow, then snapshots whatever account actually logged in.
    The browser step is a human action and needs a real terminal.
    """
    target = args.username
    if creds_path(target).is_file():
        print(f"{target} already has an OAuth snapshot. Use `refresh`, or delete "
              f"{creds_path(target).name} first if you mean to replace it.", file=sys.stderr)
        return 1
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("`add-account` needs an interactive terminal for the browser login. Run it "
              "yourself in a shell; an agent tool call cannot complete the verification step.",
              file=sys.stderr)
        return 1

    with switch_lock():
        prev = active_user()
        if prev:
            ACCOUNTS.mkdir(parents=True, exist_ok=True)
            os.chmod(ACCOUNTS, 0o700)
            shutil.copy2(LIVE, creds_path(prev))
            os.chmod(creds_path(prev), 0o600)
            print(f"snapshotted the active account {prev} before overwriting the login")
        print(f"\nLog in as {target} in the browser window the CLI opens.\n")
        if subprocess.run(["kaggle", "auth", "login", "--force"]).returncode != 0:
            print("login failed; the previous account's snapshot is intact but the live "
                  "login may be broken. Check `list`.", file=sys.stderr)
            return 1
        now = active_user()
        if now != target:
            print(f"the browser login produced '{now}', not '{target}'. Not saving under the "
                  f"wrong name. Re-run as {target}, or save it as {now}.", file=sys.stderr)
            return 1
        shutil.copy2(LIVE, creds_path(target))
        os.chmod(creds_path(target), 0o600)
        print(f"saved OAuth snapshot for {target}")

    if not token_path(target).is_file():
        print(f"no MCP token stored for {target}: write it to {token_path(target)} (mode 600) "
              f"before `use {target}` will work.")
    print(f"{target} is now the ACTIVE login.")
    if prev:
        print(f"To go back: {Path(__file__).name} use {prev} --confirm")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcp-config", type=Path, default=DEFAULT_MCP,
                    help=f"project .mcp.json to rewrite (default: {DEFAULT_MCP})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show saved accounts and which is active")
    sub.add_parser("quota", help="remaining GPU/TPU per account, without switching")
    sub.add_parser("refresh", help="mint a fresh access token now and re-snapshot")
    p = sub.add_parser("health", help="check every saved refresh token still works")
    p.add_argument("--quiet", action="store_true", help="print only accounts with a problem")
    p = sub.add_parser("plan", help="split a GPU budget into sessions across accounts")
    p.add_argument("--gpu-hours", type=float, required=True,
                   help="total GPU hours the training plan needs")
    p.add_argument("--session-hours", type=float, default=11.0,
                   help=f"per-session cap, at most {KAGGLE_SESSION_HOURS:.0f} (default: 11.0)")
    p.add_argument("--exclude", action="append", metavar="USER",
                   help="account the plan must not use (repeatable)")
    p = sub.add_parser("ensure", help="propose a switch when the active account cannot pay")
    p.add_argument("--gpu-hours", type=float, required=True,
                   help="GPU hours the next push needs (max_hours plus a margin)")
    p.add_argument("--tpu-hours", type=float, default=0.0,
                   help="TPU hours the next push needs (default: none)")
    p.add_argument("--confirm", action="store_true",
                   help="actually switch; without it the proposal is printed and exit is 3")
    p = sub.add_parser("save", help="snapshot the current login under a name")
    p.add_argument("username")
    p = sub.add_parser("add-account", help="snapshot current, browser-login, snapshot the new one")
    p.add_argument("username")
    p = sub.add_parser("use", help="make a saved account active (CLI + MCP)")
    p.add_argument("username")
    p.add_argument("--confirm", action="store_true",
                   help="actually switch; without it the proposal is printed and exit is 3")
    args = ap.parse_args()
    if args.cmd in ("ensure", "plan"):
        budgets = ([args.gpu_hours, args.tpu_hours] if args.cmd == "ensure"
                   else [args.gpu_hours, args.session_hours])
        if any(not math.isfinite(v) or v < 0 for v in budgets):
            ap.error("hour budgets must be finite and non-negative")
        if args.cmd == "plan" and args.session_hours <= 0:
            ap.error("--session-hours must be positive")
    names = []
    if args.cmd in ("use", "save", "add-account"):
        names.append(args.username)
    if args.cmd == "plan":
        names.extend(args.exclude or [])
    for name in names:
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
            ap.error("invalid account name")
    return {"list": cmd_list, "save": cmd_save, "use": cmd_use, "quota": cmd_quota,
            "ensure": cmd_ensure, "refresh": cmd_refresh, "health": cmd_health,
            "plan": cmd_plan, "add-account": cmd_add_account}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

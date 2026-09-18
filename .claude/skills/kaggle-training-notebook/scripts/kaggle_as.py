#!/usr/bin/env python
"""Run one command as a SAVED Kaggle account without switching the live login.

    python kaggle_as.py <user> -- kaggle kernels status <user>/<slug>
    python kaggle_as.py <user> -- python <script.py> ...

Two kinds of saved account, both under ~/.kaggle/accounts/:
  * `<user>.credentials.json` (OAuth snapshot): a short-lived access token is minted from its
    refresh token IN MEMORY, the server is asked whom it belongs to, and it is handed to the
    child through KAGGLE_API_TOKEN. Nothing on disk changes, nothing is printed.
  * `<user>.mcp-token` only (a KGAT_ API token, no OAuth yet): the child gets the FILE PATH in
    KAGGLE_API_TOKEN -- kagglesdk 2.2.x reads a path there -- so the value never enters a
    command line, a log or this process's memory.

This is the non-mutating complement of the skill's `kaggle_account.py use --confirm`: pulling
one run's output as its owner while pushing another as a different account needs no switch,
no MCP reconnect and no OAuth snapshot for a token-only account. Verified identity is still
required for every OAuth path (the introspected username must match the filename).
"""
import os, subprocess, sys
from pathlib import Path


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        sys.exit(__doc__)
    user, args = sys.argv[1], sys.argv[3:]
    acc = Path.home() / ".kaggle" / "accounts"
    snap, tokf = acc / f"{user}.credentials.json", acc / f"{user}.mcp-token"
    env = dict(os.environ)
    if snap.is_file():
        from kagglesdk.kaggle_client import KaggleClient
        from kagglesdk.kaggle_creds import KaggleCredentials
        with KaggleClient() as client:
            creds = KaggleCredentials.load(client, file_path=str(snap))
            if creds is None:
                sys.exit(f"{snap.name}: no usable refresh token")
            creds._access_token = creds.generate_access_token().token
            who = creds.introspect()
        if who != user:
            sys.exit(f"snapshot is labelled {user} but the server says {who}; refusing")
        env["KAGGLE_API_TOKEN"] = creds._access_token
    elif tokf.is_file():
        env["KAGGLE_API_TOKEN"] = str(tokf)
    else:
        sys.exit(f"no saved credential for {user} under {acc}")
    sys.exit(subprocess.run(args, env=env).returncode)


if __name__ == "__main__":
    main()

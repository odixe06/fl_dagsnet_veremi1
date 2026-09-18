"""Codex HTTP header helper. Stdout is a secret channel consumed by Codex.

Do not run this directly in a terminal or agent tool output. Use probe_kaggle_mcp.py.
Read the MCP token belonging to the current CLI account without copying OAuth tokens.
"""

import json
from pathlib import Path


def account_headers():
    directory = Path.home() / ".kaggle"
    user = json.loads((directory / "credentials.json").read_text())["username"]
    if not user or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in user):
        raise ValueError("Invalid Kaggle account name")
    token = (directory / "accounts" / f"{user}.mcp-token").read_text().strip()
    token = token.removeprefix("Bearer ").strip()
    if not token.startswith("KGAT_") or any(c.isspace() for c in token):
        raise ValueError("Invalid MCP token format")
    return user, {"Authorization": "Bearer " + token}


if __name__ == "__main__":
    try:
        _, headers = account_headers()
    except (OSError, ValueError, KeyError):
        raise SystemExit("Cannot load MCP token for the active Kaggle account")
    print(json.dumps(headers))

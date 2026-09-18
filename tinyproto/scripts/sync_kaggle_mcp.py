"""Sync local MCP configs from ~/.kaggle/accounts for the current CLI account.

Never prints credentials or switches accounts. Codex reads the token through its
header helper; Claude/VS Code receive local private headers in their existing files.
"""

import json
import os
import re
import shlex
import sys
import tempfile
import tomllib
from pathlib import Path

from kaggle_mcp_headers import account_headers


def private_write(path, text):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    user, headers = account_headers()
    root = Path(__file__).resolve().parents[1]
    path = root / ".codex/config.toml"
    original = path.read_text()
    config = tomllib.loads(original)["mcp_servers"]["kaggle"]
    if config["url"] != "https://www.kaggle.com/mcp":
        raise ValueError("Unexpected Kaggle endpoint")
    helper = shlex.join([sys.executable, str(root / "scripts/kaggle_mcp_headers.py")])
    match = re.search(r"(?ms)^\[mcp_servers\.kaggle\]\s*\n(.*?)(?=^\[|\Z)", original)
    if match is None:
        raise ValueError("Kaggle configuration section missing")
    section = match.group(1)
    for key in ("env_http_headers", "http_headers", "bearer_token_env_var"):
        if config.get(key) and (not isinstance(config[key], dict)
                               or set(config[key]) != {"Authorization"}):
            raise ValueError("Review nonstandard authentication before replacing it")
    section = re.sub(r"(?m)^(?:env_http_headers|http_headers|http_headers_helper|bearer_token_env_var)\s*=.*\n?",
                     "", section)
    section = "http_headers_helper = " + json.dumps(helper) + "\n" + section
    updated = original[:match.start(1)] + section + original[match.end(1):]
    parsed = tomllib.loads(updated)["mcp_servers"]["kaggle"]
    if parsed.get("http_headers_helper") != helper or parsed.get("env_http_headers"):
        raise ValueError("Codex header configuration did not validate")
    changes = [(path, updated)]
    for filename, key in ((".mcp.json", "mcpServers"), (".vscode/mcp.json", "servers")):
        path = root / filename
        data = json.loads(path.read_text())
        server = data[key]["kaggle"]
        if server["url"] != "https://www.kaggle.com/mcp":
            raise ValueError("Unexpected Kaggle endpoint")
        server.setdefault("headers", {}).pop("authorization", None)
        server["headers"].update(headers)
        changes.append((path, json.dumps(data, indent=2) + "\n"))
    for path, text in changes:
        private_write(path, text)
    print(json.dumps({"active_account": user, "updated": [str(p.relative_to(root)) for p, _ in changes],
                      "codex_auth": "http_headers_helper reads saved account token",
                      "permissions": "0600", "credentials_printed": False,
                      "running_client_reloaded": False}))


if __name__ == "__main__":
    main()

"""Read-only MCP account probe; never prints or changes credentials.

Select saved account credentials or an on-disk host configuration explicitly.
This cannot update the running IDE's environment or authenticate its existing tools.
"""

import argparse
import json
import os
import tomllib
from pathlib import Path

import requests


def payload(response):
    response.raise_for_status()
    if "application/json" in response.headers.get("Content-Type", ""):
        return response.json()
    for line in response.text.splitlines():
        if line.startswith("data:"):
            candidate = json.loads(line[5:].strip())
            if "result" in candidate or "error" in candidate:
                return candidate
    raise ValueError("No JSON-RPC response")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["saved", "oauth", "codex", "claude", "vscode"],
                        default="codex")
    parser.add_argument("--account", help="Explicit saved account for a read-only probe")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source
    authorization = None
    if source == "oauth":
        credentials = json.loads((Path.home() / ".kaggle/credentials.json").read_text())
        authorization = "Bearer " + credentials["access_token"]
        source = f"current OAuth access token for {credentials['username']}"
    elif source == "saved":
        directory = Path.home() / ".kaggle"
        user = args.account or json.loads((directory / "credentials.json").read_text())["username"]
        if not user or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in user):
            raise ValueError("Invalid account name")
        token = (directory / "accounts" / f"{user}.mcp-token").read_text().strip()
        authorization = "Bearer " + token.removeprefix("Bearer ").strip()
        source = f"saved MCP token for account {user}"
    elif source == "codex":
        config = tomllib.loads((root / ".codex/config.toml").read_text())["mcp_servers"]["kaggle"]
        authorization = config.get("http_headers", {}).get("Authorization")
        if config.get("http_headers_helper"):
            import shlex
            import subprocess

            command = shlex.split(config["http_headers_helper"])
            expected = str(root / "scripts/kaggle_mcp_headers.py")
            if len(command) != 2 or command[1] != expected:
                raise ValueError("Unrecognized header helper; review before execution")
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            authorization = json.loads(result.stdout)["Authorization"]
        variable = config.get("env_http_headers", {}).get("Authorization")
        if variable:
            authorization = os.environ.get(variable)
    else:
        path, key = ((".mcp.json", "mcpServers") if source == "claude"
                     else (".vscode/mcp.json", "servers"))
        authorization = json.loads((root / path).read_text())[key]["kaggle"].get(
            "headers", {}).get("Authorization")
    if not authorization:
        print(json.dumps({"authenticated": False, "reason": "credential unavailable"}))
        return 1
    session = requests.Session()
    session.headers.update({"Authorization": authorization,
                            "Accept": "application/json, text/event-stream"})
    endpoint = "https://www.kaggle.com/mcp"

    def call(method, params, request_id):
        response = session.post(endpoint, json={"jsonrpc": "2.0", "id": request_id,
                                                "method": method, "params": params},
                                timeout=30, allow_redirects=False)
        if response.headers.get("Mcp-Session-Id"):
            session.headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
        document = payload(response)
        if "error" in document:
            raise ValueError("JSON-RPC error")
        return document["result"]

    try:
        result = call("initialize", {"protocolVersion": "2024-11-05",
                                    "capabilities": {},
                                    "clientInfo": {"name": "afpha-readonly-probe",
                                                   "version": "1.0"}}, 1)
        session.headers["MCP-Protocol-Version"] = result["protocolVersion"]
        response = session.post(endpoint, json={"jsonrpc": "2.0",
                                                "method": "notifications/initialized"},
                                timeout=30, allow_redirects=False)
        response.raise_for_status()
        inventory = call("tools/list", {}, 2)["tools"]
        matches = [tool for tool in inventory if tool["name"] == "get_accelerator_quota"]
        if len(matches) != 1:
            raise ValueError("Account quota tool missing")
        schema = matches[0]["inputSchema"]
        arguments = {"request": {}} if "request" in schema.get("properties", {}) else {}
        quota = call("tools/call", {"name": matches[0]["name"], "arguments": arguments}, 3)
        if quota.get("isError"):
            error_text = " ".join(item.get("text", "") for item in quota.get("content", [])
                                  if item.get("type") == "text")
            for secret in (authorization, authorization.removeprefix("Bearer ").strip()):
                if secret:
                    error_text = error_text.replace(secret, "[redacted]")
            print(json.dumps({"authenticated": False, "credential_source": source,
                              "reason": "account-scoped quota tool returned an error",
                              "tool_error": error_text[:1000],
                              "ide_connection_changed": False}))
            return 1
        print(json.dumps({"authenticated": True, "credential_source": source,
                          "evidence": quota, "ide_connection_changed": False}))
        return 0
    except requests.HTTPError as error:
        print(json.dumps({"authenticated": False, "http_status": error.response.status_code,
                          "credential_source": source, "ide_connection_changed": False}))
        return 1
    except (requests.RequestException, ValueError, KeyError) as error:
        print(json.dumps({"authenticated": False, "error_type": type(error).__name__,
                          "credential_source": source, "ide_connection_changed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

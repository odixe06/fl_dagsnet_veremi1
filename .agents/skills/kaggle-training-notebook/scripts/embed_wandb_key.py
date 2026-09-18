#!/usr/bin/env python3
"""Embed the W&B credential from netrc into one explicitly private notebook."""

from __future__ import annotations

import argparse
import json
import netrc
import os
import re
from pathlib import Path


BEGIN = "# BEGIN INLINE WANDB CREDENTIAL — owner-authorized private notebook"
END = "# END INLINE WANDB CREDENTIAL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--netrc", type=Path, default=Path.home() / ".netrc")
    return parser.parse_args()


def read_key(path: Path) -> str:
    auth = netrc.netrc(path).authenticators("api.wandb.ai")
    if not auth or not auth[2]:
        raise SystemExit(f"No api.wandb.ai credential found in {path}")
    return auth[2]


def replace_login(source: str, key: str) -> str:
    replacement = (
        f"{BEGIN}\n"
        f"wandb.login(key={json.dumps(key)}, relogin=True, verify=True)\n"
        f"{END}\n"
    )
    inline = re.compile(rf"{re.escape(BEGIN)}\n.*?{re.escape(END)}\n", re.DOTALL)
    if inline.search(source):
        return inline.sub(replacement, source, count=1)

    attached_secret = re.compile(
        r'from kaggle_secrets import UserSecretsClient\n\n'
        r'try:\n.*?del _wandb_key\n',
        re.DOTALL,
    )
    if not attached_secret.search(source):
        raise SystemExit("Could not find the expected W&B authentication block")
    source = attached_secret.sub(replacement, source, count=1)
    return source.replace(
        "# W&B fails fast before dataset decode or model preparation. The key is never persisted.",
        "# W&B authenticates before dataset decode; inline key use was explicitly authorized by the owner.",
    )


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("is_private") is not True:
        raise SystemExit("Refusing to embed a credential: kernel metadata is not private")
    if metadata.get("code_file") != args.notebook.name:
        raise SystemExit("Metadata code_file does not match the target notebook")

    key = read_key(args.netrc)
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    matches = 0
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        # The credential cell is the one holding the authentication block — NOT
        # necessarily the one calling wandb.init(). In a DDP notebook the parent process
        # authenticates and a spawned rank-0 worker opens the run, so the two live in
        # different cells and keying off wandb.init() finds nothing.
        is_credential_cell = (BEGIN in source
                              or "from kaggle_secrets import UserSecretsClient" in source
                              or "wandb.init(" in source)
        if cell.get("cell_type") == "code" and is_credential_cell:
            if BEGIN not in source and "UserSecretsClient" not in source:
                continue          # a bare wandb.init() cell carries no credential
            cell["source"] = replace_login(source, key).splitlines(keepends=True)
            matches += 1
        elif cell.get("cell_type") == "markdown":
            updated = source.replace(
                "The W&B credential is read at runtime from the attached private Kaggle secret "
                "`wandb_key`; no key is\npersisted in source or artifacts.",
                "The owner explicitly authorized embedding the W&B credential in this private "
                "notebook so CLI batch runs require no Add-ons step.",
            )
            cell["source"] = updated.splitlines(keepends=True)

    if matches != 1:
        raise SystemExit(f"Expected one W&B initialization cell, found {matches}")

    temp = args.notebook.with_suffix(args.notebook.suffix + ".tmp")
    temp.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.chmod(temp, 0o600)
    os.replace(temp, args.notebook)
    print(f"Embedded W&B credential into private notebook {args.notebook}; value not displayed")


if __name__ == "__main__":
    main()

"""Read the W&B API key from ~/.netrc for embedding into a PRIVATE Kaggle notebook.

Why this exists: a Kaggle CLI/API push cannot attach an editor-selected secret, so a notebook
pushed from here cannot reach W&B through `UserSecretsClient`. The owner of this environment
gave a standing authorization (2026-09-07) to embed their own key in their own `is_private: true`
notebooks instead.

The cost is restated wherever this is used, not assumed known: the key becomes PERMANENT in
Kaggle's private version history, sharing or publishing such a notebook leaks it, and rotating
the key means rebuilding and re-pushing every notebook carrying it. Removal procedure:
`.agents/skills/kaggle-training-notebook/references/wandb.md` §1.1 -- revoke at W&B FIRST.

The key is never printed and never passed as a shell argument.
"""
from __future__ import annotations

import netrc
from pathlib import Path

WANDB_HOST = "api.wandb.ai"


def wandb_key() -> str | None:
    """The key from ~/.netrc, or None. Never logs or prints it."""
    p = Path.home() / ".netrc"
    if not p.exists():
        return None
    try:
        auth = netrc.netrc(str(p)).authenticators(WANDB_HOST)
    except Exception:
        return None
    if not auth:
        return None
    password = auth[2]
    return password or None


def redacted(key: str | None) -> str:
    """A safe identifier for logs: enough to match against a rotation, not enough to use."""
    return f"{key[:4]}...({len(key)} chars)" if key else "none"


if __name__ == "__main__":       # deliberately does NOT print the key
    k = wandb_key()
    print(f"wandb key in ~/.netrc: {'yes' if k else 'no'} {redacted(k) if k else ''}")

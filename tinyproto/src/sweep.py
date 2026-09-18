"""Selection contracts shared by the sweep notebooks and production preflight."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

# The mu-selection protocol, frozen. Production does not merely check that *a* sweep finished;
# it checks that the sweep which produced the number ran THIS procedure. Without this, a summary
# from a shortened grid, a different validation split, or an older criterion is indistinguishable
# from a valid one -- every one of those was reproduced against the previous contract.
# Bump `version` only with a deliberate decision; old summaries then stop validating, which is
# the point.
MU_PROTOCOL = {
    "version": 1,
    "grid": [0.03, 0.1, 0.3, 1.0, 3.0],
    "rounds": 4,
    "val_fraction": 0.02,
    "val_seed": 42,
    "mu_kind": "inv_mean_nij",
    "criterion": "mean-over-clients proto f1_macro at the last sweep round",
}

# `rounds` is deliberately NOT part of the signature: the sweep runs 4 and production runs 50,
# and they must still be recognised as the same scientific configuration. The protocol block
# above is what pins the sweep's own round count.
_SIGNATURE_KEYS = ("scenario", "n_clients", "batch", "local_epochs", "lr", "weight_decay",
                   "betas", "eps", "clip", "lam", "amp", "seed", "cps_s", "mask_seed",
                   "model_cfg", "feature_cols", "class_names", "data_fingerprint",
                   "scaler_fingerprint")


def selection_signature(cfg: dict) -> str:
    missing = [k for k in _SIGNATURE_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"cfg is missing signature keys: {missing}")
    return hashlib.sha256(json.dumps({k: cfg[k] for k in _SIGNATURE_KEYS},
                                     sort_keys=True).encode()).hexdigest()


def candidate_name(scenario: str, k: float) -> str:
    """One definition of a grid candidate's run name, shared by the sweep and its resume gate."""
    return f"musweep_{scenario}_k{str(k).replace('.', 'p')}"


def choose_winner(results: dict, grid: list, rounds: int) -> dict | None:
    """An interrupted/failed candidate cannot win, nor can an incomplete grid."""
    for k in grid:
        r = results.get(str(k), {})
        if (r.get("rounds") != rounds or "error" in r
                or not math.isfinite(r.get("last_proto_f1_macro", float("nan")))):
            return None
    win = max(grid, key=lambda k: results[str(k)]["last_proto_f1_macro"])
    return {"k": float(win), "mu_absolute": results[str(win)]["mu_absolute"],
            "criterion": MU_PROTOCOL["criterion"]}


def _check_protocol(document: dict) -> None:
    proto = document.get("protocol")
    if proto != MU_PROTOCOL:
        raise ValueError(
            "mu sweep did not run the frozen selection protocol "
            f"(expected {MU_PROTOCOL}, document has {proto})")
    if document.get("grid") != MU_PROTOCOL["grid"]:
        raise ValueError(f"sweep grid {document.get('grid')} != protocol {MU_PROTOCOL['grid']}")
    if document.get("rounds") != MU_PROTOCOL["rounds"]:
        raise ValueError(f"sweep ran {document.get('rounds')} rounds, protocol requires "
                         f"{MU_PROTOCOL['rounds']}")
    vf = document.get("validation_fingerprint") or {}
    if vf.get("fraction") != MU_PROTOCOL["val_fraction"] or vf.get("seed") != MU_PROTOCOL["val_seed"]:
        raise ValueError(
            f"validation split (fraction={vf.get('fraction')}, seed={vf.get('seed')}) does not "
            f"match the protocol (fraction={MU_PROTOCOL['val_fraction']}, "
            f"seed={MU_PROTOCOL['val_seed']})")


def selected_mu(document: dict, cfg: dict) -> tuple[float, dict]:
    """The absolute mu production may use, or an exception saying why it may not.

    Every check here exists because the weaker contract accepted a document that failed it.
    """
    if document.get("status") != "complete" or "winner" not in document:
        raise ValueError("mu sweep is incomplete; finish every candidate before production")
    if document.get("selection_signature") != selection_signature(cfg):
        raise ValueError("mu sweep belongs to a different scenario or scientific configuration")
    _check_protocol(document)

    results = document.get("results") or {}
    grid = MU_PROTOCOL["grid"]
    missing = [k for k in grid if str(k) not in results]
    if missing:
        raise ValueError(f"sweep results are missing grid candidates {missing}")

    # Recompute rather than trust: a stored winner is a claim, and the results it claims to
    # summarise are right there in the same document.
    recomputed = choose_winner(results, grid, MU_PROTOCOL["rounds"])
    if recomputed is None:
        raise ValueError("sweep results do not support any winner (a candidate is short, "
                         "failed, or has a non-finite score)")
    stored = document["winner"]
    if float(stored.get("k", float("nan"))) != recomputed["k"]:
        raise ValueError(f"stored winner k={stored.get('k')} disagrees with the winner "
                         f"recomputed from results (k={recomputed['k']})")
    if not math.isclose(float(stored["mu_absolute"]), float(recomputed["mu_absolute"]),
                        rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("stored winner mu_absolute disagrees with the value in results")

    value = float(stored["mu_absolute"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("selected mu must be finite and positive")
    return value, stored


def read_claims(scenario: str, search_roots: list) -> dict:
    """Committed-round counts a previous session recorded for this scenario's candidates."""
    claims: dict[str, int] = {}
    for base in search_roots:
        base = Path(base)
        if not base.exists():
            continue
        for p in list(base.rglob("mu_sweep_progress.json")) + list(base.rglob("mu_sweep.json")):
            try:
                doc = json.loads(p.read_text())
            except Exception:
                continue
            if doc.get("scenario") != scenario:
                continue
            for k, v in (doc.get("candidates") or {}).items():
                claims[k] = max(claims.get(k, 0), int(v.get("committed_rounds", 0)))
            for k, v in (doc.get("results") or {}).items():
                if isinstance(v, dict) and isinstance(v.get("rounds"), int):
                    claims[k] = max(claims.get(k, 0), v["rounds"])
    return claims


def plan_sweep(grid: list, scenario: str, search_roots: list, claims: dict | None = None) -> dict:
    """Committed rounds per grid candidate, so the resume gate can be applied per candidate.

    A sweep must never set `require_resume` for the whole grid: on a fresh sweep every candidate
    has nothing to resume by construction, and on a continuation the candidates after the one
    that ran out of budget were never started. Both are normal, and neither is an attachment
    error -- but a *forgotten* attachment looks exactly like a fresh sweep and silently retrains
    the whole grid, which is what this distinguishes.

    `claims` (from `read_claims`) records what a previous session said it committed. A candidate
    that was recorded as committed but whose bundle is absent means the attachment is PARTIAL:
    scheduling it fresh would silently discard finished work and build the final grid out of two
    different sweeps. That is an error, not a fresh start.

    Matching is by run_name and committed markers only. The exact fingerprint check stays in
    `driver.run`'s importer, which refuses a source whose scientific configuration differs.
    """
    claims = claims or {}
    plan, lost = {}, []
    for k in grid:
        name = candidate_name(scenario, k)
        committed = 0
        for base in search_roots:
            base = Path(base)
            if not base.exists():
                continue
            for done_dir in base.rglob(f"{name}/complete"):
                committed = max(committed, len(list(done_dir.glob("round_*.done"))))
        claimed = int(claims.get(str(k), 0))
        if claimed > committed:
            lost.append(f"k={k} ({name}): a previous session recorded {claimed} committed "
                        f"round(s) but only {committed} are present")
        plan[str(k)] = {"run_name": name, "committed_rounds": committed, "claimed_rounds": claimed}
    if lost:
        raise RuntimeError(
            "Partial sweep attachment -- refusing to continue:\n  " + "\n  ".join(lost) +
            "\nAttach the complete previous sweep output. Restarting these candidates would "
            "mix two different sweeps into one grid.")
    return plan

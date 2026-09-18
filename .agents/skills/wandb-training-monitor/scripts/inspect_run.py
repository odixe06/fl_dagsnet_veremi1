#!/usr/bin/env python3
"""Read one W&B run and emit a compact JSON health snapshot."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


DEFAULT_KEYS = (
    "progress/round",
    "progress/completed_round",
    "progress/global_step",
    "progress/round_step",
    "train/loss",
    "train/skip_pct",
    "train/grad_norm",
    "train/abs_logit_max",
    "train/softmax_entropy",
    "round/duration_s",
    "eval/f1_macro",
    "eval/accuracy",
)


def load_api_key(path: Path) -> str | None:
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, sep, value = line.partition("=")
        if sep and name.strip() == "WANDB_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value or None
    return None


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def latest_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Coalesce sparse W&B rows so round logs do not hide the latest diagnostics."""
    result: dict[str, Any] = {}
    for row in rows:
        result.update({key: value for key, value in row.items() if value is not None})
    return result


def analyze(
    state: str,
    rows: list[dict[str, Any]],
    total_rounds: int | None,
    stale_minutes: float,
    now: float,
    started_at: float | None = None,
) -> dict[str, Any]:
    latest = latest_values(rows)
    alerts: list[str] = []
    lowered = state.lower()
    if lowered in {"failed", "crashed", "killed"}:
        alerts.append(f"run_state={lowered}")

    latest_timestamp = max(
        (float(row["_timestamp"]) for row in rows if finite_number(row.get("_timestamp"))),
        default=None,
    )
    reference_timestamp = latest_timestamp if latest_timestamp is not None else started_at
    stale_seconds = (
        None if reference_timestamp is None else max(0.0, now - reference_timestamp)
    )
    if lowered == "running" and (stale_seconds is None or stale_seconds > stale_minutes * 60):
        alerts.append("heartbeat_stale")

    for key in (
        "train/loss",
        "train/skip_pct",
        "train/grad_norm",
        "train/abs_logit_max",
        "train/softmax_entropy",
    ):
        value = latest.get(key)
        if value is not None and not finite_number(value):
            alerts.append(f"non_finite:{key}")
    if finite_number(latest.get("train/skip_pct")) and latest["train/skip_pct"] > 50:
        alerts.append("skip_rate_above_50pct")

    losses = [float(row["train/loss"]) for row in rows if finite_number(row.get("train/loss"))]
    if len(losses) >= 6:
        baseline = statistics.median(losses[: max(3, len(losses) // 3)])
        if baseline > 0 and losses[-1] > 2 * baseline:
            alerts.append("loss_more_than_2x_recent_baseline")

    def flag_jump(key: str, factor: float, floor: float, label: str) -> None:
        values = [float(row[key]) for row in rows if finite_number(row.get(key))]
        if len(values) < 2:
            return
        recent_baseline = statistics.median(values[:-1][-5:])
        initial_baseline = max(abs(values[0]), 1e-12)
        exploded_from_initial = max(values[1:]) > factor * initial_baseline
        jumped_now = values[-1] > factor * max(recent_baseline, 1e-12)
        if values[-1] > floor and (exploded_from_initial or jumped_now):
            alerts.append(label)

    flag_jump("train/grad_norm", 1_000.0, 1e6, "gradient_norm_explosion")
    flag_jump("train/abs_logit_max", 20.0, 100.0, "attention_logit_explosion")

    entropies = [
        float(row["train/softmax_entropy"])
        for row in rows
        if finite_number(row.get("train/softmax_entropy"))
    ]
    if len(entropies) >= 2 and entropies[-1] < 0.1 and statistics.median(entropies[:-1][-5:]) > 0.5:
        alerts.append("softmax_entropy_collapse")

    durations = [
        float(row["round/duration_s"])
        for row in rows
        if finite_number(row.get("round/duration_s")) and row["round/duration_s"] > 0
    ]
    completed_round = latest.get("progress/completed_round")
    eta_seconds = None
    if total_rounds and isinstance(completed_round, int) and durations:
        eta_seconds = max(0, total_rounds - completed_round - 1) * statistics.median(durations[-5:])

    return {
        "healthy": not alerts,
        "alerts": alerts,
        "latest_timestamp": latest_timestamp,
        "stale_seconds": stale_seconds,
        "eta_seconds": eta_seconds,
    }


def self_test() -> None:
    now = 1_800_000_000.0
    rows = [
        {"_timestamp": now - 1_000, "train/loss": 1.0, "progress/round": 0},
        {
            "_timestamp": now - 900,
            "train/loss": 1.1,
            "train/skip_pct": 60.0,
            "progress/round": 1,
            "progress/completed_round": 1,
            "round/duration_s": 100.0,
        },
    ]
    result = analyze("running", rows, 50, 10, now)
    assert "heartbeat_stale" in result["alerts"]
    assert "skip_rate_above_50pct" in result["alerts"]
    assert result["eta_seconds"] == 4_800.0
    starting = analyze("running", [], 50, 10, now, started_at=now - 60)
    assert starting["healthy"] and "heartbeat_stale" not in starting["alerts"]
    exploding = analyze(
        "running",
        [
            {"_timestamp": now - 2, "train/grad_norm": 1.0, "train/abs_logit_max": 25.0},
            {"_timestamp": now - 1, "train/grad_norm": 1e10, "train/abs_logit_max": 1000.0},
        ],
        50,
        10,
        now,
    )
    assert "gradient_norm_explosion" in exploding["alerts"]
    assert "attention_logit_explosion" in exploding["alerts"]
    print("self-test passed")


def iso_timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Exact ENTITY/PROJECT/RUN_ID")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--history", type=int, default=40)
    parser.add_argument("--stale-minutes", type=float, default=20.0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.run or len(args.run.split("/")) != 3:
        raise SystemExit("--run must be ENTITY/PROJECT/RUN_ID")
    if args.history < 2:
        raise SystemExit("--history must be at least 2")

    import wandb
    from wandb.errors import CommError

    key = load_api_key(args.env_file)
    if key:
        wandb.login(key=key, relogin=False, verify=True)
    api = wandb.Api(timeout=30)
    api.flush()
    try:
        run = api.run(args.run)
    except CommError as exc:
        raise SystemExit(f"W&B run lookup failed for {args.run}: {exc}") from None

    # `min_step` is a W&B step value, not a row offset. Heartbeats in this project use
    # optimizer steps (250, 500, ...), so subtracting `--history` from lastHistoryStep
    # discarded earlier rows and hid relative anomalies. Scan exact rows, then tail them.
    rows = list(run.scan_history(page_size=1000))
    rows = rows[-args.history :]

    config = dict(run.config or {})
    total_rounds = config.get("rounds")
    if not isinstance(total_rounds, int):
        total_rounds = config.get("ROUNDS")
    health = analyze(
        str(run.state),
        rows,
        total_rounds,
        args.stale_minutes,
        time.time(),
        started_at=iso_timestamp(getattr(run, "created_at", None)),
    )
    latest = latest_values(rows)
    selected = {key: latest[key] for key in DEFAULT_KEYS if key in latest}
    output = {
        "run": args.run,
        "name": run.name,
        "url": run.url,
        "state": run.state,
        "history_rows_inspected": len(rows),
        "latest": selected,
        **health,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

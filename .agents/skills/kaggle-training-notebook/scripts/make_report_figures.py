#!/usr/bin/env python
"""Generate report figures and a machine-readable summary from pulled run artifacts."""

from __future__ import annotations

import argparse
import json
from numbers import Number
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_HISTORY = {"round", "accuracy", "f1_macro", "f1_weighted"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Pulled run directory containing metrics/, confusion/, and meta.json",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        help="Output directory; defaults to <paper-dir>/figures",
    )
    return parser.parse_args()


def numeric_row(row: pd.Series) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in row.items():
        if isinstance(value, Number) and pd.notna(value):
            result[key] = int(value) if key == "round" else float(value)
    return result


def class_metrics(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cm = cm.astype(float, copy=False)
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    return precision, recall, f1, cm.sum(1)


def load_confusion(run_dir: Path, round_idx: int, class_count: int) -> np.ndarray:
    path = run_dir / "confusion" / f"round_{round_idx:03d}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing confusion matrix: {path}")
    cm = np.load(path)
    if cm.shape != (class_count, class_count):
        raise ValueError(f"{path} has shape {cm.shape}; expected {(class_count, class_count)}")
    return cm


def save_confusion(cm: np.ndarray, names: list[str], round_idx: int, path: Path) -> None:
    normalised = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(normalised, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Row-normalised confusion — round {round_idx}")
    for row in range(len(names)):
        for col in range(len(names)):
            value = normalised[row, col]
            if value > 0.005:
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value < 0.6 else "black",
                )
    fig.colorbar(image, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    figures_dir = (
        args.figures_dir.resolve()
        if args.figures_dir
        else (run_dir.parent.parent / "figures").resolve()
    )

    history_path = run_dir / "metrics" / "history.csv"
    meta_path = run_dir / "meta.json"
    if not history_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Expected {history_path} and {meta_path}")

    history = (
        pd.read_csv(history_path)
        .drop_duplicates(subset="round", keep="last")
        .sort_values("round")
        .reset_index(drop=True)
    )
    missing = sorted(REQUIRED_HISTORY - set(history.columns))
    if missing or history.empty:
        raise ValueError(f"Invalid history.csv; missing={missing}, rows={len(history)}")

    meta = json.loads(meta_path.read_text())
    names = list(meta["class_names"])
    final_round = int(history["round"].iloc[-1])
    peak_round = int(history.loc[history["f1_macro"].idxmax(), "round"])
    final_cm = load_confusion(run_dir, final_round, len(names))
    peak_cm = load_confusion(run_dir, peak_round, len(names))
    figures_dir.mkdir(parents=True, exist_ok=True)

    rounds = history["round"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for key, color in (
        ("accuracy", "#4C78A8"),
        ("f1_macro", "#F58518"),
        ("f1_weighted", "#54A24B"),
    ):
        ax.plot(rounds, history[key].to_numpy(), marker="o", ms=3, lw=1.4, color=color, label=key)
    ax.axvline(peak_round, color="#F58518", ls=":", lw=1)
    ax.set_xlabel("round")
    ax.set_ylabel("test metric")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if "train_loss" in history:
        ax2 = ax.twinx()
        train_line = ax2.plot(
            rounds,
            history["train_loss"].to_numpy(),
            ls="--",
            lw=1.2,
            color="#999999",
            label="train_loss",
        )
        ax2.set_ylabel("train loss")
        handles += train_line
        labels += ["train_loss"]
    n_test = int(meta.get("n_test", final_cm.sum()))
    ax.legend(handles, labels, loc="center right", fontsize=8)
    ax.set_title(f"Test metrics per round — full {n_test:,}-row test set")
    fig.tight_layout()
    fig.savefig(figures_dir / "convergence.png", dpi=150)
    plt.close(fig)

    save_confusion(
        final_cm,
        names,
        final_round,
        figures_dir / f"confusion_final_round{final_round:03d}.png",
    )
    if peak_round != final_round:
        save_confusion(
            peak_cm,
            names,
            peak_round,
            figures_dir / f"confusion_peak_round{peak_round:03d}.png",
        )

    final_precision, final_recall, final_f1, support = class_metrics(final_cm)
    peak_precision, peak_recall, peak_f1, _ = class_metrics(peak_cm)
    order = np.argsort(-support)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    if peak_round != final_round:
        ax.bar(
            x - 0.2,
            peak_f1[order],
            0.4,
            color="#F58518",
            label=f"round {peak_round} (post-hoc test peak)",
        )
        final_x = x + 0.2
        final_width = 0.4
    else:
        final_x = x
        final_width = 0.7
    ax.bar(final_x, final_f1[order], final_width, color="#4C78A8", label=f"round {final_round} (final)")
    ax.set_xticks(x)
    ax.set_xticklabels([names[idx] for idx in order], rotation=90, fontsize=8)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Per-class F1, classes ordered by descending test support")
    fig.tight_layout()
    fig.savefig(figures_dir / "per_class_f1.png", dpi=150)
    plt.close(fig)

    if peak_round != final_round:
        gap = history["f1_macro"].max() - history["f1_macro"].to_numpy()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.fill_between(rounds, 0, gap, color="#E45756", alpha=0.25)
        ax.plot(rounds, gap, color="#E45756", lw=1.5)
        ax.set_xlabel("round")
        ax.set_ylabel("f1_macro below post-hoc peak")
        ax.set_title("Generalisation gap relative to the observed test peak")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "generalisation_gap.png", dpi=150)
        plt.close(fig)

    final_row = history.loc[history["round"] == final_round].iloc[-1]
    peak_row = history.loc[history["round"] == peak_round].iloc[-1]
    summary = {
        "rounds_present": [int(value) for value in history["round"]],
        "final_round": final_round,
        "peak_round_post_hoc": peak_round,
        "final": numeric_row(final_row),
        "peak_post_hoc": numeric_row(peak_row),
        "per_class": {
            names[idx]: {
                "support": int(support[idx]),
                "final": {
                    "precision": float(final_precision[idx]),
                    "recall": float(final_recall[idx]),
                    "f1": float(final_f1[idx]),
                },
                "peak_post_hoc": {
                    "precision": float(peak_precision[idx]),
                    "recall": float(peak_recall[idx]),
                    "f1": float(peak_f1[idx]),
                },
            }
            for idx in range(len(names))
        },
    }
    if "seconds" in history:
        summary["wall_hours"] = float(history["seconds"].sum() / 3600)
        summary["mean_round_seconds"] = float(history["seconds"].mean())
    if "samples_per_sec" in history:
        summary["mean_samples_per_sec"] = float(history["samples_per_sec"].mean())

    (figures_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    outputs = sorted(path.name for path in figures_dir.iterdir() if path.is_file())
    print(f"run_dir: {run_dir}")
    print(f"figures_dir: {figures_dir}")
    print(f"final_round: {final_round}; peak_round_post_hoc: {peak_round}")
    print("wrote:", *outputs)


if __name__ == "__main__":
    main()

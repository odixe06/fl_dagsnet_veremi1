"""Figures for report.md, generated from the pulled run artifacts.

The training notebook renders its plots inline with plt.show() and writes no image files,
so every figure the report embeds is produced here, after the pull, from
metrics/history.csv and confusion/round_XXX.npy. Run from the repo root.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("papers/build1-cmso-before-extractor")
RUN = BASE / "runs" / "edl_cmso_r50_b4096"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

hist = (pd.read_csv(RUN / "metrics" / "history.csv")
          .drop_duplicates(subset="round", keep="last")
          .sort_values("round").reset_index(drop=True))
meta = json.loads((RUN / "meta.json").read_text())
names = meta["class_names"]
final = int(hist["round"].iloc[-1])
peak = int(hist.loc[hist["f1_macro"].idxmax(), "round"])

# ── 1 · convergence, with the train loss that produced it on a second axis ──────
fig, ax = plt.subplots(figsize=(10, 4.8))
for k, c in (("accuracy", "#4C78A8"), ("f1_macro", "#F58518"), ("f1_weighted", "#54A24B")):
    ax.plot(hist["round"], hist[k], marker="o", ms=3, lw=1.4, color=c, label=k)
ax.axvline(peak, color="#F58518", ls=":", lw=1)
ax.annotate(f"đỉnh f1_macro (round {peak})", xy=(peak, hist["f1_macro"].max()),
            xytext=(peak + 3, hist["f1_macro"].max() + 0.004), fontsize=8, color="#F58518")
ax.set_xlabel("round"); ax.set_ylabel("test metric"); ax.grid(alpha=.3)
ax2 = ax.twinx()
ax2.plot(hist["round"], hist["train_loss"], ls="--", lw=1.2, color="#999", label="train_loss")
ax2.set_ylabel("train loss")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8, framealpha=.9)
ax.set_title(f"Metric trên toàn bộ {meta['n_test']:,} dòng test, sau mỗi round")
fig.tight_layout(); fig.savefig(FIG / "convergence.png", dpi=150); plt.close(fig)

# ── 2 · row-normalised confusion at the two reported rounds ────────────────────
for rnd, tag in ((final, "final"), (peak, "peak")):
    cm = np.load(RUN / "confusion" / f"round_{rnd:03d}.npy")
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cmn, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("dự đoán"); ax.set_ylabel("thực tế")
    ax.set_title(f"Confusion chuẩn hoá theo hàng — round {rnd}")
    for i in range(len(names)):
        for j in range(len(names)):
            if cmn[i, j] > 0.005:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] < 0.6 else "black")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(FIG / f"confusion_{tag}_round{rnd:03d}.png", dpi=150); plt.close(fig)


def per_class(rnd):
    cm = np.load(RUN / "confusion" / f"round_{rnd:03d}.npy").astype(float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1, cm.sum(1)


# ── 3 · per-class F1 at both rounds, ordered by support ────────────────────────
# Ordered by SUPPORT, not by score: sorting by score puts the flattering classes first
# and hides that they are the rare ones.
p_f, r_f, f1_f, sup = per_class(final)
p_p, r_p, f1_p, _ = per_class(peak)
order = np.argsort(-sup)
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.bar(x - 0.2, f1_p[order], 0.4, color="#F58518", label=f"round {peak} (đỉnh, chọn bằng test)")
ax.bar(x + 0.2, f1_f[order], 0.4, color="#4C78A8", label=f"round {final} (cuối)")
ax.set_xticks(x); ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=8)
ax.set_ylabel("F1"); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.3)
ax.legend(fontsize=8)
ax.set_title("F1 theo lớp, sắp theo số dòng test giảm dần")
for xi, i in enumerate(order):
    ax.text(xi, 1.005, f"{sup[i] / 1000:.0f}k", ha="center", fontsize=6, color="#666")
fig.tight_layout(); fig.savefig(FIG / "per_class_f1.png", dpi=150); plt.close(fig)

# ── 4 · the generalisation gap, stated directly ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
gap = hist["f1_macro"].max() - hist["f1_macro"]
ax.fill_between(hist["round"], 0, gap, color="#E45756", alpha=.25)
ax.plot(hist["round"], gap, color="#E45756", lw=1.5)
ax.set_xlabel("round"); ax.set_ylabel("f1_macro thấp hơn đỉnh")
ax.set_title("Khoảng tụt so với đỉnh — train loss giảm đơn điệu trong khi test đi xuống")
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "generalisation_gap.png", dpi=150); plt.close(fig)

summary = {
    "rounds": int(len(hist)), "final_round": final, "peak_round": peak,
    "final": {k: float(hist.loc[hist["round"] == final, k].iloc[0])
              for k in hist.columns if k != "round"},
    "peak": {k: float(hist.loc[hist["round"] == peak, k].iloc[0])
             for k in hist.columns if k != "round"},
    "per_class": {names[i]: {"support": int(sup[i]),
                             "final": {"precision": float(p_f[i]), "recall": float(r_f[i]),
                                       "f1": float(f1_f[i])},
                             "peak": {"precision": float(p_p[i]), "recall": float(r_p[i]),
                                      "f1": float(f1_p[i])}}
                  for i in range(len(names))},
    "wall_hours": float(hist["seconds"].sum() / 3600),
    "mean_round_seconds": float(hist["seconds"].mean()),
    "mean_samples_per_sec": float(hist["samples_per_sec"].mean()),
}
(BASE / "figures" / "summary.json").write_text(json.dumps(summary, indent=2))
print("figures:", *(p.name for p in sorted(FIG.glob("*.png"))))
print(f"rounds {len(hist)}  final {final}  peak {peak}  wall {summary['wall_hours']:.2f} h")

"""Figures for report.md, built from the pulled run artifacts.

The notebooks render plots inline and write no image files, so every figure the report embeds
is produced here, after the pull, from metrics/history.csv and confusion/round_XXX.npy.

Run from the repo root:  python make_figures.py
"""
import csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
P = Path("papers")

BUILDS = {
    "build1": dict(dir=P/"build1-cmso-before-extractor/runs/edl_cmso_r50_b4096",
                   label="Build 1 — CMSO trước extractor", color="#9C755F"),
    "build2run1": dict(dir=P/"build2-paper-order/run1-wd1e4-diverged-r25/edl_cmso_v2_r50_b4096",
                       label="Build 2 run 1 — pipeline đầy đủ (phân kỳ r25)", color="#4C78A8"),
    "build2c50": dict(dir=P/"build2-paper-order/run3-c50-compiled-diverged-r22/edl_cmso_v2_c50",
                      label="Build 2 run 3 c50 — compiled (phân kỳ r22)", color="#72B7B2"),
    "build4": dict(dir=P/"build4-dagsnet-only/runs/edl_cmso_v4_dagsnet",
                   label="Build 4 — DAGSNet trần", color="#F58518"),
}
BUILDS["build2final"] = dict(
    dir=P/"build2-paper-order/run4-wd5e2-complete-50round/edl_cmso_v2_final",
    label="Build 2 final — pipeline đầy đủ, wd 5e-2 (50/50)", color="#54A24B")


def history(d):
    rows = list(csv.DictReader(open(d/"metrics"/"history.csv")))
    seen, out = set(), []
    for r in rows:                       # a resumed run can repeat a round; last wins
        out = [x for x in out if x["round"] != r["round"]] + [r]
    out.sort(key=lambda r: int(r["round"]))
    return out


def col(h, k):
    return np.array([int(r["round"]) for r in h]), np.array([float(r[k]) for r in h])


for b in BUILDS.values():
    b["h"] = history(b["dir"]) if (b["dir"]/"metrics"/"history.csv").exists() else []

names = json.loads((BUILDS["build4"]["dir"]/"meta.json").read_text())["class_names"]


# ── 1 · every build on one axis. Two panels, because a collapse to 0.02 and a
#       0.02-wide difference between healthy runs cannot share a y axis. ────────
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5),
                               gridspec_kw=dict(width_ratios=[1, 1.15]))
for ax, zoom in ((axL, False), (axR, True)):
    for b in BUILDS.values():
        if not b["h"]: continue
        x, y = col(b["h"], "f1_macro")
        ax.plot(x, y, lw=1.6, marker="o", ms=2.6, color=b["color"], label=b["label"])
        i = int(np.argmax(y))
        ax.plot(x[i], y[i], marker="*", ms=13, color=b["color"], mec="white", mew=.8, zorder=5)
    ax.set_xlabel("round (1 epoch mỗi round)"); ax.grid(alpha=.3)
axL.set_ylabel("test f1_macro")
axL.set_title("toàn cảnh — hai run phân kỳ rơi xuống 0,02")
axL.legend(fontsize=7.6, loc="center left")
axR.set_ylim(.78, .862)
axR.set_title("phóng to vùng lành mạnh — ★ là đỉnh của mỗi bản")
for b, r, yt in (("build2c50", 22, .7815), ("build2run1", 25, .7855)):
    axR.axvline(r, color=BUILDS[b]["color"], ls=":", lw=1.1)
    axR.annotate(f"phân kỳ r{r}", xy=(r, yt), xytext=(r + .8, yt),
                 fontsize=7.6, color=BUILDS[b]["color"])
fig.suptitle("f1_macro trên đủ 10.761.343 dòng test, mọi bản dựng", y=.99, fontsize=12)
fig.tight_layout(); fig.savefig(FIG/"convergence_all.png", dpi=150); plt.close(fig)

# ── 2 · build 4 in detail: the peak, the overfit, the session boundary ─────────
h = BUILDS["build4"]["h"]
x, f1 = col(h, "f1_macro"); _, acc = col(h, "accuracy"); _, loss = col(h, "train_loss")
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(x, f1, lw=1.7, marker="o", ms=3, color="#F58518", label="f1_macro")
ax.plot(x, acc, lw=1.3, marker="o", ms=2.4, color="#4C78A8", label="accuracy")
pk = int(np.argmax(f1))
ax.axvline(x[pk], color="#F58518", ls=":", lw=1.2)
ax.annotate(f"đỉnh r{x[pk]} = {f1[pk]:.5f}", xy=(x[pk], f1[pk]),
            xytext=(x[pk]+3.5, f1[pk]+.0035), fontsize=9, color="#F58518",
            bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5),
            arrowprops=dict(arrowstyle="->", color="#F58518", lw=.9))
ax.axvline(45.5, color="#888", ls="--", lw=1)
ax.annotate("ranh giới session\n(r0–45 | r46–49)", xy=(45.5, f1.max()-.0015),
            xytext=(34.5, f1.max()-.004), fontsize=7.5, color="#666", ha="left",
            bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5),
            arrowprops=dict(arrowstyle="->", color="#888", lw=.8))
ax.set_xlabel("round"); ax.set_ylabel("test metric"); ax.grid(alpha=.3)
ax2 = ax.twinx(); ax2.plot(x, loss, ls="--", lw=1.2, color="#999", label="train loss")
ax2.set_ylabel("train loss")
hh, ll = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(hh+h2, ll+l2, fontsize=8, loc="upper right")
ax.set_title("Build 4 — train loss giảm đơn điệu trong khi f1_macro đi xuống từ round 5")
fig.tight_layout(); fig.savefig(FIG/"build4_convergence.png", dpi=150); plt.close(fig)

# ── 3 · per-class: where dropping the extractor actually helps ─────────────────
def per_class_f1(d, rnd):
    c = np.load(d/"confusion"/f"round_{rnd:03d}.npy").astype(np.float64)
    tp = np.diag(c); fp = c.sum(0)-tp; fn = c.sum(1)-tp
    return np.where(2*tp+fp+fn > 0, 2*tp/np.maximum(2*tp+fp+fn, 1e-12), 0.0), c.sum(1)

# Baseline = the full pipeline run that STOOD UP (50/50, wd 5e-2), not run 1, which read
# 0.83081 once at round 0 and then destroyed itself. Comparing against a number a run could
# not hold is comparing against nothing.
b1, b4 = BUILDS["build2final"], BUILDS["build4"]
r1 = int(max(b1["h"], key=lambda r: float(r["f1_macro"]))["round"])
r4 = int(max(b4["h"], key=lambda r: float(r["f1_macro"]))["round"])
f_full, support = per_class_f1(b1["dir"], r1)
f_bare, _ = per_class_f1(b4["dir"], r4)
order = np.argsort(f_bare - f_full)
fig, ax = plt.subplots(figsize=(9.5, 6.4))
y = np.arange(len(order))
ax.barh(y-.2, f_full[order], .38, color="#54A24B",
        label=f"pipeline đầy đủ, wd 5e-2, 50/50 round (r{r1})")
ax.barh(y+.2, f_bare[order], .38, color="#F58518", label=f"DAGSNet trần (r{r4})")
for i, k in enumerate(order):
    d = f_bare[k]-f_full[k]
    ax.text(max(f_full[k], f_bare[k])+.012, i, f"{d:+.4f}", va="center", fontsize=7.4,
            color="#2A7A2A" if d > 0 else "#B03030")
ax.set_yticks(y); ax.set_yticklabels([names[k] for k in order], fontsize=8)
ax.set_xlabel("F1 mỗi lớp"); ax.set_xlim(0, 1.14); ax.grid(axis="x", alpha=.3)
# below the axes: every bar reaches far right, so an in-plot legend hides a row
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.06), ncol=2, frameon=False)
ax.set_title("Bỏ extractor giúp đúng những lớp khó (sắp theo mức thay đổi)")
fig.tight_layout(); fig.savefig(FIG/"per_class_f1.png", dpi=150); plt.close(fig)

# ── 4 · confusion at build 4's peak, row-normalised ────────────────────────────
c = np.load(b4["dir"]/"confusion"/f"round_{r4:03d}.npy").astype(np.float64)
cn = c / np.maximum(c.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(8.6, 7.4))
im = ax.imshow(cn, cmap="magma_r", vmin=0, vmax=1)
ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=90, fontsize=7); ax.set_yticklabels(names, fontsize=7)
for i in range(len(names)):
    for j in range(len(names)):
        if cn[i, j] >= .01:
            ax.text(j, i, f"{cn[i,j]:.2f}"[1:], ha="center", va="center", fontsize=5.6,
                    color="white" if cn[i, j] > .55 else "#333")
ax.set_xlabel("dự đoán"); ax.set_ylabel("thật")
ax.set_title(f"Build 4, round {r4} — confusion chuẩn hoá theo hàng (recall trên đường chéo)")
fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
fig.tight_layout(); fig.savefig(FIG/"confusion_build4_peak.png", dpi=150); plt.close(fig)

# ── 5 · the two divergences, as the report describes them ─────────────────────
fig, ax = plt.subplots(figsize=(10, 4.6))
for key in ("build2run1", "build2c50"):
    b = BUILDS[key]; x, y = col(b["h"], "f1_macro")
    ax.plot(x, y, lw=1.6, marker="o", ms=3, color=b["color"], label=b["label"])
    ax.annotate(f"dừng ở r{x[-1]}", xy=(x[-1], y[-1]), xytext=(x[-1]-6, y[-1]-.06),
                fontsize=8, color=b["color"],
                arrowprops=dict(arrowstyle="->", color=b["color"], lw=.9))
bf = BUILDS["build2final"]; x, y = col(bf["h"], "f1_macro")
ax.plot(x, y, lw=2.0, marker="o", ms=3, color=bf["color"],
        label="cùng pipeline, wd 5e-2 — 50/50 round, KHÔNG phân kỳ")
ax.set_xlabel("round"); ax.set_ylabel("test f1_macro"); ax.grid(alpha=.3)
ax.legend(fontsize=8, loc="center right")
ax.set_title("Cùng một pipeline, chỉ đổi weight_decay 1e-4 → 5e-2")
fig.tight_layout(); fig.savefig(FIG/"divergence_wd1e4.png", dpi=150); plt.close(fig)

print("figures written to", FIG.resolve())
for p in sorted(FIG.glob("*.png")):
    print(f"  {p.name:<32} {p.stat().st_size/1024:6.0f} KB")

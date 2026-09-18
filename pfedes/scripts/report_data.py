#!/usr/bin/env python
"""Numbers and figures for the pFedES report, rebuilt from pulled run artifacts only.

    python scripts/report_data.py --out papers/pfedes-yi-2025/report_data \
        20c=papers/pfedes-yi-2025/runs/merged/pfedes_20c_cos \
        50c=papers/pfedes-yi-2025/runs/pulls/50c_cos_s2/runs/pfedes_50c_cos \
        100c=papers/pfedes-yi-2025/runs/pulls/100c_cos_s2/runs/pfedes_100c_cos

Per scenario (K = 20/50/100 clients) it writes, under --out:
  history_Kc.csv          every round, every column of the run's history.csv (mean/std/min/max
                          of the 10 metrics over the N clients, lr, timing, skips)
  rounds_Kc.md            the same as a markdown table: all 10 metrics (mean over clients)
                          for EVERY round — the reporting contract, no round left out
  per_client_final_Kc.csv the 10 metrics of every client at the last completed round
  per_class_final_Kc.csv  per class at the last round: support, precision/recall/f1 as the
                          mean over clients AND as computed on the pooled (summed) confusion
  confusion_final_Kc.csv  pooled confusion matrix (sum over clients) of the last round
  summary.json            headline numbers of all scenarios: final round, post-hoc peak, timing,
                          sessions, config read from reports/manifest.json
and the figures convergence.png, client_spread.png, per_class_f1_Kc.png, confusion_Kc.png,
round_time.png. Rounds are read from metrics/round_NNN.json; nothing comes from W&B.
"""
import argparse, csv, json
from pathlib import Path

import numpy as np

METRICS = ("accuracy", "precision_macro", "precision_micro", "precision_weighted",
           "recall_macro", "recall_micro", "recall_weighted", "f1_macro", "f1_micro",
           "f1_weighted")
# dataviz reference palette, categorical slots 1-3 in fixed order; sequential = blue ramp
SERIES = {"20c": "#2a78d6", "50c": "#eb6834", "100c": "#1baf7a"}
BLUES = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load_run(d):
    rounds = sorted(int(p.stem.split("_")[1]) for p in (d / "metrics").glob("round_*.json"))
    rows = [json.loads((d / "metrics" / f"round_{r:03d}.json").read_text()) for r in rounds]
    assert [r["round"] for r in rows] == rounds
    mf = json.loads((d / "reports" / "manifest.json").read_text())
    return rounds, rows, mf


def prf_from_cm(cm):
    tp = np.diag(cm).astype(float)
    p = tp / np.maximum(cm.sum(0), 1)
    r = tp / np.maximum(cm.sum(1), 1)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return p, r, f


def write_history(d, out, tag):
    src = d / "history.csv"
    rows = list(csv.DictReader(open(src)))
    rows = sorted({int(r["round"]): r for r in rows}.values(), key=lambda r: int(r["round"]))
    with open(out / f"history_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    lines = ["| round | lr | " + " | ".join(METRICS) + " | f1_macro std | min | max | train s | eval s |",
             "|---:|---:|" + "---:|" * len(METRICS) + "---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['round']} | {float(r['lr']):.2e} | "
                     + " | ".join(f"{float(r[m]):.4f}" for m in METRICS)
                     + f" | {float(r['f1_macro_std']):.4f} | {float(r['f1_macro_min']):.4f} | "
                       f"{float(r['f1_macro_max']):.4f} | {float(r['train_sec']):.0f} | "
                       f"{float(r['eval_sec']):.0f} |")
    (out / f"rounds_{tag}.md").write_text(
        f"# {tag}: 10 metrics per round (mean over all N clients, full 10 761 343-row test set)\n\n"
        "accuracy = precision_micro = recall_micro = recall_weighted = f1_micro (single-label "
        "multiclass identity).\n\n" + "\n".join(lines) + "\n")
    return rows


def write_readme(out, summary):
    """Headline tables (Vietnamese) regenerated from summary.json — the report copies from here."""
    tags = list(summary)
    L = ["# Số liệu cho report pFedES (`_cos`: C = 100 %, AdamW, LR cosine 1e-3 → 1e-5, T = 50)", "",
         "Sinh tự động bởi `scripts/report_data.py` từ artifact đã kéo về và verify local; "
         "không có số nào lấy từ W&B. Mọi metric là **trung bình không trọng số trên N client**, "
         "mỗi client đo trên **toàn bộ 10 761 343 dòng test toàn cục** bằng F_k(x). "
         "Caveat bắt buộc: `CONTEXT.md` §8 (test toàn cục ≠ test riêng client của bài báo; "
         "C = 100 % ≠ Table 1–2 của bài báo; một seed).", ""]
    L += ["## 1. Tình trạng run", "", "| kịch bản | client | round | trạng thái | phiên Kaggle | round-time trung vị | tổng giờ round |",
          "|---|---:|---:|---|---|---:|---:|"]
    for t in tags:
        s = summary[t]
        sess = s["sessions"]
        ns = f"{len(sess)} phiên ({', '.join(str(x['n_rounds'] if i == 0 else len(x['rounds_produced'])) + ' round' for i, x in enumerate(sess))})" if sess else "chưa ghép (pull phiên gần nhất)"
        L.append(f"| {t} | {s['n_clients']} | **{s['rounds_completed']}/{s['rounds_planned']}** | "
                 f"{'HOÀN TẤT' if s['complete'] else 'ĐANG CHẠY — số liệu tạm'} | {ns} | "
                 f"{s['timing']['round_sec_median']/60:.1f} phút (train {s['timing']['train_sec_median']/60:.1f} + eval {s['timing']['eval_sec_median']/60:.1f}) | "
                 f"{s['timing']['total_round_hours']:.1f} h |")
    L += ["", "## 2. Round cuối — đủ 10 metric (mean trên client; std/min/max của f1_macro)", "",
          "accuracy = precision_micro = recall_micro = recall_weighted = f1_micro (đồng nhất thức đa lớp đơn nhãn).", "",
          "| metric | " + " | ".join(f"{t} (r{summary[t]['final_round']})" for t in tags) + " |",
          "|---|" + "---:|" * len(tags)]
    for m in METRICS:
        L.append(f"| {m} | " + " | ".join(f"{summary[t]['final'][m]['mean']:.4f}" for t in tags) + " |")
    L.append("| f1_macro std / min / max | " + " | ".join(
        f"{summary[t]['final']['f1_macro']['std']:.4f} / {summary[t]['final']['f1_macro']['min']:.4f} / {summary[t]['final']['f1_macro']['max']:.4f}" for t in tags) + " |")
    L.append("| client yếu nhất / mạnh nhất (f1_macro) | " + " | ".join(
        f"#{summary[t]['worst_client_final']['cid']} {summary[t]['worst_client_final']['f1_macro']:.3f} / #{summary[t]['best_client_final']['cid']} {summary[t]['best_client_final']['f1_macro']:.3f}" for t in tags) + " |")
    L += ["", "## 3. Hình dạng đường cong (round 1 → đỉnh hậu kiểm → cuối)", "",
          "Đỉnh là quan sát *sau khi chạy*, không phải checkpoint được chọn; số công bố là round cuối.", "",
          "| | " + " | ".join(tags) + " |", "|---|" + "---:|" * len(tags)]
    L.append("| f1_macro round 1 | " + " | ".join(f"{summary[t]['round1']['f1_macro']:.4f}" for t in tags) + " |")
    L.append("| f1_macro đỉnh (round) | " + " | ".join(f"{summary[t]['peak_f1_macro_posthoc']['f1_macro']:.4f} (r{summary[t]['peak_f1_macro_posthoc']['round']})" for t in tags) + " |")
    L.append("| f1_macro cuối | " + " | ".join(f"{summary[t]['final']['f1_macro']['mean']:.4f}" for t in tags) + " |")
    L.append("| giảm so đỉnh | " + " | ".join(f"−{summary[t]['drop_from_peak_pct']:.1f} %" for t in tags) + " |")
    L.append("| accuracy round 1 → cuối | " + " | ".join(f"{summary[t]['round1']['accuracy']:.4f} → {summary[t]['final']['accuracy']['mean']:.4f}" for t in tags) + " |")
    L += ["", "## 4. Kiểm soát chất lượng pipeline", "", "| | " + " | ".join(tags) + " |", "|---|" + "---:|" * len(tags)]
    L.append("| backend train/eval | " + " | ".join("/".join(summary[t]["backend"]) for t in tags) + " |")
    L.append("| tổng cache_mismatch | " + " | ".join(str(summary[t]["cache_mismatch_total"]) for t in tags) + " |")
    L.append("| evaluated = N mọi round | " + " | ".join("có" if summary[t]["evaluated_always_N"] else "KHÔNG" for t in tags) + " |")
    L.append("| fingerprint | " + " | ".join(f"`{summary[t]['fingerprint']}`" for t in tags) + " |")
    L += ["", "## 5. File", "",
          "| file | nội dung |", "|---|---|",
          "| `rounds_Kc.md`, `history_Kc.csv` | **mọi round**, đủ 10 metric (mean) + std/min/max, lr, train/eval giây — bảng bắt buộc của report |",
          "| `per_client_final_Kc.csv` | 10 metric của từng client ở round cuối |",
          "| `per_class_final_Kc.csv` | mỗi lớp ở round cuối: support, P/R/F1 trung bình trên client và trên confusion gộp |",
          "| `confusion_final_Kc.csv` | confusion gộp (tổng N client) round cuối, 16×16 |",
          "| `summary.json` | mọi số ở trên dạng máy đọc + cấu hình từ `reports/manifest.json` + provenance phiên |",
          "| `convergence.png` | f1_macro / accuracy (mean ± std) theo round, cả ba K; lịch LR |",
          "| `client_spread.png` | phân bố f1_macro theo client ở round cuối |",
          "| `per_class_f1_Kc.png`, `confusion_Kc.png` | F1 từng lớp (theo tỉ lệ test) và confusion gộp chuẩn hoá theo hàng |",
          "| `round_time.png` | phút train / eval mỗi round trên 2×T4 |", ""]
    (out / "README.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="tag=run_dir, e.g. 20c=.../pfedes_20c_cos")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e5e5e0", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#c3c2b7", "figure.dpi": 150, "axes.axisbelow": True})

    summary, hists, spread = {}, {}, {}
    for spec in a.runs:
        tag, d = spec.split("=", 1)
        d = Path(d)
        rounds, rows, mf = load_run(d)
        hist = write_history(d, a.out, tag)
        hists[tag] = hist
        last = rows[-1]
        cfg = mf["cfg"]
        N = cfg["n_clients"]
        classes = mf.get("class_names") or [c["class"] for c in last["clients"][0]["per_class"]]

        # per client, last round
        with open(a.out / f"per_client_final_{tag}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["cid", *METRICS])
            for c in last["clients"]:
                w.writerow([c["cid"], *[f"{c[m]:.6f}" for m in METRICS]])
        f1s = np.array([c["f1_macro"] for c in last["clients"]])
        spread[tag] = f1s

        # per class, last round: mean over clients + pooled confusion
        cm = np.load(d / "confusion" / f"round_{last['round']:03d}.npy")   # (N, C, C)
        assert cm.shape[0] == N
        pooled = cm.sum(0)
        pp, pr, pf = prf_from_cm(pooled)
        pc = {i: [] for i in range(len(classes))}
        for c in last["clients"]:
            for e in c["per_class"]:
                pc[e["idx"]].append((e["precision"], e["recall"], e["f1"]))
        support = cm[0].sum(1)
        with open(a.out / f"per_class_final_{tag}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "class", "support", "share", "precision_mean", "recall_mean",
                        "f1_mean", "f1_std", "f1_min", "f1_max", "precision_pooled",
                        "recall_pooled", "f1_pooled"])
            for i, name in enumerate(classes):
                arr = np.array(pc[i])
                w.writerow([i, name, int(support[i]), f"{support[i]/support.sum():.5f}",
                            f"{arr[:,0].mean():.6f}", f"{arr[:,1].mean():.6f}",
                            f"{arr[:,2].mean():.6f}", f"{arr[:,2].std():.6f}",
                            f"{arr[:,2].min():.6f}", f"{arr[:,2].max():.6f}",
                            f"{pp[i]:.6f}", f"{pr[i]:.6f}", f"{pf[i]:.6f}"])
        np.savetxt(a.out / f"confusion_final_{tag}.csv", pooled, fmt="%d", delimiter=",",
                   header=",".join(classes), comments="")

        # timing + sessions
        train = np.array([float(r["train_sec"]) for r in hist])
        ev = np.array([float(r["eval_sec"]) for r in hist])
        sec = np.array([float(r["seconds"]) for r in hist])
        sess = d / "logs" / "sessions.json"
        sessions = json.loads(sess.read_text()) if sess.is_file() else None
        f1_mean = np.array([float(r["f1_macro"]) for r in hist])
        peak = int(np.argmax(f1_mean))
        summary[tag] = {
            "run_dir": str(d), "run_name": mf.get("run_name", d.name),
            "fingerprint": mf.get("fingerprint"),
            "n_clients": N, "rounds_completed": len(rounds), "rounds_planned": cfg["rounds"],
            "complete": len(rounds) == cfg["rounds"],
            "config": {k: cfg[k] for k in ("lr", "lr_schedule", "lr_min", "weight_decay", "mu",
                                            "local_epochs", "proxy_epochs", "participation",
                                            "rounds", "batch", "eval_batch", "seed", "clip")},
            "n_test": cfg.get("n_test"),
            "final_round": last["round"],
            "final": {m: {"mean": last[m], "std": last[m + "_std"], "min": last[m + "_min"],
                          "max": last[m + "_max"]} for m in METRICS},
            "round1": {m: rows[0][m] for m in METRICS},
            "peak_f1_macro_posthoc": {"round": int(hist[peak]["round"]),
                                      "f1_macro": float(f1_mean[peak]),
                                      "accuracy": float(hist[peak]["accuracy"])},
            "drop_from_peak_pct": float((f1_mean[peak] - f1_mean[-1]) / f1_mean[peak] * 100),
            "timing": {"round_sec_median": float(np.median(sec)),
                       "train_sec_median": float(np.median(train)),
                       "eval_sec_median": float(np.median(ev)),
                       "round_sec_max": float(sec.max()),
                       "total_round_hours": float(sec.sum() / 3600),
                       "total_train_hours": float(train.sum() / 3600),
                       "total_eval_hours": float(ev.sum() / 3600)},
            "backend": sorted({r["backend"] for r in hist}),
            "cache_mismatch_total": int(sum(int(r["cache_mismatch"]) for r in hist)),
            "evaluated_always_N": all(int(r["evaluated"]) == N for r in hist),
            "sessions": sessions,
            "worst_client_final": {"cid": int(np.argmin(f1s)), "f1_macro": float(f1s.min())},
            "best_client_final": {"cid": int(np.argmax(f1s)), "f1_macro": float(f1s.max())},
        }

        # figure: per-class f1 (mean over clients), ordered by support
        order = np.argsort(-support)
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        arr = np.array([np.mean([x[2] for x in pc[i]]) for i in range(len(classes))])
        ax.bar(range(len(classes)), arr[order], color=SERIES[tag], width=0.7)
        ax.plot(range(len(classes)), pf[order], "o", color="#1a1a19", ms=4, label="pooled confusion")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([f"{classes[i]}\n{support[i]/support.sum()*100:.1f}%" for i in order],
                           rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1); ax.set_ylabel("F1 (round %d)" % last["round"])
        ax.set_title(f"{tag}, round {last['round']}: per-class F1 — mean over {N} clients (bars), "
                     "pooled confusion (dots); ordered by test share", fontsize=8.5)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout(); fig.savefig(a.out / f"per_class_f1_{tag}.png"); plt.close(fig)

        # figure: pooled confusion, row-normalised
        rn = pooled / np.maximum(pooled.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6.5, 5.8))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
        im = ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=7); ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if rn[i, j] >= 0.05:
                    ax.text(j, i, f"{rn[i,j]:.2f}", ha="center", va="center", fontsize=5.5,
                            color="white" if rn[i, j] > 0.55 else "#1a1a19")
        fig.colorbar(im, ax=ax, fraction=0.04, label="row share (recall)")
        ax.set_title(f"{tag}: confusion at round {last['round']}, summed over {N} clients, "
                     "row-normalised")
        fig.tight_layout(); fig.savefig(a.out / f"confusion_{tag}.png"); plt.close(fig)

    # convergence: f1_macro and accuracy, one panel each (no dual axis), std band
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for tag, hist in hists.items():
        r = np.array([int(x["round"]) for x in hist])
        for ax, m in zip(axes[:2], ("f1_macro", "accuracy")):
            mu = np.array([float(x[m]) for x in hist]); sd = np.array([float(x[m + "_std"]) for x in hist])
            ax.fill_between(r, mu - sd, mu + sd, color=SERIES[tag], alpha=0.12, lw=0)
            ax.plot(r, mu, color=SERIES[tag], lw=2, label=f"{tag} ({len(r)} round)")
            ax.text(r[-1] + 0.5, mu[-1], f"{mu[-1]:.3f}", color="#1a1a19", fontsize=7, va="center")
    longest = max(hists.values(), key=len)
    axes[2].plot([int(x["round"]) for x in longest], [float(x["lr"]) for x in longest],
                 color="#1a1a19", lw=2)
    axes[0].set_title("f1_macro, mean ± std over clients"); axes[1].set_title("accuracy, mean ± std")
    axes[2].set_title("learning rate, identical for all K (cosine 1e-3 → 1e-5, T = 50)"); axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("round"); ax.set_xlim(0, 51)
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(a.out / "convergence.png"); plt.close(fig)

    # client spread at the final round
    fig, ax = plt.subplots(figsize=(6, 3.2))
    data = [spread[t] for t in spread]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
    for patch, t in zip(bp["boxes"], spread):
        patch.set_facecolor(SERIES[t]); patch.set_alpha(0.35); patch.set_edgecolor(SERIES[t])
    for i, t in enumerate(spread, 1):
        x = np.random.default_rng(0).normal(i, 0.05, len(spread[t]))
        ax.plot(x, spread[t], "o", ms=3, color=SERIES[t], alpha=0.8, mec="white", mew=0.5)
    ax.set_xticks(range(1, len(spread) + 1))
    ax.set_xticklabels([f"{t}\nround {summary[t]['final_round']}" for t in spread])
    ax.set_ylabel("f1_macro per client"); ax.set_title("Per-client f1_macro at the last completed round")
    fig.tight_layout(); fig.savefig(a.out / "client_spread.png"); plt.close(fig)

    # round time
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for tag, hist in hists.items():
        r = [int(x["round"]) for x in hist]
        axes[0].plot(r, [float(x["train_sec"]) / 60 for x in hist], color=SERIES[tag], lw=2, label=tag)
        axes[1].plot(r, [float(x["eval_sec"]) / 60 for x in hist], color=SERIES[tag], lw=2)
    axes[0].set_title("train minutes per round (2×T4)"); axes[1].set_title("eval minutes per round")
    for ax in axes:
        ax.set_xlabel("round"); ax.set_xlim(0, 51)
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(a.out / "round_time.png"); plt.close(fig)

    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_readme(a.out, summary)
    for t, s in summary.items():
        print(f"{t}: rounds {s['rounds_completed']}/{s['rounds_planned']} "
              f"f1_macro {s['final']['f1_macro']['mean']:.4f} (r1 {s['round1']['f1_macro']:.4f}, "
              f"peak {s['peak_f1_macro_posthoc']['f1_macro']:.4f}@r{s['peak_f1_macro_posthoc']['round']}) "
              f"acc {s['final']['accuracy']['mean']:.4f} round {s['timing']['round_sec_median']:.0f}s")


if __name__ == "__main__":
    main()

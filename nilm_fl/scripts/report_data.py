#!/usr/bin/env python
"""Numbers and figures for the NILM-FL (federated mutual learning) report, rebuilt from pulled
run artifacts only.

    python scripts/report_data.py --out papers/nilm-li-2024/report_data \
        20c=papers/nilm-li-2024/runs/merged/nilm_20c \
        50c=papers/nilm-li-2024/runs/merged/nilm_50c \
        100c=papers/nilm-li-2024/runs/merged/nilm_100c

Per scenario (K = 20/50/100 clients) it writes, under --out:
  history_Kc.csv          every round, every column of the run's history.csv (mean/std/min/max
                          of the 10 metrics over the N personalized models w_s^k, the 10
                          global_* metrics of the server proxy w̄_r, loss terms, lr, timing, skips)
  rounds_Kc.md            the same as a markdown table: all 10 metrics (mean over clients) for
                          EVERY round plus proxy f1_macro/accuracy — the reporting contract
  per_client_final_Kc.csv the 10 metrics of every client at the last completed round
  per_class_final_Kc.csv  per class at the last round: support, P/R/F1 as the mean over clients,
                          on the pooled (summed) client confusion, and of the proxy w̄_r
  confusion_final_Kc.csv  pooled confusion matrix (sum over clients) of the last round
  confusion_global_Kc.csv confusion matrix of the proxy w̄_r at the last round
  summary.json            headline numbers of all scenarios: final round, post-hoc peak, proxy,
                          timing, sessions, config read from reports/manifest.json
and the figures convergence.png, mutual_loss.png, client_spread.png, per_class_f1_Kc.png,
confusion_Kc.png, round_time.png. Rounds are read from metrics/round_NNN.json; nothing comes
from W&B.
"""
import argparse, csv, json
from pathlib import Path

import numpy as np

METRICS = ("accuracy", "precision_macro", "precision_micro", "precision_weighted",
           "recall_macro", "recall_micro", "recall_weighted", "f1_macro", "f1_micro",
           "f1_weighted")
LOSS_COLS = ("ce_s", "ce_r", "kl_s", "kl_r", "loss_s", "loss_r", "gnorm_s", "gnorm_r")
# dataviz reference palette, categorical slots 1-3 in fixed order; sequential = blue ramp
SERIES = {"20c": "#2a78d6", "50c": "#eb6834", "100c": "#1baf7a"}
BLUES = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load_run(d):
    rounds = sorted(int(p.stem.split("_")[1]) for p in (d / "metrics").glob("round_*.json"))
    rows = [json.loads((d / "metrics" / f"round_{r:03d}.json").read_text()) for r in rounds]
    assert [r["round"] for r in rows] == rounds
    assert rounds == list(range(1, len(rounds) + 1)), "rounds must be contiguous from 1 (merge first)"
    mf = json.loads((d / "reports" / "manifest.json").read_text())
    return rounds, rows, mf


def prf_from_cm(cm):
    tp = np.diag(cm).astype(float)
    p = tp / np.maximum(cm.sum(0), 1)
    r = tp / np.maximum(cm.sum(1), 1)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return p, r, f


def write_history(d, out, tag, rows):
    src = d / "history.csv"
    hist = list(csv.DictReader(open(src)))
    hist = sorted({int(r["round"]): r for r in hist}.values(), key=lambda r: int(r["round"]))
    # history.csv is derived; the metrics JSON is the source — they must agree before display
    for h, m in zip(hist, rows):
        assert int(h["round"]) == m["round"]
        for k in METRICS:
            assert abs(float(h[k]) - m[k]) < 1e-9 and abs(float(h["global_" + k]) - m["global_" + k]) < 1e-9, (tag, h["round"], k)
    with open(out / f"history_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader(); w.writerows(hist)
    lines = ["| round | lr | " + " | ".join(METRICS) + " | f1_macro std | min | max | proxy f1_macro | proxy acc | skip | train s | eval s |",
             "|---:|---:|" + "---:|" * len(METRICS) + "---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in hist:
        lines.append(f"| {r['round']} | {float(r['lr']):.2e} | "
                     + " | ".join(f"{float(r[m]):.4f}" for m in METRICS)
                     + f" | {float(r['f1_macro_std']):.4f} | {float(r['f1_macro_min']):.4f} | "
                       f"{float(r['f1_macro_max']):.4f} | {float(r['global_f1_macro']):.4f} | "
                       f"{float(r['global_accuracy']):.4f} | {int(r['skipped'])} | "
                       f"{float(r['train_sec']):.0f} | {float(r['eval_sec']):.0f} |")
    (out / f"rounds_{tag}.md").write_text(
        f"# {tag}: 10 metrics per round (mean over all N personalized models w_s^k, full "
        "10 761 343-row test set); proxy = server model w̄_r on the same test\n\n"
        "accuracy = precision_micro = recall_micro = recall_weighted = f1_micro (single-label "
        "multiclass identity). skip = AMP steps skipped by the GradScaler, summed over clients.\n\n"
        + "\n".join(lines) + "\n")
    return hist


def write_readme(out, summary):
    """Headline tables (Vietnamese) regenerated from summary.json — the report copies from here."""
    tags = list(summary)
    L = ["# Số liệu cho REPORT.md — Lightweight-FL NILM (học tương hỗ liên bang) trên VeReMi / DAGSNet", "",
         "Sinh tự động bởi `scripts/report_data.py` từ artifact đã kéo về, ghép phiên và verify local; "
         "không có số nào lấy từ W&B. Metric **cá nhân hoá** là trung bình không trọng số trên N model "
         "w_s^k, mỗi model đo trên **toàn bộ 10 761 343 dòng test toàn cục**; metric **proxy** là của "
         "một model w̄_r (server) trên cùng tập test. Caveat bắt buộc: `CONTEXT.md` §7.", ""]
    L += ["## 1. Tình trạng run", "", "| kịch bản | client | round | trạng thái | phiên Kaggle | round-time trung vị | tổng giờ round |",
          "|---|---:|---:|---|---|---:|---:|"]
    for t in tags:
        s = summary[t]
        sess = s["sessions"]
        ns = f"{len(sess)} phiên ({', '.join(str(len(x['rounds_produced'])) + ' round' for x in sess)})" if sess else "một phiên"
        L.append(f"| {t} | {s['n_clients']} | **{s['rounds_completed']}/{s['rounds_planned']}** | "
                 f"{'HOÀN TẤT' if s['complete'] else 'CHƯA ĐỦ — số liệu tạm'} | {ns} | "
                 f"{s['timing']['round_sec_median']/60:.1f} phút (train {s['timing']['train_sec_median']/60:.1f} + eval {s['timing']['eval_sec_median']/60:.1f}) | "
                 f"{s['timing']['total_round_hours']:.1f} h |")
    L += ["", "## 2. Round cuối — đủ 10 metric (mean trên client; std/min/max của f1_macro; proxy w̄_r)", "",
          "accuracy = precision_micro = recall_micro = recall_weighted = f1_micro (đồng nhất thức đa lớp đơn nhãn).", "",
          "| metric | " + " | ".join(f"{t} w_s (r{summary[t]['final_round']}) | {t} w̄_r" for t in tags) + " |",
          "|---|" + "---:|---:|" * len(tags)]
    for m in METRICS:
        L.append(f"| {m} | " + " | ".join(f"{summary[t]['final'][m]['mean']:.4f} | {summary[t]['final_global'][m]:.4f}" for t in tags) + " |")
    L.append("| f1_macro std / min / max | " + " | ".join(
        f"{summary[t]['final']['f1_macro']['std']:.4f} / {summary[t]['final']['f1_macro']['min']:.4f} / {summary[t]['final']['f1_macro']['max']:.4f} | —" for t in tags) + " |")
    L.append("| client yếu nhất / mạnh nhất (f1_macro) | " + " | ".join(
        f"#{summary[t]['worst_client_final']['cid']} {summary[t]['worst_client_final']['f1_macro']:.3f} / #{summary[t]['best_client_final']['cid']} {summary[t]['best_client_final']['f1_macro']:.3f} | —" for t in tags) + " |")
    L += ["", "## 3. Hình dạng đường cong (round 1 → đỉnh hậu kiểm → cuối)", "",
          "Đỉnh là quan sát *sau khi chạy*, không phải checkpoint được chọn; số công bố là round cuối.", "",
          "| | " + " | ".join(tags) + " |", "|---|" + "---:|" * len(tags)]
    L.append("| f1_macro w_s round 1 | " + " | ".join(f"{summary[t]['round1']['f1_macro']:.4f}" for t in tags) + " |")
    L.append("| f1_macro w_s đỉnh (round) | " + " | ".join(f"{summary[t]['peak_f1_macro_posthoc']['f1_macro']:.4f} (r{summary[t]['peak_f1_macro_posthoc']['round']})" for t in tags) + " |")
    L.append("| f1_macro w_s cuối | " + " | ".join(f"{summary[t]['final']['f1_macro']['mean']:.4f}" for t in tags) + " |")
    L.append("| chênh so đỉnh | " + " | ".join(f"−{summary[t]['drop_from_peak_pct']:.1f} %" for t in tags) + " |")
    L.append("| accuracy w_s round 1 → cuối | " + " | ".join(f"{summary[t]['round1']['accuracy']:.4f} → {summary[t]['final']['accuracy']['mean']:.4f}" for t in tags) + " |")
    L.append("| f1_macro w̄_r round 1 → 2 → cuối | " + " | ".join(f"{summary[t]['global_round1']['f1_macro']:.4f} → {summary[t]['global_round2']['f1_macro']:.4f} → {summary[t]['final_global']['f1_macro']:.4f}" for t in tags) + " |")
    L.append("| accuracy w̄_r round 1 → cuối | " + " | ".join(f"{summary[t]['global_round1']['accuracy']:.4f} → {summary[t]['final_global']['accuracy']:.4f}" for t in tags) + " |")
    L += ["", "## 4. Thành phần loss (trung bình client, các bước áp dụng) — round 1 / 2 / cuối", "",
          "| | " + " | ".join(tags) + " |", "|---|" + "---:|" * len(tags)]
    for k in ("ce_s", "ce_r", "kl_s", "kl_r", "gnorm_s", "gnorm_r"):
        L.append(f"| {k} | " + " | ".join(f"{summary[t]['loss']['round1'][k]:.3f} / {summary[t]['loss']['round2'][k]:.3f} / {summary[t]['loss']['final'][k]:.3f}" for t in tags) + " |")
    L += ["", "## 5. Kiểm soát chất lượng pipeline", "", "| | " + " | ".join(tags) + " |", "|---|" + "---:|" * len(tags)]
    L.append("| backend train | " + " | ".join("/".join(summary[t]["backend"]) for t in tags) + " |")
    L.append("| evaluated = N mọi round | " + " | ".join("có" if summary[t]["evaluated_always_N"] else "KHÔNG" for t in tags) + " |")
    L.append("| bước/round, skip AMP tổng (max một round) | " + " | ".join(f"{summary[t]['steps_per_round']}, {summary[t]['skipped_total']} ({summary[t]['skipped_max_round']})" for t in tags) + " |")
    L.append("| VRAM train max (GiB) | " + " | ".join(f"{summary[t]['vram_train_max_gb']:.2f}" for t in tags) + " |")
    L.append("| fingerprint | " + " | ".join(f"`{summary[t]['fingerprint']}`" for t in tags) + " |")
    L.append("| torch / CUDA | " + " | ".join(f"{summary[t]['torch']} / {summary[t]['cuda']}" for t in tags) + " |")
    L += ["", "## 6. File", "",
          "| file | nội dung |", "|---|---|",
          "| `rounds_Kc.md`, `history_Kc.csv` | **mọi round**, đủ 10 metric (mean) + std/min/max, proxy, skip, lr, train/eval giây — bảng bắt buộc của report |",
          "| `per_client_final_Kc.csv` | 10 metric của từng client ở round cuối |",
          "| `per_class_final_Kc.csv` | mỗi lớp ở round cuối: support, P/R/F1 trung bình trên client, trên confusion gộp, và của w̄_r |",
          "| `confusion_final_Kc.csv`, `confusion_global_Kc.csv` | confusion gộp (tổng N client) và của w̄_r, round cuối, 16×16 |",
          "| `summary.json` | mọi số ở trên dạng máy đọc + cấu hình từ `reports/manifest.json` + provenance phiên |",
          "| `convergence.png` | f1_macro / accuracy (mean ± std, nét liền) và proxy w̄_r (nét đứt) theo round, cả ba K; lịch LR |",
          "| `mutual_loss.png` | ce_s, ce_r, kl_s, kl_r trung bình client theo round |",
          "| `client_spread.png` | phân bố f1_macro theo client ở round cuối |",
          "| `per_class_f1_Kc.png`, `confusion_Kc.png` | F1 từng lớp (theo tỉ lệ test) và confusion gộp chuẩn hoá theo hàng |",
          "| `round_time.png` | phút train / eval mỗi round trên 2×T4 |", ""]
    (out / "README.md").write_text("\n".join(L))


def write_appendix(report, out, summary):
    """Replace the marked appendix block of REPORT.md with one table per scenario (all rounds)."""
    text = report.read_text()
    b, e = "<!-- APPENDIX:BEGIN", "<!-- APPENDIX:END -->"
    i, j = text.index(b), text.index(e)
    i = text.index("\n", i) + 1                      # keep the BEGIN marker line itself
    parts = []
    for n, (tag, s) in enumerate(summary.items()):
        state = ("chính thức" if s["complete"] else
                 f"TẠM — {s['rounds_completed']}/{s['rounds_planned']} round, cập nhật khi đủ 50")
        table = (out / f"rounds_{tag}.md").read_text().split("\n\n", 2)[2].rstrip()
        parts.append(f"\n### {'ABC'[n]}. {s['n_clients']} client — {s['rounds_completed']}/"
                     f"{s['rounds_planned']} round ({state})\n\n{table}\n")
    report.write_text(text[:i] + "".join(parts) + text[j:])
    print(f"appendix -> {report}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="tag=run_dir, e.g. 20c=.../merged/nilm_20c")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None,
                    help="REPORT.md whose <!-- APPENDIX:BEGIN --> … <!-- APPENDIX:END --> block is "
                         "replaced with the per-round tables of every scenario")
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
        hist = write_history(d, a.out, tag, rows)
        hists[tag] = hist
        last = rows[-1]
        cfg = mf["cfg"]
        N = cfg["n_clients"]
        classes = mf["class_names"]

        # per client, last round
        with open(a.out / f"per_client_final_{tag}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["cid", *METRICS])
            for c in last["clients"]:
                w.writerow([c["cid"], *[f"{c[m]:.6f}" for m in METRICS]])
        f1s = np.array([c["f1_macro"] for c in last["clients"]])
        spread[tag] = f1s

        # per class, last round: mean over clients + pooled confusion + proxy
        cm = np.load(d / "confusion" / f"round_{last['round']:03d}.npy")   # (N, C, C)
        assert cm.shape[0] == N
        gcm = np.load(d / "confusion" / f"global_{last['round']:03d}.npy")  # (C, C)
        assert gcm.sum() == cfg["n_test"] and (cm.sum((1, 2)) == cfg["n_test"]).all()
        pooled = cm.sum(0)
        pp, pr, pf = prf_from_cm(pooled)
        gp, gr, gf = prf_from_cm(gcm)
        pc = {i: [] for i in range(len(classes))}
        for c in last["clients"]:
            for e in c["per_class"]:
                pc[e["idx"]].append((e["precision"], e["recall"], e["f1"]))
        for e in last["global"]["per_class"]:        # metrics JSON and confusion must agree
            assert abs(e["f1"] - gf[e["idx"]]) < 1e-9, (tag, e)
        support = cm[0].sum(1)
        with open(a.out / f"per_class_final_{tag}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "class", "support", "share", "precision_mean", "recall_mean",
                        "f1_mean", "f1_std", "f1_min", "f1_max", "precision_pooled",
                        "recall_pooled", "f1_pooled", "precision_proxy", "recall_proxy", "f1_proxy"])
            for i, name in enumerate(classes):
                arr = np.array(pc[i])
                w.writerow([i, name, int(support[i]), f"{support[i]/support.sum():.5f}",
                            f"{arr[:,0].mean():.6f}", f"{arr[:,1].mean():.6f}",
                            f"{arr[:,2].mean():.6f}", f"{arr[:,2].std():.6f}",
                            f"{arr[:,2].min():.6f}", f"{arr[:,2].max():.6f}",
                            f"{pp[i]:.6f}", f"{pr[i]:.6f}", f"{pf[i]:.6f}",
                            f"{gp[i]:.6f}", f"{gr[i]:.6f}", f"{gf[i]:.6f}"])
        np.savetxt(a.out / f"confusion_final_{tag}.csv", pooled, fmt="%d", delimiter=",",
                   header=",".join(classes), comments="")
        np.savetxt(a.out / f"confusion_global_{tag}.csv", gcm, fmt="%d", delimiter=",",
                   header=",".join(classes), comments="")

        # timing + sessions
        train = np.array([float(r["train_sec"]) for r in hist])
        ev = np.array([float(r["eval_sec"]) for r in hist])
        sec = np.array([float(r["seconds"]) for r in hist])
        skipped = np.array([int(r["skipped"]) for r in hist])
        sess = d / "logs" / "sessions.json"
        sessions = json.loads(sess.read_text()) if sess.is_file() else None
        f1_mean = np.array([float(r["f1_macro"]) for r in hist])
        gf1 = np.array([float(r["global_f1_macro"]) for r in hist])
        peak = int(np.argmax(f1_mean)); gpeak = int(np.argmax(gf1))
        loss_at = lambda m: {k: m[k + "_client_mean"] for k in LOSS_COLS}
        summary[tag] = {
            "run_dir": str(d), "run_name": cfg["run_name"],
            "fingerprint": mf["fingerprint"], "data_id": mf["data_id"], "content_id": mf["content_id"],
            "torch": mf["torch"], "cuda": mf["cuda"], "n_params": mf["n_params"],
            "n_clients": N, "rounds_completed": len(rounds), "rounds_planned": cfg["rounds"],
            "complete": len(rounds) == cfg["rounds"],
            "config": {k: cfg[k] for k in ("lr", "lr_schedule", "lr_min", "weight_decay",
                                            "local_epochs", "rounds", "batch", "eval_batch",
                                            "seed", "clip", "max_skips_per_client", "preds_rounds")},
            "n_test": cfg["n_test"], "n_train": mf["n_train"],
            "final_round": last["round"],
            "final": {m: {"mean": last[m], "std": last[m + "_std"], "min": last[m + "_min"],
                          "max": last[m + "_max"]} for m in METRICS},
            "final_global": {m: last["global_" + m] for m in METRICS},
            "round1": {m: rows[0][m] for m in METRICS},
            "global_round1": {m: rows[0]["global_" + m] for m in METRICS},
            "global_round2": {m: rows[1]["global_" + m] for m in METRICS} if len(rows) > 1 else None,
            "peak_f1_macro_posthoc": {"round": int(hist[peak]["round"]),
                                      "f1_macro": float(f1_mean[peak]),
                                      "accuracy": float(hist[peak]["accuracy"])},
            "peak_global_f1_macro_posthoc": {"round": int(hist[gpeak]["round"]),
                                             "f1_macro": float(gf1[gpeak])},
            "drop_from_peak_pct": float((f1_mean[peak] - f1_mean[-1]) / f1_mean[peak] * 100),
            "loss": {"round1": loss_at(rows[0]), "round2": loss_at(rows[1]) if len(rows) > 1 else None,
                     "final": loss_at(last)},
            "timing": {"round_sec_median": float(np.median(sec)),
                       "train_sec_median": float(np.median(train)),
                       "eval_sec_median": float(np.median(ev)),
                       "round_sec_max": float(sec.max()),
                       "total_round_hours": float(sec.sum() / 3600),
                       "total_train_hours": float(train.sum() / 3600),
                       "total_eval_hours": float(ev.sum() / 3600)},
            "backend": sorted({r["backend"] for r in hist}),
            "evaluated_always_N": all(int(r["evaluated"]) == N for r in hist),
            "steps_per_round": int(hist[0]["steps"]),
            "skipped_total": int(skipped.sum()), "skipped_max_round": int(skipped.max()),
            "vram_train_max_gb": max(float(r["vram_train_gb"]) for r in hist),
            "sessions": sessions,
            "worst_client_final": {"cid": int(np.argmin(f1s)), "f1_macro": float(f1s.min())},
            "best_client_final": {"cid": int(np.argmax(f1s)), "f1_macro": float(f1s.max())},
        }

        # figure: per-class f1 (mean over clients), ordered by support; pooled + proxy as markers
        order = np.argsort(-support)
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        arr = np.array([np.mean([x[2] for x in pc[i]]) for i in range(len(classes))])
        ax.bar(range(len(classes)), arr[order], color=SERIES[tag], width=0.7, label=f"mean over {N} w_s^k")
        ax.plot(range(len(classes)), pf[order], "o", color="#1a1a19", ms=4, label="pooled confusion")
        ax.plot(range(len(classes)), gf[order], "D", color="#8a3ffc", ms=4, label="proxy w̄_r")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([f"{classes[i]}\n{support[i]/support.sum()*100:.1f}%" for i in order],
                           rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1); ax.set_ylabel("F1 (round %d)" % last["round"])
        ax.set_title(f"{tag}, round {last['round']}: per-class F1 — ordered by test share", fontsize=8.5)
        ax.legend(frameon=False, loc="upper right", fontsize=7)
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

    # convergence: f1_macro and accuracy (personalized mean ± std, proxy dashed), LR
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for tag, hist in hists.items():
        r = np.array([int(x["round"]) for x in hist])
        for ax, m in zip(axes[:2], ("f1_macro", "accuracy")):
            mu = np.array([float(x[m]) for x in hist]); sd = np.array([float(x[m + "_std"]) for x in hist])
            g = np.array([float(x["global_" + m]) for x in hist])
            ax.fill_between(r, mu - sd, mu + sd, color=SERIES[tag], alpha=0.12, lw=0)
            ax.plot(r, mu, color=SERIES[tag], lw=2, label=f"{tag} w_s ({len(r)} round)")
            ax.plot(r, g, color=SERIES[tag], lw=1.3, ls="--", label=f"{tag} w̄_r")
            ax.text(r[-1] + 0.5, mu[-1], f"{mu[-1]:.3f}", color="#1a1a19", fontsize=7, va="center")
            ax.text(r[-1] + 0.5, g[-1], f"{g[-1]:.3f}", color="#5a5a58", fontsize=7, va="center")
    longest = max(hists.values(), key=len)
    axes[2].plot([int(x["round"]) for x in longest], [float(x["lr"]) for x in longest],
                 color="#1a1a19", lw=2)
    axes[0].set_title("f1_macro — w_s mean ± std (solid), w̄_r (dashed)", fontsize=8.5)
    axes[1].set_title("accuracy — w_s mean ± std (solid), w̄_r (dashed)", fontsize=8.5)
    axes[2].set_title("learning rate (cosine 1e-3 → 1e-5, T = 50)", fontsize=8.5); axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("round"); ax.set_xlim(0, 51)
    axes[0].set_ylim(0, 1); axes[1].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout(); fig.savefig(a.out / "convergence.png"); plt.close(fig)

    # mutual-learning loss terms per round
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for tag, hist in hists.items():
        r = [int(x["round"]) for x in hist]
        axes[0].plot(r, [float(x["ce_s_client_mean"]) for x in hist], color=SERIES[tag], lw=2, label=f"{tag} ce_s")
        axes[0].plot(r, [float(x["ce_r_client_mean"]) for x in hist], color=SERIES[tag], lw=1.3, ls="--", label=f"{tag} ce_r")
        axes[1].plot(r, [float(x["kl_s_client_mean"]) for x in hist], color=SERIES[tag], lw=2, label=f"{tag} kl_s")
        axes[1].plot(r, [float(x["kl_r_client_mean"]) for x in hist], color=SERIES[tag], lw=1.3, ls="--", label=f"{tag} kl_r")
    axes[0].set_title("label loss ℓ_s (solid) / ℓ_r (dashed), client mean")
    axes[1].set_title("distillation KL(p_r‖p_s) (solid) / KL(p_s‖p_r) (dashed)")
    for ax in axes:
        ax.set_xlabel("round"); ax.set_xlim(0, 51); ax.legend(frameon=False, fontsize=6.5, ncol=3)
    fig.tight_layout(); fig.savefig(a.out / "mutual_loss.png"); plt.close(fig)

    # client spread at the final round
    fig, ax = plt.subplots(figsize=(6, 3.2))
    data = [spread[t] for t in spread]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
    for patch, t in zip(bp["boxes"], spread):
        patch.set_facecolor(SERIES[t]); patch.set_alpha(0.35); patch.set_edgecolor(SERIES[t])
    for i, t in enumerate(spread, 1):
        x = np.random.default_rng(0).normal(i, 0.05, len(spread[t]))
        ax.plot(x, spread[t], "o", ms=3, color=SERIES[t], alpha=0.8, mec="white", mew=0.5)
        ax.plot([i - 0.3, i + 0.3], [summary[t]["final_global"]["f1_macro"]] * 2, color="#8a3ffc", lw=1.5)
    ax.set_xticks(range(1, len(spread) + 1))
    ax.set_xticklabels([f"{t}\nround {summary[t]['final_round']}" for t in spread])
    ax.set_ylabel("f1_macro per client"); ax.set_title("Per-client f1_macro at the last completed round (purple bar = proxy w̄_r)")
    fig.tight_layout(); fig.savefig(a.out / "client_spread.png"); plt.close(fig)

    # round time
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for tag, hist in hists.items():
        r = [int(x["round"]) for x in hist]
        axes[0].plot(r, [float(x["train_sec"]) / 60 for x in hist], color=SERIES[tag], lw=2, label=tag)
        axes[1].plot(r, [float(x["eval_sec"]) / 60 for x in hist], color=SERIES[tag], lw=2)
    axes[0].set_title("train minutes per round (2×T4)"); axes[1].set_title("eval minutes per round (N + 1 models)")
    for ax in axes:
        ax.set_xlabel("round"); ax.set_xlim(0, 51)
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(a.out / "round_time.png"); plt.close(fig)

    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_readme(a.out, summary)
    if a.report:
        write_appendix(a.report, a.out, summary)
    for t, s in summary.items():
        print(f"{t}: rounds {s['rounds_completed']}/{s['rounds_planned']} "
              f"f1_macro {s['final']['f1_macro']['mean']:.4f} (r1 {s['round1']['f1_macro']:.4f}, "
              f"peak {s['peak_f1_macro_posthoc']['f1_macro']:.4f}@r{s['peak_f1_macro_posthoc']['round']}) "
              f"proxy {s['final_global']['f1_macro']:.4f} acc {s['final']['accuracy']['mean']:.4f} "
              f"round {s['timing']['round_sec_median']:.0f}s")


if __name__ == "__main__":
    main()

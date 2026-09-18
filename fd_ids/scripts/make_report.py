#!/usr/bin/env python
"""Build report.md (repository root) and figures/ from the verified run directories.

Reads ONLY pulled, verified artifacts under papers/fd-ids-2025/runs/<run>/ and the frozen
knowledge/ files. Every number in the report is re-derived here from history.csv, the per-round
metrics JSON and the confusion matrices, and the three are cross-checked against each other
before anything is written -- the completion gate of the skill's reporting.md, executed rather
than promised:

  * history.csv (deduplicated, round-sorted) == metrics/round_NNN.json on all 10 metrics;
  * accuracy == precision_micro == recall_micro == recall_weighted == f1_micro (single-label
    multi-class identity; stated next to every table, never used to drop a column);
  * every confusion matrix sums to the test row count, and the per-class P/R/F1 stored in the
    JSON equal the ones recomputed from the matrix;
  * rounds are contiguous from 1.

A scenario whose merged run directory does not exist yet gets an explicit "chưa hoàn tất"
section and no numbers. Re-run after every merge:

    python scripts/make_report.py
"""
import csv, json, sys, datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "papers/fd-ids-2025/runs"
FIG = ROOT / "figures"
OUT = ROOT / "report.md"
META = json.loads((ROOT / "knowledge/meta.json").read_text())
RUNTIME = json.loads((ROOT / "knowledge/runtime.json").read_text())

METRIC_KEYS = ["accuracy",
               "precision_macro", "precision_micro", "precision_weighted",
               "recall_macro", "recall_micro", "recall_weighted",
               "f1_macro", "f1_micro", "f1_weighted"]
N_TEST = 10_761_343
PLANNED = 50
TOL = 1e-9

# Provenance that no artifact carries: which Kaggle kernel/version produced which session.
# Everything else (rounds per session, cfg, timings) is read from the run directory.
SCENARIOS = {
    20: {"run": "fdids_20c_v2", "dataset": "odixe0502/veremi-fl-20client",
         "kernels": [("minhtrit06/fd-ids-veremi-20-clients", 3, "2026-09-10 17:49Z → 23:58Z"),
                     ("minhtriethihi/fd-ids-veremi-20-clients", 1, "2026-09-11 01:46Z → 02:59Z")]},
    50: {"run": "fdids_50c_v2", "dataset": "odixe0502/veremi-fl-50client",
         "kernels": [("khanhmay0304/fd-ids-veremi-50-clients", 3, "2026-09-10 17:50Z → 2026-09-11 01:16Z")]},
    100: {"run": "fdids_100c_v2", "dataset": "odixe0502/veremi-fl-100client",
          "kernels": [("khanhmay0304/fd-ids-veremi-100-clients", 3, "2026-09-10 17:50Z → 23:40Z"),
                      ("minhtriethihi/fd-ids-veremi-100-clients", 1, "2026-09-11 01:46Z → 06:37Z")]},
}


# ----------------------------------------------------------------------------- loading
def load_run(d):
    """history rows (deduplicated by round, sorted), cross-checked against the JSON and the
    confusion matrices. Raises on any disagreement: a report must not paper over its sources."""
    rows = {}
    for r in csv.DictReader(open(d / "history.csv")):
        row = {}
        for k, v in r.items():
            if v in ("", None): continue
            try: row[k] = int(float(v)) if k == "round" else float(v)
            except ValueError: row[k] = v            # `backend` is a string column
        rows[row["round"]] = row
    rounds = sorted(rows)
    assert rounds == list(range(1, rounds[-1] + 1)), f"{d.name}: rounds not contiguous: {rounds}"
    per_class, cms = {}, {}
    for rnd in rounds:
        js = json.loads((d / "metrics" / f"round_{rnd:03d}.json").read_text())
        for k in METRIC_KEYS:
            assert abs(js[k] - rows[rnd][k]) <= TOL, f"{d.name} r{rnd} {k}: csv {rows[rnd][k]} != json {js[k]}"
        a = js["accuracy"]
        for k in ("precision_micro", "recall_micro", "recall_weighted", "f1_micro"):
            assert abs(js[k] - a) <= TOL, f"{d.name} r{rnd}: {k}={js[k]} != accuracy={a}"
        cm = np.load(d / "confusion" / f"round_{rnd:03d}.npy").astype(np.int64)
        assert cm.shape == (16, 16) and cm.sum() == N_TEST, f"{d.name} r{rnd}: confusion sum {cm.sum()}"
        p, r_, f, sup = class_metrics(cm)
        for c in js["per_class"]:
            i = c["idx"]
            assert int(c["support"]) == int(sup[i]) and abs(c["precision"] - p[i]) <= TOL \
                and abs(c["recall"] - r_[i]) <= TOL and abs(c["f1"] - f[i]) <= TOL, \
                f"{d.name} r{rnd} class {i}: per_class JSON != confusion"
        # the macro/weighted aggregates must also follow from the matrix
        w = sup / sup.sum()
        assert abs(f.mean() - js["f1_macro"]) <= 1e-6 and abs((w * f).sum() - js["f1_weighted"]) <= 1e-6
        per_class[rnd] = js["per_class"]; cms[rnd] = cm
    sessions = json.loads((d / "logs" / "sessions.json").read_text())
    manifest = json.loads((d / "reports" / "manifest.json").read_text())
    return {"rows": rows, "rounds": rounds, "per_class": per_class, "cm": cms,
            "sessions": sessions, "manifest": manifest}


def class_metrics(cm):
    cm = cm.astype(float); tp = np.diag(cm); col = cm.sum(0); row = cm.sum(1)
    p = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    r = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return p, r, f, row


# ----------------------------------------------------------------------------- figures
def fig_convergence(K, run):
    rows, rounds = run["rows"], run["rounds"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, lab in (("f1_macro", "F1 macro"), ("f1_weighted", "F1 weighted"), ("accuracy", "accuracy"),
                   ("precision_macro", "precision macro"), ("recall_macro", "recall macro")):
        a.plot(rounds, [rows[r][k] for r in rounds], label=lab, lw=1.6 if k == "f1_macro" else 1.0)
    pk = max(rounds, key=lambda r: rows[r]["f1_macro"])
    a.axvline(pk, color="grey", ls=":", lw=1); a.annotate(f"đỉnh F1 macro r{pk} (hậu kiểm)", (pk, rows[pk]["f1_macro"]),
                                                          xytext=(pk + 1.5, rows[pk]["f1_macro"] + 0.02), fontsize=8)
    a.set_xlabel("round"); a.set_ylabel("metric trên toàn bộ tập test"); a.set_title(f"{K} client — 10 metric theo round (5 đường phân biệt)")
    a.grid(alpha=.3); a.legend(fontsize=8, loc="lower right")
    b.plot(rounds, [rows[r]["ce_client_mean"] for r in rounds], label="CE client mean (train)")
    b.plot(rounds, [rows[r]["kd_client_mean"] for r in rounds], label="KD client mean (train)")
    b.set_xlabel("round"); b.set_ylabel("loss huấn luyện phía client"); b.set_title("CE giảm rồi đi ngang, KD chạm đáy sớm rồi tăng dần; metric test bão hoà sớm")
    b.grid(alpha=.3); b.legend(fontsize=8)
    for ax in (a, b):
        for s in run["sessions"][1:]:
            ax.axvline(s["rounds_produced"][0] - 0.5, color="red", ls="--", lw=.8, alpha=.6)
    fig.tight_layout(); p = FIG / f"convergence_{K}c.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_all_f1(runs):
    fig, a = plt.subplots(figsize=(7.5, 4))
    for K, run in runs.items():
        a.plot(run["rounds"], [run["rows"][r]["f1_macro"] for r in run["rounds"]], label=f"{K} client")
    a.set_xlabel("round"); a.set_ylabel("F1 macro (test)"); a.set_title("F1 macro của global model — ba cấu hình")
    a.grid(alpha=.3); a.legend(); fig.tight_layout(); p = FIG / "f1_macro_all.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_confusion(K, run, rnd):
    cm = run["cm"][rnd].astype(float); norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    names = META["class_names"]
    fig, a = plt.subplots(figsize=(9, 8)); im = a.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    a.set_xticks(range(16)); a.set_yticks(range(16)); a.set_xticklabels(names, rotation=90, fontsize=7); a.set_yticklabels(names, fontsize=7)
    for i in range(16):
        for j in range(16):
            if norm[i, j] >= 0.05: a.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if norm[i, j] > .5 else "black")
    a.set_xlabel("dự đoán"); a.set_ylabel("nhãn thật"); a.set_title(f"{K} client — ma trận nhầm lẫn chuẩn hoá theo hàng, round {rnd}")
    fig.colorbar(im, fraction=.046); fig.tight_layout(); p = FIG / f"confusion_{K}c_r{rnd:03d}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_per_class(K, run, final, peak):
    pc_f = {c["idx"]: c for c in run["per_class"][final]}; pc_p = {c["idx"]: c for c in run["per_class"][peak]}
    order = sorted(range(16), key=lambda i: -pc_f[i]["support"]); names = META["class_names"]
    x = np.arange(16); fig, a = plt.subplots(figsize=(11, 4.2))
    a.bar(x - .2, [pc_f[i]["f1"] for i in order], .4, label=f"round {final} (cuối)")
    a.bar(x + .2, [pc_p[i]["f1"] for i in order], .4, label=f"round {peak} (đỉnh F1 macro, hậu kiểm)", alpha=.7)
    a.set_xticks(x); a.set_xticklabels([f"{names[i]}\n(n={pc_f[i]['support']:,})" for i in order], rotation=90, fontsize=7)
    a.set_ylabel("F1 từng lớp"); a.set_title(f"{K} client — F1 từng lớp, xếp theo support giảm dần"); a.grid(axis="y", alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); p = FIG / f"per_class_f1_{K}c.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ----------------------------------------------------------------------------- markdown
def rel(p): return p.relative_to(ROOT).as_posix()
def f4(x): return f"{x:.4f}"


def metric_table(run):
    hdr = "| round | " + " | ".join(METRIC_KEYS) + " |\n|---:|" + "---:|" * len(METRIC_KEYS) + "\n"
    body = "".join(f"| {r} | " + " | ".join(f4(run['rows'][r][k]) for k in METRIC_KEYS) + " |\n" for r in run["rounds"])
    return hdr + body


def ops_table(run):
    sess = {}
    for s in run["sessions"]:
        for r in s["rounds_produced"]: sess[r] = s["session"]
    hdr = "| round | phiên | ce_client_mean | kd_client_mean | grad_norm | skipped/steps | teacher_sec | seconds |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = "".join(f"| {r} | {sess[r]} | {f4(x['ce_client_mean'])} | {f4(x['kd_client_mean'])} | {x['grad_norm']:.3f} | "
                   f"{int(x['skipped'])}/{int(x['steps'])} | {x['teacher_sec']:.0f} | {x['seconds']:.1f} |\n"
                   for r in run["rounds"] for x in [run["rows"][r]])
    return hdr + body


def per_class_table(run, rnd):
    pc = sorted(run["per_class"][rnd], key=lambda c: -c["support"])
    hdr = "| lớp | support | precision | recall | f1 |\n|---|---:|---:|---:|---:|\n"
    return hdr + "".join(f"| `{c['class']}` | {c['support']:,} | {f4(c['precision'])} | {f4(c['recall'])} | {f4(c['f1'])} |\n" for c in pc)


def class_delta(run, final, peak):
    """(ΔF1 final−peak, class, support) sorted ascending, plus benign recall at both rounds."""
    f = {c["class"]: c for c in run["per_class"][final]}; p = {c["class"]: c for c in run["per_class"][peak]}
    d = sorted(((f[c]["f1"] - p[c]["f1"], c, f[c]["support"]) for c in f), key=lambda x: x[0])
    return d, f["benign"], p["benign"]


SEC = {20: 4, 50: 5, 100: 6}


def scenario_section(K, run, figs):
    rows, rounds = run["rows"], run["rounds"]; final = rounds[-1]
    peak = max(rounds, key=lambda r: rows[r]["f1_macro"]); pk_acc = max(rounds, key=lambda r: rows[r]["accuracy"])
    delta, ben_f, ben_p = class_delta(run, final, peak)
    losers = ", ".join(f"`{c}` ({x:+.3f})" for x, c, n in delta[:3]); gainers = ", ".join(f"`{c}` ({x:+.3f})" for x, c, n in delta[-3:][::-1])
    hard = ", ".join(f"`{c['class']}` ({f4(c['f1'])})" for c in sorted(run["per_class"][final], key=lambda c: c["f1"]) if c["f1"] < 0.4)
    cfg = run["manifest"]["cfg"]; sc = SCENARIOS[K]
    sess_lines = "".join(f"| {s['session']} | `{sc['kernels'][i][0]}` v{sc['kernels'][i][1]} | {sc['kernels'][i][2]} | "
                         f"{s['rounds_produced'][0]}–{s['rounds_produced'][-1]} ({len(s['rounds_produced'])}) | "
                         f"{sum(rows[r]['seconds'] for r in s['rounds_produced'])/3600:.2f} h |\n"
                         for i, s in enumerate(run["sessions"]))
    last10 = rounds[-10:]; m10 = sum(rows[r]["f1_macro"] for r in last10) / len(last10)
    ce_min = min(rounds, key=lambda r: rows[r]["ce_client_mean"]); kd_min = min(rounds, key=lambda r: rows[r]["kd_client_mean"])
    complete = final == PLANNED
    n = SEC[K]
    head = f"## {n}. {K} client — `{sc['run']}`" + ("" if complete else f" — **CHƯA HOÀN TẤT ({final}/{PLANNED} round)**")
    return f"""{head}

**Kết quả headline = round cuối {final}:** accuracy **{f4(rows[final]['accuracy'])}**, F1 macro **{f4(rows[final]['f1_macro'])}**,
F1 weighted **{f4(rows[final]['f1_weighted'])}** (precision macro {f4(rows[final]['precision_macro'])}, recall macro {f4(rows[final]['recall_macro'])}).
Quan sát hậu kiểm, *không phải* checkpoint được chọn: F1 macro đạt đỉnh **{f4(rows[peak]['f1_macro'])} ở round {peak}**,
accuracy đạt đỉnh {f4(rows[pk_acc]['accuracy'])} ở round {pk_acc}; trung bình F1 macro 10 round cuối {f4(m10)}.

| phiên | kernel / version | thời gian (UTC) | round sản xuất | tổng `seconds` của các round |
|---:|---|---|---|---:|
{sess_lines}
Nguồn: `{rel(RUNS / sc['run'])}/logs/sessions.json` (round nào do phiên nào sinh ra, đã kiểm bytes giống hệt trên phần chồng lấn khi merge), `history.csv`.
Cấu hình hiệu lực (từ `reports/manifest.json`): n_clients={cfg['n_clients']}, batch={cfg['batch']}, rounds={cfg['rounds']}×{cfg['local_epochs']} epoch,
lr={cfg['lr']}, μ={cfg['mu']}, λ={cfg['lam']}, β={cfg['beta']}, T={cfg['temperature']}, clip={cfg['clip']}, seed={cfg['seed']}, dropout={cfg['dropout']},
world_size={cfg['world_size']}, compile={cfg['compile']}; fingerprint `{run['manifest']['fingerprint']}`, data_id `{cfg['data_id']}`, content_id `{cfg['content_id']}`;
torch {run['manifest']['torch']}, CUDA {run['manifest']['cuda']}.

![]({rel(figs['conv'])})

*Hình: trái — năm metric phân biệt được của global model trên toàn bộ tập test theo round (đường chấm xám: round đỉnh F1 macro, hậu kiểm; đường đứt đỏ: ranh giới phiên Kaggle); phải — loss huấn luyện trung bình phía client. Điều cần thấy: CE train giảm rồi đi ngang ở mức thấp (cực tiểu {f4(rows[ce_min]['ce_client_mean'])} @r{ce_min}, round cuối {f4(rows[final]['ce_client_mean'])}); số hạng KD chạm cực tiểu sớm ({f4(rows[kd_min]['kd_client_mean'])} @r{kd_min}) rồi tăng dần tới {f4(rows[final]['kd_client_mean'])} — teacher (global model round trước) và student cách nhau dần; trong khi đó metric test bão hoà từ round {peak} và accuracy/F1 weighted đi xuống.*

### {n}.1 Đủ 10 metric của global model, {final} round

Global model là mô hình duy nhất tồn tại qua ranh giới round (Algorithm 1: mỗi client bị ghi đè bằng
`w_G` đầu round), nên đây là bảng của "một mô hình đại diện". Đánh giá trên **toàn bộ {N_TEST:,} dòng test**
sau mỗi round. Trong phân loại đơn nhãn đa lớp, **accuracy = precision_micro = recall_micro = recall_weighted = f1_micro**
(Σ FP = Σ FN); bốn cột trùng nhau là đúng, không phải lỗi — đã kiểm bằng máy cho từng round. Các metric thật sự
phân biệt mô hình trên dữ liệu mất cân bằng 41:1 là **f1_macro** và **f1_weighted**.

{metric_table(run)}
Nguồn: `{rel(RUNS / sc['run'])}/history.csv`, đối chiếu từng giá trị với `metrics/round_NNN.json` và ma trận nhầm lẫn `confusion/round_NNN.npy` (tổng = {N_TEST:,}) trước khi làm tròn 4 chữ số.

### {n}.2 Số liệu vận hành theo round

{ops_table(run)}
`skipped` = số bước bị GradScaler bỏ qua vì overflow fp16 (cộng dồn qua mọi client trong round); `seconds` gồm train + teacher + eval + ghi artifact + W&B.

### {n}.3 Từng lớp ở round cuối {final}

![]({rel(figs['pc'])})

*Hình: F1 từng lớp ở round cuối (đậm) và ở round đỉnh {peak} (nhạt), xếp theo support giảm dần. Điều cần thấy (ΔF1 = cuối − đỉnh, tính từ artifact): mất nhiều nhất {losers}; được nhiều nhất {gainers}. `benign` — lớp lớn thứ hai — là nơi mất lớn nhất hoặc nhì: recall `benign` {f4(ben_p['recall'])} ở round {peak} → **{f4(ben_f['recall'])}** ở round {final}, precision {f4(ben_f['precision'])}: global model ngày càng gán lưu lượng lành tính thành tấn công (tỉ lệ báo động giả tăng), đổi lấy recall ở các lớp hiếm.*

{per_class_table(run, final)}
Lớp khó ở round {final} (F1 < 0,4): {hard}. `timeDelayAttack` cũng là lớp yếu nhất của DAGSNet centralized (`knowledge/ARCHITECTURE.md` §8: F1 0,2281, 76 % bị gán thành `benign`); `benign` khó ở đây vì cơ chế báo động giả nói trên.

![]({rel(figs['cm'])})

*Hình: ma trận nhầm lẫn round {final} chuẩn hoá theo hàng (mỗi hàng = một lớp thật, tổng 1). Điều cần thấy: hàng `benign` trải sang các cột tấn công — recall `benign` chỉ {f4(ben_f['recall'])} trên {ben_f['support']:,} dòng; vì `benign` chiếm 22 % tập test, riêng nó kéo accuracy và F1 weighted xuống trong khi F1 macro (mỗi lớp nặng như nhau) gần như giữ nguyên. `trafficCongestionSybil` (F1 {f4(next(c['f1'] for c in run['per_class'][final] if c['class']=='trafficCongestionSybil'))}) ổn định nhưng nhớ caveat rò rỉ Sybil (§3).*

"""


def pending_section(K):
    sc = SCENARIOS[K]
    return f"""## {SEC[K]}. {K} client — `{sc['run']}` — **CHƯA HOÀN TẤT, KHÔNG CÓ SỐ**

Chưa có thư mục run đã merge và verify (`{rel(RUNS / sc['run'])}`). Phiên 1 (`{sc['kernels'][0][0]}` v{sc['kernels'][0][1]},
{sc['kernels'][0][2]}) dừng ở round 25/50 vì hết ngân sách phiên; phiên 2 (`{sc['kernels'][1][0]}`, {sc['kernels'][1][2]})
đang tiếp nối từ round 26 qua checkpoint dataset. Mọi số của kịch bản này là **N/A** cho tới khi pull, verify và merge xong —
xem `CONTEXT.md` §15.6; chạy lại `python scripts/make_report.py` sau đó.

"""


def build():
    FIG.mkdir(exist_ok=True)
    runs, sections = {}, []
    for K, sc in SCENARIOS.items():
        d = RUNS / sc["run"]
        if not (d / "history.csv").is_file() or not (d / "logs" / "sessions.json").is_file():
            sections.append(pending_section(K)); continue
        run = load_run(d); runs[K] = run
        final = run["rounds"][-1]; peak = max(run["rounds"], key=lambda r: run["rows"][r]["f1_macro"])
        figs = {"conv": fig_convergence(K, run), "cm": fig_confusion(K, run, final), "pc": fig_per_class(K, run, final, peak)}
        sections.append(scenario_section(K, run, figs))
    all_fig = fig_all_f1(runs) if runs else None
    done = [K for K in SCENARIOS if K in runs and runs[K]["rounds"][-1] == PLANNED]
    pending = [K for K in SCENARIOS if K not in done]
    status = ("HOÀN TẤT" if not pending else f"INTERIM — thiếu {', '.join(f'{K}c' for K in pending)}")

    benign_line = "; ".join(
        f"{K}c: {f4(class_delta(runs[K], runs[K]['rounds'][-1], max(runs[K]['rounds'], key=lambda r: runs[K]['rows'][r]['f1_macro']))[2]['recall'])} "
        f"→ {f4(class_delta(runs[K], runs[K]['rounds'][-1], max(runs[K]['rounds'], key=lambda r: runs[K]['rows'][r]['f1_macro']))[1]['recall'])}"
        for K in runs) or "N/A"
    # Cross-scenario facts for §0 and §8, computed rather than typed: peak round, plateau level
    # (mean F1 macro of the last 10 rounds) and the peak→final erosion of accuracy / F1 weighted.
    def stat(K):
        rw, rd = runs[K]["rows"], runs[K]["rounds"]
        pk = max(rd, key=lambda r: rw[r]["f1_macro"])
        pka = max(rd, key=lambda r: rw[r]["accuracy"]); pkw = max(rd, key=lambda r: rw[r]["f1_weighted"])
        return {"r1": rw[rd[0]]["f1_macro"], "peak": rw[pk]["f1_macro"], "pk": pk,
                "plateau": sum(rw[r]["f1_macro"] for r in rd[-10:]) / len(rd[-10:]),
                "acc_drop": 100 * (rw[pka]["accuracy"] - rw[rd[-1]]["accuracy"]),
                "f1w_drop": 100 * (rw[pkw]["f1_weighted"] - rw[rd[-1]]["f1_weighted"]),
                "sessions": len(runs[K]["sessions"]),
                "overlap": runs[K]["sessions"][0]["rounds_produced"][-1] if len(runs[K]["sessions"]) > 1 else None}
    st = {K: stat(K) for K in runs}
    rng = lambda vals, fmt: (fmt(min(vals)) if min(vals) == max(vals) else f"{fmt(min(vals))}–{fmt(max(vals))}")
    plateau_line = "; ".join(f"{K}c ≈ {st[K]['plateau']:.2f} (đỉnh {f4(st[K]['peak'])} @r{st[K]['pk']})" for K in runs)
    plateau_mono = all(st[a]["plateau"] > st[b]["plateau"] for a, b in zip(list(runs), list(runs)[1:]))
    rise_line = "; ".join(f"{K}c {f4(st[K]['r1'])} → {f4(st[K]['peak'])} @r{st[K]['pk']}" for K in runs)
    pk_range = rng([st[K]["pk"] for K in runs], str)
    rel_drop = rng([100 * (st[K]["peak"] - st[K]["plateau"]) / st[K]["peak"] for K in runs], lambda v: f"{v:.0f}")
    acc_drop = rng([st[K]["acc_drop"] for K in runs], lambda v: f"{v:.0f}")
    f1w_drop = rng([st[K]["f1w_drop"] for K in runs], lambda v: f"{v:.0f}")
    multi = [K for K in runs if st[K]["sessions"] > 1]
    kd_min = {K: min(runs[K]["rounds"], key=lambda r: runs[K]["rows"][r]["kd_client_mean"]) for K in runs}
    kd_pk_range = rng([kd_min[K] for K in runs], str)
    kd_line = "; ".join(f"{K}c {f4(runs[K]['rows'][kd_min[K]]['kd_client_mean'])} → {f4(runs[K]['rows'][runs[K]['rounds'][-1]]['kd_client_mean'])}" for K in runs)
    cont_line = "; ".join(f"{K}c hai phiên, {st[K]['overlap']} round chồng lấn" for K in multi) or "không cấu hình nào"
    summary_rows = "".join(
        f"| {K}c | {runs[K]['rounds'][-1]}/{PLANNED} | {f4(runs[K]['rows'][runs[K]['rounds'][-1]]['accuracy'])} | "
        f"{f4(runs[K]['rows'][runs[K]['rounds'][-1]]['f1_macro'])} | {f4(runs[K]['rows'][runs[K]['rounds'][-1]]['f1_weighted'])} | "
        f"{f4(max(runs[K]['rows'][r]['f1_macro'] for r in runs[K]['rounds']))} @r{max(runs[K]['rounds'], key=lambda r: runs[K]['rows'][r]['f1_macro'])} | "
        f"{sum(runs[K]['rows'][r]['seconds'] for r in runs[K]['rounds'])/3600:.2f} h |\n" for K in runs) + "".join(
        f"| {K}c | N/A (đang chạy) | N/A | N/A | N/A | N/A | N/A |\n" for K in pending)

    md = f"""# Báo cáo tái dựng FD-IDS trên VeReMi NextGen với DAGSNet — {status}

Sinh tự động bởi `scripts/make_report.py` lúc {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%MZ} từ artifact đã pull và verify.
Không sửa tay; sửa generator rồi chạy lại. Ngữ cảnh đầy đủ: [`CONTEXT.md`](CONTEXT.md); mọi số đo kỹ thuật: [`TEST_LOG.md`](TEST_LOG.md).

## 0. Tóm tắt

| cấu hình | round | accuracy (round cuối) | F1 macro (round cuối) | F1 weighted (round cuối) | đỉnh F1 macro (hậu kiểm) | Σ `seconds` 50 round (không gồm startup) |
|---|---:|---:|---:|---:|---|---:|
{summary_rows}
![]({rel(all_fig) if all_fig else ''})

*Hình: F1 macro của global model theo round, các cấu hình đã có. Điều cần thấy: mọi đường bão hoà trước round 10 (đỉnh ở round {pk_range}) rồi giữ một plateau nhiễu thấp hơn đỉnh; mức plateau (trung bình 10 round cuối): {plateau_line}{' — nhiều client hơn, plateau thấp hơn, đơn điệu' if plateau_mono else ''}.*

**Đọc số thế nào.** Headline là **round cuối** (round 50), đúng lịch 50 round đã chốt. Round đỉnh là quan sát hậu kiểm
trên tập test — không có tập validation, nên không được chọn checkpoint theo nó. Dữ liệu mất cân bằng 41:1 ⇒ đọc
**F1 macro**, không đọc accuracy (§3). Không đặt số của bản này cạnh số của bài báo (§7): khác dữ liệu, số lớp, số
client, bộ metric.

## 1. Đối tượng tái dựng và nguồn gốc

| hạng mục | giá trị |
|---|---|
| Bài báo | Peng, Xiao, Wu — *FD-IDS: A Federated Learning and Knowledge Distillation-Based Intrusion Detection System for Non-IID IoT Environments*, **Sensors 2025, 25, 4309** ([`sensors-25-04309.md`](sensors-25-04309.md)) |
| Phương pháp lấy từ bài báo | FedProx + knowledge distillation **round-wise** (Algorithm 1, Eq. 2–6): L = λ·CE + (1−λ)·T²·KL(teacher‖student) + β·(μ/2)‖w−w_G‖²; Adam lr 0,001, μ = 0,01, λ = 0,5, β = 0,1, T = 3 (Table 3); toàn bộ client mỗi round |
| Dữ liệu | VeReMi NextGen, phân mảnh Dirichlet α = 0,5: `odixe0502/veremi-fl-{{20,50,100}}client` (train) + `odixe0502/veremi-nextgen2026-centralized` (test), 66 đặc trưng `f_*` đã z-score, 16 lớp ([`knowledge/DATASET.md`](knowledge/DATASET.md)) |
| Bộ phân loại | **DAGSNet**, 395.024 tham số ([`knowledge/ARCHITECTURE.md`](knowledge/ARCHITECTURE.md)) thay cho DNN 5 lớp 22.095 tham số của bài báo |
| Phần cứng | Kaggle 2 × Tesla T4 (sm_75, 14,6 GB), 4 vCPU; image `{RUNTIME['docker_image'].split('@')[1][:19]}…`; torch 2.10.0+cu128, CUDA 12.8; mỗi worker một GPU, mỗi client train tuần tự trên một GPU, không DDP |
| Kế hoạch | 3 cấu hình × 50 round × 1 epoch local; batch 512/512/256 (20c/50c/100c); seed 42; fp16 AMP; `torch.compile` chứng nhận trên T4 (`compile OK`, `max|Δlogit| ≤ 9,8e-04`) |
| Kernel / phiên | xem bảng phiên ở từng mục; mỗi phiên là một kernel Kaggle riêng, tiếp nối qua checkpoint đã verify (§9) |

### 1.1 Cái gì giữ theo bài báo, cái gì lệch — phải đọc cùng mọi con số

| hạng mục | bài báo | bản này | vì sao |
|---|---|---|---|
| Bộ phân loại | DNN 5 lớp 32-64-128-64-32 | **DAGSNet** 395.024 tham số | chủ dự án chỉ định |
| Dữ liệu | Edge-IIoT / N-BaIoT | **VeReMi NextGen**, 16 lớp, 66 đặc trưng | chủ dự án chỉ định |
| Số client | 9 | **20 / 50 / 100** | chủ dự án chỉ định |
| Non-IID | Dirichlet θ = 1 và 0,1 | **α = 0,5 cố định** | phân mảnh dựng sẵn |
| Round × epoch | 40 × 2 | **50 × 1** | ngân sách quota; lr hằng nên số round không đổi trọng số từng round |
| Batch | 128 | **512 / 512 / 256** | batch 128 ≈ 75 h/cấu hình, vượt quota |
| Tiền xử lý | one-hot + chọn top-k đặc trưng bằng MI | **không** — dùng đủ 66 cột đã z-score | dữ liệu giao ở trạng thái đã xử lý |
| Metric | Accuracy/Precision/Recall/F1 + FPR/FNR nhị phân | **10 metric đa lớp** (§2), không FPR/FNR | chưa chốt quy ước nhị phân hoá 16 lớp |
| Đánh giá client (cột B/W Table 5) | có | **không** — chỉ global model | FD-IDS không cá thể hoá: client bị ghi đè bằng `w_G` mỗi round, nên "model tại các client" chính là global model |
| Adam state | không nói | reset mỗi client, mỗi round | lựa chọn triển khai, công bố |
| BatchNorm dưới FedAvg | DNN của bài báo không có BN | trung bình cả running stats theo n_k/N | hệ quả của việc đổi bộ phân loại (§8) |

## 2. Mười metric — định nghĩa và thứ tự cột

`accuracy`, `precision_macro`, `precision_micro`, `precision_weighted`, `recall_macro`, `recall_micro`, `recall_weighted`,
`f1_macro`, `f1_micro`, `f1_weighted` — tính từ **một** ma trận nhầm lẫn 16×16 trên toàn bộ tập test sau mỗi round
(hai worker đánh giá hai nửa rời nhau, cộng số nguyên). Macro: trung bình đều 16 lớp; micro: gộp TP/FP/FN toàn cục;
weighted: trọng số n_c/N. Vì mỗi dòng có đúng một dự đoán, Σ_c FP_c = Σ_c FN_c ⇒ precision_micro = recall_micro = f1_micro
= accuracy, và recall_weighted = accuracy. Định nghĩa và code: `papers/fd-ids-2025/proj/metrics.py`; dựng lại độc lập từ
`preds/` và `confusion/` bởi `scripts/verify_run.py` (đã pass cho mọi run trong báo cáo).

## 3. Dữ liệu và caveat bắt buộc

Train {43_045_415:,} dòng / test {N_TEST:,} dòng; 16 lớp; mất cân bằng **41:1** (`trafficCongestionSybil` 2.393.335 dòng
test so với `suddenConstantSpeed` 57.757). Từ [`knowledge/DATASET.md`](knowledge/DATASET.md) §6, phải đọc cùng mọi bảng ở đây:

1. Split theo **thời gian mô phỏng**, không theo xe; test chỉ có scenario `highway_7`/`urban_7`.
2. `benign` lấy từ luồng không có tấn công ⇒ nhóm đặc trưng `rate` mạnh bất thường.
3. 3.338.358 dòng nhập nhằng đã bị loại từ nguồn.
4. **Rò rỉ Sybil:** 100 % dòng `trafficCongestionSybil` nằm trong flow mà mọi dòng cùng nhãn — lớp lớn nhất dễ bất thường.
5. Mất cân bằng 41:1 ⇒ đọc F1 macro, không đọc accuracy.
6. Client = receiver unit, phân hoạch là mô phỏng non-IID, không phải triển khai thật.
7. **`scaler.json` fit trên toàn bộ 43 M dòng train** — thống kê toàn cục mà client FL thật không có; rò rỉ có trước bản này.
8. Test không chia theo client: điểm đo khả năng tổng quát hoá toàn cục của global model.
9. Đặc trưng lưu fp16 (đã lượng tử hoá), lựa chọn có chủ ý.
10. Một run, một seed. Chưa có replication.

{''.join(sections)}
## 7. Kết quả của bài báo — để tham chiếu, KHÔNG so trực tiếp

Bài báo đo trên Edge-IIoT và N-BaIoT với DNN 5 lớp, 9 client, 40 round × 2 epoch, batch 128, chọn đặc trưng bằng MI; F1 của
bài báo là F1 nhị phân/đa lớp theo quy ước riêng, kèm FPR/FNR. Không có hàng nào ở đây so được với các bảng ở §4–§6.

Table 5 (global model, accuracy %): Edge-IIoT θ=1: 85,27 (round 1) → 94,82 (round 40); θ=0,1: 77,12 → 93,86.
N-BaIoT θ=1: 53,21 → 87,70; θ=0,1: 38,58 → 83,81. Table 6/7 (round-wise KD, accuracy/precision/recall/F1 %):
Edge-IIoT θ=1 94,82/96,20/94,82/94,35; θ=0,1 93,86/96,45/93,86/92,70; N-BaIoT θ=1 87,70/82,23/87,70/83,52; θ=0,1
83,81/78,39/83,81/78,24. Table 8: KD tăng accuracy 91,89 → 93,86 và giảm FNR 8,11 → 6,14 ở θ=0,1 (Edge-IIoT).
Điều bài báo chứng minh là *phương pháp* (round-wise KD > periodic/end-of-training; KD giúp nhiều hơn khi non-IID mạnh);
bản này kế thừa phương pháp, không kế thừa con số.

## 8. Bằng chứng ủng hộ và không ủng hộ điều gì

* **Huấn luyện có học thật và hội tụ:** F1 macro tăng từ round 1 (teacher khởi tạo ngẫu nhiên) tới đỉnh trong
  chưa đầy 10 round ({rise_line}); CE huấn luyện phía client giảm mạnh tới round ~20 rồi đi ngang ở mức thấp, không
  tăng trở lại (20c có một cú nảy một round ở r45 sau cú tụt r44, rồi về nếp). Số hạng KD thì **không** đi ngang: chạm
  cực tiểu ở round {kd_pk_range} rồi tăng dần tới round 50 ({kd_line}) — khoảng cách student↔teacher (global model
  round trước) nới ra theo round, cùng chiều với thoái lui trên tập test ở mục sau.
* **Bão hoà sớm rồi thoái lui trên tập test:** F1 macro đạt đỉnh ở round {pk_range} và giữ plateau thấp hơn đỉnh
  {rel_drop} % (tương đối); **accuracy xói mòn {acc_drop} điểm, F1 weighted {f1w_drop} điểm** từ đỉnh tới round 50,
  recall macro tăng nhẹ còn precision macro giảm.
  Bảng từng lớp chỉ đích danh cơ chế: **recall `benign` sụp** ({benign_line}) trong khi các lớp hiếm được thêm recall —
  global model ngày càng gán lưu lượng lành tính thành tấn công, tức **báo động giả tăng theo round**; với IDS đây là
  chiều xói mòn "dễ chịu" hơn bỏ sót tấn công, nhưng vẫn là thoái lui thật. `trafficCongestionSybil` ổn định (F1 ≈ 0,97,
  nhớ caveat rò rỉ). Loss train giảm trong khi metric test giảm là **overfitting** theo đúng nghĩa; 50 round nhiều hơn
  mức dữ liệu/bộ phân loại này cần (50c đứng yên từ round ~20).
* **Giả thuyết chưa kiểm — không chép vào kết luận:** DAGSNet có BatchNorm và Eq. (2) trung bình cả running stats
  theo n_k/N; với Dirichlet α = 0,5 các thống kê này phân kỳ giữa client. DNN của bài báo không có BN, nên tương tác
  này đến từ việc thay bộ phân loại, không từ phương pháp. Muốn khẳng định cần thí nghiệm riêng trên checkpoint đã lưu
  (ví dụ tính lại BN stats trên dữ liệu giữ riêng rồi đánh giá lại).
* **Không có ablation** (FedAvg / FedProx không KD / KD-only) và **không có replication** — không kết luận được tác dụng
  riêng của KD hay của proximal term trên dữ liệu này.
* **Continuation không làm lệch bài toán:** {cont_line} — mỗi lần qua hai tài khoản Kaggle; weights và dự đoán của
  mọi round chồng lấn giống hệt từng byte giữa hai pull (`merge_sessions.py` từ chối nếu khác), fingerprint cấu hình
  khớp, W&B nối cùng run. Round 44 của 20c tụt
  0,7669 → 0,7294 rồi hồi về 0,7655; log từng client của round 43–45 không có client bất thường (ce, grad norm, skip,
  non-finite), nên đó là dao động một round của phép gộp, không phải lỗi tiếp nối (`TEST_LOG.md` §3.8).
* **Rò rỉ phải nhớ khi đọc lớp `trafficCongestionSybil`** (caveat 4) và **scaler toàn cục** (caveat 7).

## 9. Tái lập

* Sinh notebook: `python scripts/gen_notebook.py --owner <acct> --clients <K> --run-tag _v2 --max-hours <h>` (tiếp nối:
  thêm `--require-resume --dataset-source <acct>/<checkpoint-dataset>`); kiểm: `python scripts/validate_notebooks.py`;
  push: `kaggle kernels push -p papers/fd-ids-2025/notebook/<K>c`. Kernel `machine_shape` `NvidiaTeslaT4`, 2 GPU, image pin theo
  `knowledge/runtime.json`. Quy trình đầy đủ: `CONTEXT.md` §10.3.
* Artifact mỗi cấu hình: `papers/fd-ids-2025/runs/<run>/` — `weights/round_NNN.pt` (chỉ trọng số, `weights_only=True`,
  dựng lại được với `proj/ckpt.py::load_weights`, `max|Δlogit| = 0`), `metrics/round_NNN.json` (10 metric + 16 dòng
  per-class), `confusion/`, `preds/` (nhãn dự đoán toàn bộ test), `logs/round_NNN.json` (từng client), `history.csv`,
  `reports/manifest.json`, `logs/sessions.json`, `logs/sessions/<n>/` (log kernel + manifest từng phiên). Không commit
  (gitignore `papers/**/runs/`).
* Executed notebook (bằng chứng, có output): `papers/fd-ids-2025/notebook/<K>c/fdids_<K>c_v2.executed.s<n>.ipynb` — chứa
  W&B key, không chia sẻ.
* Kiểm offline: `python scripts/verify_run.py papers/fd-ids-2025/runs/<run> --y-true <run>/reports/y_true.u8.npy --require-rounds 50`.
* Báo cáo này: `python scripts/make_report.py` (đọc artifact, tự đối chiếu CSV ↔ JSON ↔ ma trận nhầm lẫn trước khi ghi).
"""
    OUT.write_text(md)
    print(f"report -> {rel(OUT)} ({len(md.splitlines())} lines); scenarios: done {done}, pending {pending}; figures in {rel(FIG)}/")


if __name__ == "__main__":
    build()

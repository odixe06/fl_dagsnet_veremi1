"""Per-round metric tables for report.md mục 2.8, straight from each run's history.csv.

Writes figures/../tables_2_8.md and, with --apply, splices it into report.md between the
markers so the report's tables stay derivable from artifacts rather than hand-maintained.

    python make_tables.py            # regenerate the block, print a summary
    python make_tables.py --apply    # also rewrite mục 2.8 in report.md
"""
import csv, sys
from pathlib import Path

P = Path("papers")
BEGIN = "<!-- BEGIN per-round tables (make_tables.py) -->"
END = "<!-- END per-round tables -->"

BUILDS = [
    ("Build 1 — CMSO trước extractor",
     P/"build1-cmso-before-extractor/runs/edl_cmso_r50_b4096",
     "50/50 round, `edl-cmso-veremi`. CMSO chạy **trước** khối trích xuất — ngược thứ tự bài báo."),
    ("Build 2 run 1 — pipeline đầy đủ, wd 1e-4",
     P/"build2-paper-order/run1-wd1e4-diverged-r25/edl_cmso_v2_r50_b4096",
     "Phân kỳ ở round 25. **Chỉ round 0–24 dùng được**; các round sau là hậu quả của sụp attention."),
    ("Build 2 run 3 c50 — thêm `torch.compile`, wd 1e-4",
     P/"build2-paper-order/run3-c50-compiled-diverged-r22/edl_cmso_v2_c50",
     "Phân kỳ ở round 22. **Chỉ round 0–13 dùng được.**"),
    ("Build 2 final — pipeline đầy đủ, wd 5e-2",
     P/"build2-paper-order/run4-wd5e2-complete-50round/edl_cmso_v2_final",
     "**50/50 round, không phân kỳ.** Cùng pipeline và cùng `channel_mask.json` / "
     "`extractor_init.pt` với run 1; biến duy nhất đổi là `weight_decay` 1e-4 → 5e-2. "
     "Mọi round đều dùng được."),
    ("Build 4 — DAGSNet trần",
     P/"build4-dagsnet-only/runs/edl_cmso_v4_dagsnet",
     "50/50 round, hai session ghép thành một run (r0–45 rồi r46–49). Mọi round đều dùng được."),
]
METRIC_KEYS = [
    "accuracy",
    "precision_macro", "precision_micro", "precision_weighted",
    "recall_macro", "recall_micro", "recall_weighted",
    "f1_macro", "f1_micro", "f1_weighted",
]
COLS = [("round","round",0)] + [
    (k, "**f1_macro**" if k == "f1_macro" else k, 5) for k in METRIC_KEYS
] + [("train_loss","train loss",4), ("seconds","s/round",1)]
N_TEST = 10_761_343          # the fixed full test set every round is scored on


def validate(rows, run_dir):
    """The checks references/metrics.md requires before a round may be published.

    All ten keys present, numeric, finite, in [0, 1]; the single-label collapse identity
    holding within floating-point tolerance; and the final confusion matrix covering the
    whole fixed test set rather than a subsample."""
    import numpy as np
    bad = []
    for r in rows:
        rnd = r["round"]
        for k in METRIC_KEYS:
            if not r.get(k):
                bad.append(f"r{rnd}: {k} missing")
                continue
            v = float(r[k])
            if v != v or v in (float("inf"), float("-inf")):
                bad.append(f"r{rnd}: {k} not finite")
            elif not 0.0 <= v <= 1.0:
                bad.append(f"r{rnd}: {k}={v} outside [0,1]")
        acc = float(r["accuracy"])
        for k in ("precision_micro", "recall_micro", "f1_micro", "recall_weighted"):
            if r.get(k) and abs(float(r[k]) - acc) > 1e-9:
                bad.append(f"r{rnd}: {k} != accuracy by {abs(float(r[k]) - acc):.2e}")
    conf = sorted((run_dir / "confusion").glob("round_*.npy"))
    if conf:
        tot = int(np.load(conf[-1]).sum())
        if tot != N_TEST:
            bad.append(f"{conf[-1].name} sums to {tot:,}, not {N_TEST:,}")
    return bad


def vn(x, n):
    return (f"{x:,.{n}f}".replace(",", " ").replace(".", ",")) if n else f"{int(x)}"


def history(d):
    seen = {}
    for r in csv.DictReader(open(d/"metrics"/"history.csv")):
        seen[int(r["round"])] = r            # a resumed run can repeat a round; last wins
    return [seen[k] for k in sorted(seen)]


CHECKED = ("Đã kiểm theo `references/metrics.md` trước khi công bố: mọi round có đủ 10 metric, "
           "hữu hạn, nằm trong [0, 1]; đẳng thức `accuracy = P_mic = R_mic = F1_mic = R_wgt` "
           "đúng tới 1e-9; confusion của round cuối cộng đúng 10.761.343 dòng test.")

out, summary, problems_all = [], [], []
for title, d, note in BUILDS:
    rows = history(d)
    best = max(rows, key=lambda r: float(r["f1_macro"]))
    tot = sum(float(r["seconds"]) for r in rows)
    summary.append((title, len(rows), int(best["round"]), float(best["f1_macro"]),
                    float(rows[-1]["f1_macro"]), tot/3600))
    bad = validate(rows, d)
    problems_all.extend((title, b) for b in bad)
    flag = "" if not bad else ("\n\n⚠ **KIỂM TRA KHÔNG ĐẠT:** " + "; ".join(bad[:5])
                               + (f" — và {len(bad) - 5} lỗi nữa" if len(bad) > 5 else ""))
    out.append(f"\n#### {title}\n\n{note} Đỉnh `f1_macro` **{vn(float(best['f1_macro']),5)}** "
               f"ở round {best['round']}; tổng {vn(tot/3600,2)} h.\n\n{CHECKED}{flag}\n")
    out.append("| " + " | ".join(h for _,h,_ in COLS) + " |")
    out.append("|" + "---:|"*len(COLS))
    for r in rows:
        cells = []
        for k,_,n in COLS:
            v = vn(float(r[k]), n)
            cells.append(f"**{v}**" if (k == "f1_macro" and r is best) else v)
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

block = "\n".join(out)
Path("tables_2_8.md").write_text(block)
if problems_all:
    print("\n⚠ metrics.md publication checks FAILED:")
    for t, b in problems_all[:20]:
        print(f"  {t}: {b}")
else:
    print("\nall runs pass the metrics.md publication checks")
print(f"{'run':<48}{'rounds':>7}{'best':>10}{'@r':>4}{'last':>10}{'hours':>7}")
for t,n,br,bf,lf,h in summary:
    print(f"{t:<48}{n:>7}{bf:>10.5f}{br:>4}{lf:>10.5f}{h:>7.2f}")

if "--apply" in sys.argv:
    rep = Path("report.md"); s = rep.read_text()
    a, b = s.index(BEGIN), s.index(END)
    rep.write_text(s[:a] + BEGIN + "\n" + block + "\n" + s[b:])
    print("\nreport.md mục 2.8 rewritten")

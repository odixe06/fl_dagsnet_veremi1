#!/usr/bin/env python
"""Turn verified run directories into the tables the report prints.

Two products per scenario, because they answer different questions:
  per_client_<n>client.csv       every client's 10 metrics, every round, both eval rules
  mean_over_clients_<n>client.csv the report's headline table -- the same 10 metrics averaged
                                  over clients, plus the spread that average hides

`metrics/round_NNN.json` is the authority: `aggregate.<rule>.mean_over_clients` is what the
driver itself computed, so the mean column is copied, never re-derived. std/min/max ARE derived
from `per_client`, and the script asserts the recomputed mean matches to 1e-9 -- that assertion
is the only thing standing between a transcription bug and a wrong headline number.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["accuracy", "precision_macro", "precision_micro", "precision_weighted",
           "recall_macro", "recall_micro", "recall_weighted",
           "f1_macro", "f1_micro", "f1_weighted"]
RULES = ["proto", "clf"]


def rounds_of(run: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in (run / "metrics").glob("round_*.json"))


def load(run: Path, rnd: int) -> dict:
    return json.loads((run / "metrics" / f"round_{rnd:03d}.json").read_text())


def emit(run: Path, out: Path, tag: str) -> dict:
    rounds = rounds_of(run)
    per_client_rows, mean_rows = [], []
    for rnd in rounds:
        d = load(run, rnd)
        for rule in RULES:
            clients = d["per_client"][rule]
            for cid, m in enumerate(clients):
                per_client_rows.append({"round": rnd, "rule": rule, "client_id": cid,
                                        **{k: m[k] for k in METRICS}})
            row = {"round": rnd, "rule": rule, "n_clients": len(clients)}
            declared = d["aggregate"][rule]["mean_over_clients"]
            for k in METRICS:
                vals = [m[k] for m in clients]
                got = mean(vals)
                if abs(got - declared[k]) > 1e-9:
                    raise SystemExit(f"{tag} r{rnd} {rule} {k}: per_client mean {got!r} != "
                                     f"declared {declared[k]!r}")
                row[k] = declared[k]
            f1 = [m["f1_macro"] for m in clients]
            row["f1_macro_std"] = pstdev(f1) if len(f1) > 1 else 0.0
            row["f1_macro_min"], row["f1_macro_max"] = min(f1), max(f1)
            mean_rows.append(row)

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_client", per_client_rows), ("mean_over_clients", mean_rows)):
        p = out / f"{name}_{tag}.csv"
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{p}  {len(rows)} rows")

    by = {(r["round"], r["rule"]): r for r in mean_rows}
    peak = max(rounds, key=lambda r: by[(r, "proto")]["f1_macro"])
    return {"scenario": tag, "rounds": len(rounds), "last_round": rounds[-1],
            "n_clients": by[(rounds[-1], "proto")]["n_clients"],
            "peak_round": peak,
            "peak": {rule: by[(peak, rule)] for rule in RULES},
            "final": {rule: by[(rounds[-1], rule)] for rule in RULES},
            "extra_final": load(run, rounds[-1])["extra"],
            "extra_peak": load(run, peak)["extra"],
            "communication": load(run, rounds[-1])["communication"],
            "n_test": load(run, rounds[-1])["n_test"]}


def md(headers: list[str], rows: list[list[str]], align: str = "r") -> str:
    sep = ["---" if i == 0 else (":--" if align == "l" else "--:") for i in range(len(headers))]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def f4(x: float) -> str:
    return f"{x:.4f}"


def tables(run: Path, out: Path, tag: str, s: dict) -> dict[str, str]:
    """Markdown fragments, keyed by the marker name they get spliced into."""
    t: dict[str, str] = {}
    import csv as _csv
    rows = list(_csv.DictReader((out / f"mean_over_clients_{tag}.csv").open()))
    for rule in RULES:
        rr = [r for r in rows if r["rule"] == rule]
        t[f"mean_{tag}_{rule}"] = md(
            ["round"] + [m.replace("_macro", "_M").replace("_micro", "_u")
                         .replace("_weighted", "_w").replace("precision", "prec")
                         .replace("recall", "rec") for m in METRICS],
            [[r["round"]] + [f4(float(r[m])) for m in METRICS] for r in rr])

    ap = aps_scale(run)
    t[f"aps_{tag}"] = md(
        ["class", "τ (round 1)", "‖proto‖ r1", f"‖proto‖ r{ap[0]['round_final']}", "growth"],
        [[r["class"], f'{r["tau"]:.3f}', f'{r["norm_r1"]:.2f}', f'{r["norm_final"]:.2f}',
          f'{r["growth"]:.2f}×'] for r in ap])

    pc = list(_csv.DictReader((out / f"per_client_{tag}.csv").open()))
    for label, rnd in (("final", s["last_round"]), ("peak", s["peak_round"])):
        for rule in RULES:
            sel = [r for r in pc if r["rule"] == rule and int(r["round"]) == rnd]
            t[f"perclient_{tag}_{rule}_{label}"] = md(
                ["client"] + [m.replace("_macro", "_M").replace("_micro", "_u")
                              .replace("_weighted", "_w").replace("precision", "prec")
                              .replace("recall", "rec") for m in METRICS],
                [[r["client_id"]] + [f4(float(r[m])) for m in METRICS] for r in sel])
    return t


def aps_scale(run: Path) -> list[dict]:
    """Per class: how far APS pulls the prototype, relative to the prototype's own size.

    tau_j = ||mu * cG[j] . m_j|| / mean_i ||cL[i,j] . m_j||   measured at ROUND 1, which is the
    first target APS ever sets (round 1 trains without regularization, so it is also the only
    round whose prototypes are identical for every k -- which makes tau at round 1 EXACTLY
    proportional to k, not approximately). tau > 1 means APS asks for a prototype LARGER than
    the one the network produces, and since cG is itself built from cL, satisfying it enlarges
    the next round's target too. Read with the norm growth column, this is the whole finding.
    """
    import numpy as np
    import torch
    cfg = json.loads((run / "config.json").read_text())
    mu = json.loads((run / "mu.json").read_text())["value"]
    masks = np.load(run / "masks.npy")
    rounds = rounds_of(run)
    r0, r1 = 1, rounds[-1]

    def read(r: int):
        d = torch.load(run / "protos" / f"round_{r:03d}.pt", map_location="cpu",
                       weights_only=True)
        return d["local"], d["counts"], d["global_sparse"]

    loc0, cnt0, gs0 = read(r0)
    loc1, cnt1, _ = read(r1)
    out = []
    for j, name in enumerate(cfg["class_names"]):
        m = torch.tensor(masks[j], dtype=torch.bool)
        own0, own1 = cnt0[:, j] > 0, cnt1[:, j] > 0
        masked = (loc0[own0, j, :] * m).norm(dim=1).mean().item()
        out.append({"class": name,
                    "tau": (mu * gs0[j] * m).norm().item() / masked,
                    "norm_r1": loc0[own0, j, :].norm(dim=1).mean().item(),
                    "norm_final": loc1[own1, j, :].norm(dim=1).mean().item(),
                    "round_final": r1})
    for r in out:
        r["growth"] = r["norm_final"] / r["norm_r1"]
    return sorted(out, key=lambda r: -r["tau"])


def cross_scenario(summary: list[dict], runs: Path) -> dict[str, str]:
    """The two tables that span scenarios: the headline, and the verifier's own verdict.

    The verify row is produced by RUNNING scripts/verify_run.py, not by quoting a number from a
    log. A report that quotes a stale pass count is exactly the failure this table exists to
    rule out.
    """
    import csv as _csv
    import re
    import subprocess

    head = []
    for sm in summary:
        tag = sm["scenario"]
        hist = list(_csv.DictReader((runs / f"tinyproto_fp_{tag}" / "metrics" / "history.csv").open()))
        reg = [(int(h["round"]), float(h["reg_loss"])) for h in hist if int(h["round"]) > 1]
        gn = [(int(h["round"]), float(h["grad_norm"])) for h in hist]
        rmin, gmin = min(reg, key=lambda t: t[1]), min(gn, key=lambda t: t[1])
        pk, fi = sm["peak"], sm["final"]
        head.append([
            f'{sm["n_clients"]} client',
            f'{sm["peak_round"]}', f4(pk["proto"]["f1_macro"]), f4(fi["proto"]["f1_macro"]),
            f'{100 * (fi["proto"]["f1_macro"] / pk["proto"]["f1_macro"] - 1):+.1f}%',
            f4(fi["clf"]["f1_macro"]),
            f'{pk["proto"]["f1_macro"] - pk["clf"]["f1_macro"]:+.4f}',
            f'{fi["proto"]["f1_macro"] - fi["clf"]["f1_macro"]:+.4f}',
            f'r{rmin[0]} {rmin[1]:.6f} → {reg[-1][1]:.6f} ({reg[-1][1] / rmin[1]:.0f}×)',
            f'r{gmin[0]} {gmin[1]:.4f} → {gn[-1][1]:.4f} ({gn[-1][1] / gmin[1]:.1f}×)',
        ])
    t = {"headline": md(
        ["kịch bản", "round đỉnh", "proto f1_M đỉnh", "proto f1_M r50", "thay đổi",
         "clf f1_M r50", "proto−clf đỉnh", "proto−clf r50", "reg_loss đáy → r50",
         "grad_norm đáy → r50"], head)}

    ver = []
    for sm in summary:
        run = runs / f"tinyproto_fp_{sm['scenario']}"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/verify_run.py"),
                            "--require-complete", str(run)], capture_output=True, text=True)
        m = re.search(r"(\d+) checks passed, (\d+) failed", r.stdout)
        if not m:
            raise SystemExit(f"verify_run.py gave no verdict for {run}:\n{r.stdout}{r.stderr}")
        ver.append([f'{sm["n_clients"]} client', f'{sm["rounds"]}/50',
                    f'**{int(m.group(1)):,}**'.replace(",", "."), f'**{m.group(2)}**',
                    "✅ pass" if r.returncode == 0 and m.group(2) == "0" else "❌ FAIL"])
    t["verify"] = md(["kịch bản", "round", "số phép kiểm", "số lỗi", ""], ver)
    return t


def splice(doc: Path, frags: dict[str, str]) -> None:
    """Replace the body between <!-- BEGIN TABLE x --> and <!-- END TABLE x -->.

    Idempotent, and it fails loudly on a marker with no fragment: a report that silently keeps
    a stale table is worse than one that will not build.
    """
    import re
    text = doc.read_text()
    used = set()

    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in frags:
            raise SystemExit(f"{doc}: no generated table named {name!r}")
        used.add(name)
        return f"<!-- BEGIN TABLE {name} -->\n{frags[name]}\n<!-- END TABLE {name} -->"

    text = re.sub(r"<!-- BEGIN TABLE (\S+) -->.*?<!-- END TABLE \1 -->", sub, text, flags=re.S)
    doc.write_text(text)
    print(f"{doc}: spliced {len(used)} tables")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="papers/tinyproto-lee-2026/runs")
    ap.add_argument("--out", default="papers/tinyproto-lee-2026/report")
    ap.add_argument("--scenarios", default="20,50,100")
    ap.add_argument("--splice", default="", help="report .md whose TABLE markers to refresh")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    summary = []
    for n in a.scenarios.split(","):
        tag = f"{n}client"
        run = runs / f"tinyproto_fp_{tag}"
        if not run.is_dir():
            raise SystemExit(f"missing {run}")
        summary.append(emit(run, out, tag))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{out / 'summary.json'}")
    if a.splice:
        frags: dict[str, str] = {}
        for sm in summary:
            frags.update(tables(runs / f"tinyproto_fp_{sm['scenario']}", out, sm["scenario"], sm))
        frags.update(cross_scenario(summary, runs))
        splice(Path(a.splice), frags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

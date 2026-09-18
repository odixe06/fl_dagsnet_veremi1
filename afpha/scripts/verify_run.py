#!/usr/bin/env python
"""Contract checks against a run directory.

Usage: conda run -n nckh python scripts/verify_run.py <run_dir> [--rounds all|last]

Everything is read from the run's own config.json -- client count, batch and round
count are never assumed. Checks are arithmetic or reload-based; none needs a GPU.
Exit code 0 means every check passed.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import afpha            # noqa: E402
import flatpack         # noqa: E402
from dagsnet import N_PARAMS, build_dagsnet          # noqa: E402
from metrics import METRIC_KEYS, metrics_from_confusion, per_class_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--rounds", default="all", choices=("all", "last"),
                    help="Check every completed round (default) or only the last one.")
    ap.add_argument("--require-complete", action="store_true",
                    help="Also require that all config.json rounds are finished. Without "
                         "this the verifier only certifies the rounds that exist, which is "
                         "what you want mid-run and NOT enough to call a scenario done.")
    args = ap.parse_args()

    R = Path(args.run_dir)
    cfg = json.loads((R / "config.json").read_text())
    n_clients = int(cfg["num_clients"])
    batch = int(cfg["batch"])
    n_test = int(cfg["n_test"])
    clusters = cfg["clusters"]
    done = sorted(int(q.stem.split("_")[1]) for q in (R / "complete").glob("round_*.done"))
    assert done, f"no completed rounds under {R}"
    check_rounds = done if args.rounds == "all" else done[-1:]
    ok, fail = [], []

    def check(name, fn):
        try:
            ok.append(f"{name}: {fn()}")
        except AssertionError as exc:
            fail.append(f"{name}: {exc}")

    # ---------- run-level ----------
    check("config", lambda: (
        f"{n_clients} clients, batch {batch}, {n_test:,} test rows, "
        f"{len(done)} completed rounds {done[0]}..{done[-1]}"))

    def complete_ok():
        # Deliberately opt-in. Checking only the rounds that exist is the right behaviour
        # for a mid-run checkpoint, but it means exit 0 alone never proves a scenario
        # finished -- a 3-round directory whose config says 50 passes everything else.
        want = int(cfg["rounds"])
        assert done[-1] == want, (
            f"only {done[-1]} of {want} rounds are complete; this run is unfinished")
        return f"all {want} rounds complete"
    if args.require_complete:
        check("completion gate", complete_ok)

    def contiguous():
        assert done == list(range(1, done[-1] + 1)), f"gap in completed rounds: {done}"
        return f"rounds 1..{done[-1]} with no gap"
    check("round continuity", contiguous)

    def clusters_ok():
        expect = afpha.build_clusters(n_clients)
        assert clusters == expect, "clusters in config.json differ from the seeded partition"
        flat = [c for grp in clusters for c in grp]
        assert sorted(flat) == list(range(n_clients)), "clusters are not a partition"
        assert all(len(g) == afpha.CLUSTER_SIZE for g in clusters), "cluster size drift"
        return f"{len(clusters)} clusters of {afpha.CLUSTER_SIZE}, exact partition"
    check("clusters", clusters_ok)

    def history_ok():
        import csv
        rows = list(csv.DictReader((R / "metrics" / "history.csv").open()))
        got = [int(r["round"]) for r in rows]
        assert len(got) == len(set(got)), f"duplicate rows in history.csv: {got}"
        assert set(done) <= set(got), f"completed rounds missing from history: {set(done)-set(got)}"
        # The CSV is derived from the per-round JSON. Checking only that its numbers are
        # finite lets an edited or stale row through -- and the CSV is what the report
        # tables and plots are built from, so it is exactly the file worth tampering with.
        worst = 0.0
        for r in rows:
            rnd = int(r["round"])
            if rnd not in done:
                continue
            src = json.loads((R / "metrics" / f"round_{rnd:03d}.json").read_text())
            for k in METRIC_KEYS:
                assert k in r, f"history.csv missing column {k}"
                assert math.isfinite(float(r[k])), f"round {rnd} {k} is not finite"
                d = abs(float(r[k]) - float(src[k]))
                assert d <= 5e-7, (
                    f"history.csv round {rnd} {k}={r[k]} disagrees with "
                    f"metrics/round_{rnd:03d}.json {src[k]:.6f} by {d:.2e}")
                worst = max(worst, d)
        return (f"{len(rows)} rows, no duplicates, all 10 metrics match the per-round JSON "
                f"(max|d|={worst:.1e}, 6-dp rounding tolerance)")
    check("history.csv vs per-round JSON", history_ok)

    def flatpack_ok():
        t = flatpack.Template(build_dagsnet())
        sd = torch.load(R / "weights" / f"round_{done[-1]:03d}.pt", weights_only=True)
        sd2 = t.unflatten(*t.flatten(sd))
        assert set(sd2) == set(sd)
        d = max((sd2[k].float() - sd[k].float()).abs().max().item() for k in sd)
        assert d == 0.0, f"flatpack round-trip changed weights by {d}"
        assert t.n_params == N_PARAMS
        return f"exact; parameter block = first {t.n_params:,} entries"
    check("flatpack round-trip", flatpack_ok)

    def lr_ok():
        total = int(cfg["rounds"])
        assert abs(afpha.lr_for_round(1, total) - afpha.LR_MAX) < 1e-12
        assert abs(afpha.lr_for_round(total, total) - afpha.LR_MIN) < 1e-12
        return f"round 1 = {afpha.LR_MAX}, round {total} = {afpha.LR_MIN}"
    check("cosine LR endpoints", lr_ok)

    def hier_ok():
        g = torch.Generator().manual_seed(0)
        P = 4096
        rows = {i: int(torch.randint(1000, 90000, (1,), generator=g)) for i in range(n_clients)}
        cf = {i: torch.randn(P, generator=g) for i in range(n_clients)}
        ci = {i: torch.randint(0, 50, (7,), generator=g) for i in range(n_clients)}
        hf, hi_, _ = afpha.hierarchical_aggregate(cf, ci, rows, clusters)
        tot = sum(rows.values())
        flat = sum(cf[i].double() * (rows[i] / tot) for i in range(n_clients)).float()
        d = (hf - flat).abs().max().item()
        assert d < 1e-5, d
        assert (hi_ == torch.stack([ci[i] for i in range(n_clients)]).max(0).values).all()
        return f"cluster-then-server == flat sample-weighted FedAvg, max|d|={d:.1e}"
    check("hierarchical aggregation identity", hier_ok)

    def mu_ok():
        g = torch.Generator().manual_seed(1)
        rows = {i: int(torch.randint(1000, 90000, (1,), generator=g)) for i in range(n_clients)}
        drifts = {i: float(torch.rand(1, generator=g)) for i in range(n_clients)}
        mu = afpha.next_mu(drifts, rows)
        assert all(afpha.MU_BASE <= v < 2 * afpha.MU_BASE for v in mu.values())
        zero = afpha.next_mu({i: 0.0 for i in range(n_clients)}, rows)
        assert all(abs(v - afpha.MU_BASE) < 1e-15 for v in zero.values())
        return f"in [{afpha.MU_BASE}, {2*afpha.MU_BASE}); zero drift -> {afpha.MU_BASE}"
    check("adaptive mu range", mu_ok)

    # ---------- per-round ----------
    y_true = np.load(R / "preds" / "y_true.npy")
    n_classes = len(json.loads((R / "per_class" / f"round_{done[-1]:03d}.json").read_text()))

    def rounds_ok():
        worst_metric, worst_step = 0.0, None
        tmpl = flatpack.Template(build_dagsnet())
        total_rounds = int(cfg["rounds"])
        cfg_rows = cfg.get("rows_per_client")
        prev_mu_next = None
        for rnd in check_rounds:
            tag = f"{rnd:03d}"
            sd = torch.load(R / "weights" / f"round_{tag}.pt", weights_only=True)
            model = build_dagsnet()
            model.load_state_dict(sd, strict=True)
            assert any("running_mean" in k for k in sd), f"round {rnd}: BN buffers missing"
            assert any("num_batches_tracked" in k for k in sd), f"round {rnd}: counters missing"
            # strict=True proves the *shape* of the checkpoint, nothing about its values.
            # A single NaN loaded happily and the round still verified.
            for k, v in sd.items():
                if v.is_floating_point():
                    assert torch.isfinite(v).all(), (
                        f"round {rnd}: weights/{tag}.pt key {k} holds non-finite values")
            f_vec, i_vec = tmpl.flatten(sd)
            back = tmpl.unflatten(f_vec, i_vec)
            assert set(back) == set(sd), f"round {rnd}: flatpack changed the key set"
            rt = max((back[k].float() - sd[k].float()).abs().max().item() for k in sd)
            assert rt == 0.0, f"round {rnd}: flatpack round-trip changed weights by {rt}"

            cm = np.load(R / "confusion" / f"round_{tag}.npy")
            saved = json.loads((R / "metrics" / f"round_{tag}.json").read_text())
            recomputed = metrics_from_confusion(cm)
            d = max(abs(saved[k] - recomputed[k]) for k in METRIC_KEYS)
            assert d < 1e-12, f"round {rnd}: metrics disagree with confusion by {d}"
            worst_metric = max(worst_metric, d)

            yp = np.load(R / "preds" / f"round_{tag}_ypred.npy")
            assert len(yp) == n_test == int(cm.sum()), (
                f"round {rnd}: coverage {len(yp)}/{int(cm.sum())} != {n_test}")
            rebuilt = np.bincount(y_true.astype(np.int64) * n_classes + yp.astype(np.int64),
                                  minlength=n_classes ** 2).reshape(n_classes, n_classes)
            assert (rebuilt == cm).all(), f"round {rnd}: confusion disagrees with predictions"

            pc = json.loads((R / "per_class" / f"round_{tag}.json").read_text())
            pc_ref = per_class_report(cm, [e["class_name"] for e in pc])
            for a, b in zip(pc, pc_ref):
                for f in ("precision", "recall", "f1", "support", "predicted"):
                    assert abs(float(a[f]) - float(b[f])) < 1e-12, (
                        f"round {rnd}: per_class {a['class_name']}.{f} disagrees with confusion")

            cl = json.loads((R / "client_log" / f"round_{tag}.json").read_text())
            assert len(cl["clients"]) == n_clients, (
                f"round {rnd}: {len(cl['clients'])} client entries, expected {n_clients}")
            ids = [c["client_id"] for c in cl["clients"]]
            assert sorted(ids) == list(range(n_clients)), (
                f"round {rnd}: client ids are not exactly 0..{n_clients-1} "
                f"(duplicates or gaps): {sorted(ids)[:12]}...")
            if cfg_rows is not None:
                for c in cl["clients"]:
                    assert c["rows"] == cfg_rows[c["client_id"]], (
                        f"round {rnd}: client {c['client_id']} logged {c['rows']} rows, "
                        f"config.json says {cfg_rows[c['client_id']]}")
            need = {"client_id", "rows", "steps", "skipped", "applied_steps", "skip_pct",
                    "ce_mean", "grad_norm_mean", "lr", "mu", "seed", "finite_weights",
                    "finite_loss", "finite_grad", "seconds", "device"}
            missing = need - set(cl["clients"][0])
            assert not missing, f"round {rnd}: client log missing {sorted(missing)}"
            for c in cl["clients"]:
                assert c["finite_weights"] and c["finite_loss"] and c["finite_grad"], (
                    f"round {rnd}: client {c['client_id']} committed with non-finite stats")
                assert c["skipped"] < c["steps"] or c["steps"] == 0, (
                    f"round {rnd}: client {c['client_id']} had every step skipped")
                assert c["applied_steps"] == c["steps"] - c["skipped"], (
                    f"round {rnd}: client {c['client_id']} applied-step count inconsistent")
                # The shuffle and dropout seed must be a pure function of (round, client),
                # not of which GPU picked the client up or how many ran before it.
                assert c["seed"] == afpha.client_shuffle_seed(rnd, c["client_id"]), (
                    f"round {rnd}: client {c['client_id']} seed {c['seed']} does not match "
                    "the (round, client) formula -- scheduling is leaking into the RNG")
            want_lr = afpha.lr_for_round(rnd, total_rounds)
            for c in cl["clients"]:
                assert abs(float(c["lr"]) - want_lr) < 1e-12, (
                    f"round {rnd}: client {c['client_id']} trained at lr {c['lr']}, "
                    f"the cosine schedule over {total_rounds} rounds says {want_lr}")
            # mu is a chain: round t's drift sets round t+1's coefficient. Checking the
            # formula on random numbers proves the function, not the run; these are the
            # values the run actually stored and carried across the resume boundary.
            drift = {int(k): float(v) for k, v in cl["drift"].items()}
            rows_map = {c["client_id"]: c["rows"] for c in cl["clients"]}
            mu_ref = afpha.next_mu(drift, rows_map)
            for i, v in cl["mu_next"].items():
                assert abs(float(v) - mu_ref[int(i)]) < 1e-12, (
                    f"round {rnd}: mu_next[{i}]={v} does not match the drift formula "
                    f"({mu_ref[int(i)]})")
            if prev_mu_next is not None:
                for c in cl["clients"]:
                    want = prev_mu_next[c["client_id"]]
                    assert abs(float(c["mu"]) - want) < 1e-12, (
                        f"round {rnd}: client {c['client_id']} used mu {c['mu']}, but "
                        f"round {rnd-1} set {want}")
            elif rnd == 1:
                for c in cl["clients"]:
                    assert abs(float(c["mu"]) - afpha.MU_BASE) < 1e-15, (
                        f"round 1: client {c['client_id']} used mu {c['mu']}, expected "
                        f"{afpha.MU_BASE}")
            prev_mu_next = {int(i): float(v) for i, v in cl["mu_next"].items()}

            total_steps = sum(c["steps"] for c in cl["clients"])
            expect = sum(math.ceil(c["rows"] / batch) for c in cl["clients"])
            assert total_steps == expect, (
                f"round {rnd}: {total_steps} steps, expected {expect} (tail batch dropped?)")
            worst_step = expect
            assert len(cl["mu_next"]) == n_clients and len(cl["drift"]) == n_clients
            assert sum(cl["cluster_rows"]) == sum(c["rows"] for c in cl["clients"])

            st = torch.load(R / "resume" / f"round_{tag}.pt", weights_only=True)
            assert st["round"] == rnd, f"round {rnd}: resume state says round {st['round']}"
            assert st["fingerprint"] == cfg["fingerprint"], f"round {rnd}: fingerprint drift"
            # The resume file is what the next session trains from. If its mu disagrees
            # with the committed client log, the continuation silently runs different
            # mathematics from the run it claims to continue.
            st_mu = {int(k): float(v) for k, v in st["mu"].items()}
            assert set(st_mu) == set(mu_ref), f"round {rnd}: resume mu covers the wrong clients"
            for i, v in st_mu.items():
                assert abs(v - mu_ref[i]) < 1e-12, (
                    f"round {rnd}: resume/round_{tag}.pt mu[{i}]={v} disagrees with the "
                    f"committed mu_next {mu_ref[i]}")
        return (f"{len(check_rounds)} round(s) checked: weights reload strict and every "
                f"tensor finite, metrics match confusion (max|d|={worst_metric:.1e}), "
                f"per_class matches confusion, predictions cover all {n_test:,} test rows, "
                f"{worst_step:,} steps == sum ceil(n_i/{batch}), client ids exactly "
                f"0..{n_clients-1} with config row counts, lr on the cosine schedule, "
                f"mu chained from drift and matching the resume state, seeds match the "
                f"(round, client) formula")
    check("per-round artifacts", rounds_ok)

    for line in ok:
        print("PASS  " + line)
    for line in fail:
        print("FAIL  " + line)
    print(f"\n{len(ok)} passed, {len(fail)} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

"""End-to-end on REAL VeReMi rows: prepack -> 2 pFedES rounds -> per-client metrics ->
crash recovery -> next-session resume.

Small enough for the 8 GB WSL box (4 truncated clients of the 100-client partition, CPU,
no compile), but it runs the same proj/ modules the Kaggle notebook runs, with partial
participation (K = 2 of 4) so non-selected clients are evaluated and carried unchanged.
It proves the pipeline, not the speed.
"""
import csv, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papers/pfedes-yi-2025"))
from proj.model import CFG, build_model, build_proxy, N_PARAMS_MODEL, N_PARAMS_PROXY
from proj.data import assert_fp16_safe
from proj.metrics import METRIC_KEYS
from proj.pfedes import select_clients, lr_at
from proj import ckpt as C
from proj import driver as D
from proj.verify import verify_run

TRAIN = Path("/home/odixe/nckh/veremi/dataset/fl_client/alpha05/100_client/train")
TEST = Path("/home/odixe/nckh/veremi/dataset/centralized/test")
META = json.load(open(ROOT / "knowledge/meta.json"))
SCALER = json.load(open(ROOT / "knowledge/scaler.json"))["features"]
FEATS = META["feature_cols"]
ROWS_PER_CLIENT, TEST_ROWS = 12_000, 30_000


def head(dataset, cols, n):
    # imported here, not at module level: spawned workers re-import this module, and
    # pyarrow alone is ~150 MiB of RSS per process on the 8 GB WSL box
    import pyarrow as pa
    import pyarrow.dataset as ds
    out, got = [], 0
    for b in ds.dataset(str(dataset), format="parquet").to_batches(columns=cols,
                                                                  batch_size=8192):
        out.append(b); got += b.num_rows
        if got >= n: break
    return pa.Table.from_batches(out).slice(0, n)


def prepack(cache, cids):
    xs, ys, spans, off = [], [], {}, 0
    for cid in cids:
        t = head(TRAIN / f"client_id={cid:03d}", FEATS + ["label"], ROWS_PER_CLIENT)
        a = np.empty((t.num_rows, 66), np.float16)
        for j, c in enumerate(FEATS):
            a[:, j] = t.column(c).to_numpy(zero_copy_only=False)
        xs.append(a); ys.append(t.column("label").to_numpy(zero_copy_only=False).astype(np.uint8))
        spans[cid] = (off, off + t.num_rows); off += t.num_rows
    X = np.concatenate(xs); Y = np.concatenate(ys)
    probe = X[:, FEATS.index("f_rcv_x_rel")].astype(np.float32).mean()
    assert abs(probe) < 50, f"train does not look standardized: {probe}"
    t = head(TEST, FEATS + ["label"], TEST_ROWS)
    raw_spd = t.column("f_snd_spd").to_numpy(zero_copy_only=False).mean()
    assert raw_spd > 5.0, "test looks pre-scaled - do not scale it twice"
    TX = np.empty((t.num_rows, 66), np.float16)
    for j, c in enumerate(FEATS):
        v = np.nan_to_num(t.column(c).to_numpy(zero_copy_only=False).astype(np.float64))
        TX[:, j] = (v - SCALER[c]["mean"]) / SCALER[c]["std_used"]
    TY = t.column("label").to_numpy(zero_copy_only=False).astype(np.uint8)
    print(f"   train mean(f_rcv_x_rel)={probe:.3f} (pre-scaled) | test raw mean(f_snd_spd)="
          f"{raw_spd:.4f} (raw) | fp16 max|x| train={assert_fp16_safe(X,'train'):.1f} "
          f"test={assert_fp16_safe(TX,'test'):.1f}")
    for n, a in (("train_X.f16", X), ("train_y.u8", Y), ("test_X.f16", TX), ("test_y.u8", TY)):
        np.save(cache / f"{n}.npy", a)
    return spans, len(TY)


def make_cfg(cache, n_clients, n_test, rounds, world_size=1):
    return {**CFG, "n_features": 66, "device": "cpu", "world_size": world_size,
            "compile": False, "batch": 256, "eval_batch": 4096,
            "lr": 1e-3, "lr_schedule": "cosine", "lr_min": 1e-5,
            "weight_decay": 1e-4, "mu": 0.5, "clip": 1.0, "seed": 42,
            "rounds": rounds, "local_epochs": 1, "proxy_epochs": 1, "participation": 0.5,
            "data_id": "smoke0000smoke00", "n_clients": n_clients, "n_test": n_test,
            "cache": str(cache), "run_name": "smoke", "max_seconds": 3600,
            "require_resume": False, "max_skips_per_client": 16,
            "finalize_reserve_seconds": 0, "preds_rounds": [3], "eval_all_every": 100}


def _mk(d):
    for s in C.SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d


def main():
    tmp = Path(tempfile.mkdtemp()); cache = tmp / "cache"; cache.mkdir()
    cids = [0, 1, 2, 3]
    spans, n_test = prepack(cache, cids)
    # The plan is 4 rounds from the start: `rounds` is in the fingerprint (the per-round
    # cosine schedule spans the run), so every session of one run must plan the same T.
    cfg = dict(make_cfg(cache, len(cids), n_test, rounds=4), preds_rounds=[4])
    C.run_dir = lambda name, base=None, _t=tmp: _mk(_t / "runs" / name)
    D.write_manifest(cfg, META["class_names"], spans, y_true_src=cache / "test_y.u8.npy",
                     extra={"n_test": n_test, "content_id": "smoke_content"})
    hist = D.run(cfg, spans, META["class_names"])
    assert len(hist) == 4, hist
    d = tmp / "runs" / "smoke"
    fp = C.fingerprint(cfg)
    N = len(cids)

    for r in (1, 2, 3, 4):
        mj = json.loads((d / "metrics" / f"round_{r:03d}.json").read_text())
        for k in METRIC_KEYS:
            for suf in ("", "_std", "_min", "_max"):
                assert k + suf in mj, f"round {r}: no {k}{suf}"
        assert len(mj["clients"]) == N and all(len(c["per_class"]) == 16 for c in mj["clients"])
        sel = select_clients(N, cfg["participation"], cfg["seed"], r)
        assert [c["cid"] for c in mj["clients"] if c["selected"]] == sel
        assert mj["selected"] == 2
        # round 1 and the last round evaluate everyone; round 2 only the 2 trained clients
        partial = r in (2, 3)             # r1: cache empty; r4: last round + preds
        assert mj["evaluated"] == (2 if partial else N) and mj["cache_mismatch"] == 0, mj["evaluated"]
        assert [c["cid"] for c in mj["clients"] if c["evaluated"]] == (sel if partial else list(range(N)))
        assert abs(mj["lr"] - lr_at(cfg, r)) < 1e-15 and all(abs(c["lr"] - lr_at(cfg, r)) < 1e-15 for c in
                                                            json.loads((d / "logs" / f"round_{r:03d}.json").read_text())["clients"])
        cm = np.load(d / "confusion" / f"round_{r:03d}.npy")
        assert cm.shape == (N, 16, 16) and (cm.sum(axis=(1, 2)) == n_test).all()
        for sub, name in (("weights", f"round_{r:03d}.pt"), ("resume", f"round_{r:03d}.pt"),
                          ("logs", f"round_{r:03d}.json"), ("complete", f"round_{r:03d}.done")):
            assert (d / sub / name).exists(), f"round {r}: no {sub}/{name}"
        lg = json.loads((d / "logs" / f"round_{r:03d}.json").read_text())
        assert lg["selected"] == sel and [c["cid"] for c in lg["clients"]] == sel
        assert all(c["applied_w"] > 0 and c["applied_theta"] > 0 for c in lg["clients"])
    assert (d / "preds" / "round_004.u8.npy").exists() and not (d / "preds" / "round_003.u8.npy").exists()
    assert np.load(d / "preds" / "round_004.u8.npy", mmap_mode="r").shape == (N, n_test)
    # carried-forward matrices are the previous round's, exactly; re-evaluated ones at r4 agree
    cm1, cm2 = (np.load(d / "confusion" / f"round_{r:03d}.npy") for r in (1, 2))
    sel2 = select_clients(N, cfg["participation"], cfg["seed"], 2)
    for c in range(N):
        if c not in sel2:                 # a trained client's CM MAY coincide on a tiny fixture
            assert np.array_equal(cm1[c], cm2[c]), c
    print("   eval cache: unselected clients carry the previous confusion; re-eval at r4 matched")
    print(f"   round 1 f1_macro mean={hist[0]['f1_macro']:.6f} | round 4 mean="
          f"{hist[3]['f1_macro']:.6f} std={hist[3]['f1_macro_std']:.4f} "
          f"loss_w={hist[3]['loss_w_client_mean']:.4f} loss_theta={hist[3]['loss_theta_client_mean']:.4f} "
          f"| lr per round {[round(h['lr'], 7) for h in hist]}")
    h = hist[-1]
    assert abs(h["precision_micro"] - h["accuracy"]) < 1e-12 and abs(h["f1_micro"] - h["accuracy"]) < 1e-12

    # non-selected clients carry their previous weights unchanged; selected ones moved
    w1 = torch.load(d / "weights/round_001.pt", map_location="cpu", weights_only=True)
    w2 = torch.load(d / "weights/round_002.pt", map_location="cpu", weights_only=True)
    sel2 = select_clients(N, cfg["participation"], cfg["seed"], 2)
    for c in range(N):
        same = all(torch.equal(w1["clients"][c][k], w2["clients"][c][k]) for k in w1["clients"][c])
        assert same == (c not in sel2), f"client {c}: selected={c in sel2} unchanged={same}"
    assert not all(torch.equal(w1["global"][k], w2["global"][k]) for k in w1["global"])
    assert set(w2) == {"round", "global", "clients", "cfg", "fingerprint", "prev_sha", "metrics"}
    assert w2["prev_sha"] == C.file_sha(d / "weights/round_001.pt")
    print("   non-selected clients unchanged across the round, selected ones and theta moved")

    # weights file is weights-only and rebuilds G and every F_k with strict=True
    G, Fs, meta = C.load_weights(d / "weights/round_004.pt", build_model, build_proxy,
                                 N_PARAMS_MODEL, N_PARAMS_PROXY)
    assert len(Fs) == N and meta["round"] == 4
    print(f"   round-4 weights rebuild: G {sum(p.numel() for p in G.parameters()):,} + "
          f"{N} x {N_PARAMS_MODEL:,}, weights_only=True, "
          f"{(d/'weights/round_004.pt').stat().st_size/1e6:.2f} MB")

    ok, lines = verify_run(d, cfg=cfg, build_model=build_model, build_proxy=build_proxy,
                           expect_model=N_PARAMS_MODEL, expect_proxy=N_PARAMS_PROXY,
                           y_true_path=d / "reports" / "y_true.u8.npy",
                           require_rounds=4, full=True)
    assert ok, "\n".join(lines)
    print("   verify_run: per-client metrics == confusion == json == csv; preds rebuild the CMs")
    rows = list(csv.DictReader(open(d / "history.csv")))
    assert [int(r["round"]) for r in rows] == [1, 2, 3, 4], rows
    crows = list(csv.DictReader(open(d / "clients.csv")))
    assert len(crows) == 4 * N

    def wdiff(a, b):
        m = max((a["clients"][c][k].float() - b["clients"][c][k].float()).abs().max().item()
                for c in range(N) for k in a["clients"][c])
        return max(m, max((a["global"][k].float() - b["global"][k].float()).abs().max().item()
                          for k in a["global"]))

    # ---- crash recovery: drop round 4's marker, resume must redo round 4 IDENTICALLY
    w4 = torch.load(d / "weights/round_004.pt", map_location="cpu", weights_only=True)
    (d / "complete" / "round_004.done").unlink()
    assert C.last_complete_round(d, fp) == 3
    hist2 = D.run(cfg, spans, META["class_names"])
    assert len(hist2) == 1 and hist2[0]["round"] == 4, hist2
    w4b = torch.load(d / "weights/round_004.pt", map_location="cpu", weights_only=True)
    dmax = wdiff(w4, w4b)
    assert dmax == 0.0, f"replayed round 4 differs by {dmax}"
    assert abs(hist2[0]["f1_macro"] - hist[3]["f1_macro"]) < 1e-12
    rows = list(csv.DictReader(open(d / "history.csv")))
    assert [int(r["round"]) for r in rows] == [1, 2, 3, 4], "duplicate history row after redo"
    assert len(list(csv.DictReader(open(d / "clients.csv")))) == 4 * N
    print(f"   crash recovery: round 4 replayed bit-identically, history still rounds 1..4")

    # ---- next session: the session boundary is emulated by removing round 4 entirely (as
    #      if the budget had stopped session 1 after round 3); the continuation must plan the
    #      SAME rounds=4 (fingerprint), rebuild the eval cache from round 3's confusion file,
    #      and reproduce round 4 bit-identically -- a continuation is not a different run.
    for sub, name in (("weights", "round_004.pt"), ("resume", "round_004.pt"),
                      ("confusion", "round_004.npy"), ("metrics", "round_004.json"),
                      ("logs", "round_004.json"), ("preds", "round_004.u8.npy")):
        (d / sub / name).unlink()
    assert C.rebuild_history(d, fp) == 3
    cfg3 = dict(cfg, require_resume=True)
    hist3 = D.run(cfg3, spans, META["class_names"])
    assert len(hist3) == 1 and hist3[0]["round"] == 4 and hist3[0]["cache_mismatch"] == 0
    assert C.last_complete_round(d, fp) == 4
    w4c = torch.load(d / "weights/round_004.pt", map_location="cpu", weights_only=True)
    assert wdiff(w4, w4c) == 0.0, "continuation session's round 4 differs from the uninterrupted run"
    ok, lines = verify_run(d, cfg=cfg3, build_model=build_model, build_proxy=build_proxy,
                           expect_model=N_PARAMS_MODEL, expect_proxy=N_PARAMS_PROXY,
                           y_true_path=d / "reports" / "y_true.u8.npy",
                           require_rounds=4, full=True)
    assert ok, "\n".join(lines)
    print("   resume: next session re-ran round 4 on top of the committed round 3 bit-identically; "
          "verify passes 1..4")
    # a continuation that plans a different horizon is a different schedule: refused at the gate
    try:
        D.run(dict(cfg3, rounds=5), spans, META["class_names"])
        raise AssertionError("a rounds=5 continuation resumed a rounds=4 run")
    except SystemExit:
        print("   resume with a different `rounds`: fingerprint mismatch, SystemExit at the gate")

    # ---- a "patch" session: every round already committed -> the driver trains nothing,
    #      changes nothing on disk, and the verifier still passes (the 50c/100c re-verify
    #      sessions of 2026-09-12 take this path)
    before = {p.name: p.stat().st_size for p in (d / "weights").iterdir()}
    hist4 = D.run(cfg3, spans, META["class_names"])
    assert hist4 == [] and C.last_complete_round(d, fp) == 4
    assert {p.name: p.stat().st_size for p in (d / "weights").iterdir()} == before
    rows = list(csv.DictReader(open(d / "history.csv")))
    assert [int(r["round"]) for r in rows] == [1, 2, 3, 4], rows
    ok, lines = verify_run(d, cfg=cfg3, build_model=build_model, build_proxy=build_proxy,
                           expect_model=N_PARAMS_MODEL, expect_proxy=N_PARAMS_PROXY,
                           y_true_path=d / "reports" / "y_true.u8.npy",
                           require_rounds=4, full=True)
    assert ok, "\n".join(lines)
    print("   nothing-to-do session: 0 rounds trained, artifacts untouched, verify passes 1..4")

    # ---- require_resume with nothing to resume must die before training
    shutil.rmtree(tmp / "runs")
    try:
        D.run(dict(cfg, require_resume=True), spans, META["class_names"])
        raise AssertionError("require_resume did not stop an empty run")
    except SystemExit:
        print("   require_resume with no checkpoint: SystemExit at the gate")

    shutil.rmtree(tmp)
    print("\nSMOKE TEST PASSED (real VeReMi rows, 4 clients, K=2, 4 rounds + replay + resume, 1 worker)")


# mp start method is "spawn": the child re-imports this module, so main() must not run
# at import time or every worker restarts the whole test.
if __name__ == "__main__":
    main()

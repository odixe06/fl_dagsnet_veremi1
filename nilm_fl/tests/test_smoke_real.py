"""End-to-end on REAL VeReMi rows: prepack -> 3 mutual-learning rounds -> per-client and
proxy metrics -> crash recovery -> next-session resume -> nothing-to-do session.

Small enough for the 8 GB WSL box (4 truncated clients of the 100-client partition, CPU,
no compile), but it runs the same proj/ modules the Kaggle notebook runs. It proves the
pipeline, not the speed.
"""
import csv, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papers/nilm-li-2024"))
from proj.model import CFG, build_model, N_PARAMS
from proj.data import assert_fp16_safe
from proj.metrics import METRIC_KEYS
from proj.nilm import lr_at, ACC_KEYS
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
            "weight_decay": 1e-4, "clip": 1.0, "seed": 42,
            "rounds": rounds, "local_epochs": 1,
            "data_id": "smoke0000smoke00", "n_clients": n_clients, "n_test": n_test,
            "cache": str(cache), "run_name": "smoke", "max_seconds": 3600,
            "require_resume": False, "max_skips_per_client": 16,
            "finalize_reserve_seconds": 0, "preds_rounds": [rounds]}


def _mk(d):
    for s in C.SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d


def main():
    tmp = Path(tempfile.mkdtemp()); cache = tmp / "cache"; cache.mkdir()
    cids = [0, 1, 2, 3]
    spans, n_test = prepack(cache, cids)
    # The plan is 3 rounds from the start: `rounds` is in the fingerprint (the per-round
    # cosine schedule spans the run), so every session of one run must plan the same T.
    R = 3
    cfg = make_cfg(cache, len(cids), n_test, rounds=R)
    C.run_dir = lambda name, base=None, _t=tmp: _mk(_t / "runs" / name)
    D.write_manifest(cfg, META["class_names"], spans, y_true_src=cache / "test_y.u8.npy",
                     extra={"n_test": n_test, "content_id": "smoke_content"})
    hist = D.run(cfg, spans, META["class_names"])
    assert len(hist) == R, hist
    d = tmp / "runs" / "smoke"
    fp = C.fingerprint(cfg)
    N = len(cids)

    for r in range(1, R + 1):
        mj = json.loads((d / "metrics" / f"round_{r:03d}.json").read_text())
        for k in METRIC_KEYS:
            for suf in ("", "_std", "_min", "_max"):
                assert k + suf in mj, f"round {r}: no {k}{suf}"
            assert f"global_{k}" in mj and k in mj["global"], f"round {r}: no proxy {k}"
        assert len(mj["clients"]) == N and all(len(c["per_class"]) == 16 for c in mj["clients"])
        assert len(mj["global"]["per_class"]) == 16
        assert mj["evaluated"] == N
        lg = json.loads((d / "logs" / f"round_{r:03d}.json").read_text())
        assert [c["cid"] for c in lg["clients"]] == list(range(N))
        assert abs(mj["lr"] - lr_at(cfg, r)) < 1e-15 and all(abs(c["lr"] - lr_at(cfg, r)) < 1e-15
                                                            for c in lg["clients"])
        assert all(c["applied"] > 0 and c["nonfinite"] == 0 for c in lg["clients"])
        assert all(np.isfinite([c[k] for k in ACC_KEYS]).all() for c in lg["clients"])
        cm = np.load(d / "confusion" / f"round_{r:03d}.npy")
        gcm = np.load(d / "confusion" / f"global_{r:03d}.npy")
        assert cm.shape == (N, 16, 16) and (cm.sum(axis=(1, 2)) == n_test).all()
        assert gcm.shape == (16, 16) and gcm.sum() == n_test
        for sub, name in (("weights", f"round_{r:03d}.pt"), ("resume", f"round_{r:03d}.pt"),
                          ("logs", f"round_{r:03d}.json"), ("complete", f"round_{r:03d}.done")):
            assert (d / sub / name).exists(), f"round {r}: no {sub}/{name}"
    assert (d / "preds" / f"round_{R:03d}.u8.npy").exists() and (d / "preds" / f"global_{R:03d}.u8.npy").exists()
    assert not (d / "preds" / "round_001.u8.npy").exists()
    assert np.load(d / "preds" / f"round_{R:03d}.u8.npy", mmap_mode="r").shape == (N, n_test)
    print(f"   round 1 f1_macro mean={hist[0]['f1_macro']:.6f} proxy={hist[0]['global_f1_macro']:.6f} | "
          f"round {R} mean={hist[-1]['f1_macro']:.6f} std={hist[-1]['f1_macro_std']:.4f} "
          f"proxy={hist[-1]['global_f1_macro']:.6f} | ce_s={hist[-1]['ce_s_client_mean']:.4f} "
          f"ce_r={hist[-1]['ce_r_client_mean']:.4f} kl_s={hist[-1]['kl_s_client_mean']:.4f} "
          f"kl_r={hist[-1]['kl_r_client_mean']:.4f} | lr per round {[round(h['lr'], 7) for h in hist]}")
    h = hist[-1]
    assert abs(h["precision_micro"] - h["accuracy"]) < 1e-12 and abs(h["f1_micro"] - h["accuracy"]) < 1e-12

    # every client's w_s and the proxy move every round; the proxy is the MEAN of the
    # per-client proxies (checked structurally: same init => round-1 proxies equal the
    # personalized models only up to dropout noise, so no equality is claimed there)
    w1 = torch.load(d / "weights/round_001.pt", map_location="cpu", weights_only=True)
    w2 = torch.load(d / "weights/round_002.pt", map_location="cpu", weights_only=True)
    for c in range(N):
        assert not all(torch.equal(w1["clients"][c][k], w2["clients"][c][k]) for k in w1["clients"][c]), c
    assert not all(torch.equal(w1["global"][k], w2["global"][k]) for k in w1["global"])
    assert set(w2) == {"round", "global", "clients", "cfg", "fingerprint", "prev_sha", "metrics",
                       "global_metrics"}
    assert w2["prev_sha"] == C.file_sha(d / "weights/round_001.pt")
    # the N personalized models differ from each other after round 1 (different data)
    assert not torch.equal(w1["clients"][0]["head.5.weight"], w1["clients"][1]["head.5.weight"])
    print("   every w_s and the proxy moved each round; clients differ; weights file has the 8 keys")

    # weights file is weights-only and rebuilds the proxy and every w_s with strict=True
    Rm, Ss, meta = C.load_weights(d / f"weights/round_{R:03d}.pt", build_model, N_PARAMS)
    assert len(Ss) == N and meta["round"] == R
    print(f"   round-{R} weights rebuild: proxy + {N} x {N_PARAMS:,}, weights_only=True, "
          f"{(d/f'weights/round_{R:03d}.pt').stat().st_size/1e6:.2f} MB")

    ok, lines = verify_run(d, cfg=cfg, build_model=build_model, expect_params=N_PARAMS,
                           y_true_path=d / "reports" / "y_true.u8.npy",
                           require_rounds=R, full=True)
    assert ok, "\n".join(lines)
    print("   verify_run: per-client + proxy metrics == confusion == json == csv; preds rebuild the CMs")
    rows = list(csv.DictReader(open(d / "history.csv")))
    assert [int(r["round"]) for r in rows] == list(range(1, R + 1)), rows
    assert "global_f1_macro" in rows[0]
    crows = list(csv.DictReader(open(d / "clients.csv")))
    assert len(crows) == R * N

    def wdiff(a, b):
        m = max((a["clients"][c][k].float() - b["clients"][c][k].float()).abs().max().item()
                for c in range(N) for k in a["clients"][c])
        return max(m, max((a["global"][k].float() - b["global"][k].float()).abs().max().item()
                          for k in a["global"]))

    # ---- crash recovery: drop the last marker, resume must redo the round IDENTICALLY
    wl = torch.load(d / f"weights/round_{R:03d}.pt", map_location="cpu", weights_only=True)
    (d / "complete" / f"round_{R:03d}.done").unlink()
    assert C.last_complete_round(d, fp) == R - 1
    hist2 = D.run(cfg, spans, META["class_names"])
    assert len(hist2) == 1 and hist2[0]["round"] == R, hist2
    wlb = torch.load(d / f"weights/round_{R:03d}.pt", map_location="cpu", weights_only=True)
    dmax = wdiff(wl, wlb)
    assert dmax == 0.0, f"replayed round {R} differs by {dmax}"
    assert abs(hist2[0]["f1_macro"] - hist[-1]["f1_macro"]) < 1e-12
    assert abs(hist2[0]["global_f1_macro"] - hist[-1]["global_f1_macro"]) < 1e-12
    rows = list(csv.DictReader(open(d / "history.csv")))
    assert [int(r["round"]) for r in rows] == list(range(1, R + 1)), "duplicate history row after redo"
    assert len(list(csv.DictReader(open(d / "clients.csv")))) == R * N
    print(f"   crash recovery: round {R} replayed bit-identically, history still rounds 1..{R}")

    # ---- next session: emulated by removing the last round entirely (as if the budget had
    #      stopped session 1 one round earlier); the continuation must plan the SAME
    #      rounds (fingerprint) and reproduce the round bit-identically.
    for sub, name in (("weights", f"round_{R:03d}.pt"), ("resume", f"round_{R:03d}.pt"),
                      ("confusion", f"round_{R:03d}.npy"), ("confusion", f"global_{R:03d}.npy"),
                      ("metrics", f"round_{R:03d}.json"), ("logs", f"round_{R:03d}.json"),
                      ("preds", f"round_{R:03d}.u8.npy"), ("preds", f"global_{R:03d}.u8.npy")):
        (d / sub / name).unlink()
    assert C.rebuild_history(d, fp) == R - 1
    cfg3 = dict(cfg, require_resume=True)
    hist3 = D.run(cfg3, spans, META["class_names"])
    assert len(hist3) == 1 and hist3[0]["round"] == R
    assert C.last_complete_round(d, fp) == R
    wlc = torch.load(d / f"weights/round_{R:03d}.pt", map_location="cpu", weights_only=True)
    assert wdiff(wl, wlc) == 0.0, "continuation session's round differs from the uninterrupted run"
    ok, lines = verify_run(d, cfg=cfg3, build_model=build_model, expect_params=N_PARAMS,
                           y_true_path=d / "reports" / "y_true.u8.npy",
                           require_rounds=R, full=True)
    assert ok, "\n".join(lines)
    print(f"   resume: next session re-ran round {R} on top of round {R-1} bit-identically; "
          f"verify passes 1..{R}")
    # a continuation that plans a different horizon is a different schedule: refused at the gate
    try:
        D.run(dict(cfg3, rounds=R + 1), spans, META["class_names"])
        raise AssertionError(f"a rounds={R+1} continuation resumed a rounds={R} run")
    except SystemExit:
        print("   resume with a different `rounds`: fingerprint mismatch, SystemExit at the gate")

    # ---- a "patch" session: every round already committed -> the driver trains nothing
    before = {p.name: p.stat().st_size for p in (d / "weights").iterdir()}
    hist4 = D.run(cfg3, spans, META["class_names"])
    assert hist4 == [] and C.last_complete_round(d, fp) == R
    assert {p.name: p.stat().st_size for p in (d / "weights").iterdir()} == before
    rows = list(csv.DictReader(open(d / "history.csv")))
    assert [int(r["round"]) for r in rows] == list(range(1, R + 1)), rows
    ok, lines = verify_run(d, cfg=cfg3, build_model=build_model, expect_params=N_PARAMS,
                           y_true_path=d / "reports" / "y_true.u8.npy",
                           require_rounds=R, full=True)
    assert ok, "\n".join(lines)
    print(f"   nothing-to-do session: 0 rounds trained, artifacts untouched, verify passes 1..{R}")

    # ---- require_resume with nothing to resume must die before training
    shutil.rmtree(tmp / "runs")
    try:
        D.run(dict(cfg, require_resume=True), spans, META["class_names"])
        raise AssertionError("require_resume did not stop an empty run")
    except SystemExit:
        print("   require_resume with no checkpoint: SystemExit at the gate")

    shutil.rmtree(tmp)
    print(f"\nSMOKE TEST PASSED (real VeReMi rows, 4 clients, {R} rounds + replay + resume, 1 worker)")


# mp start method is "spawn": the child re-imports this module, so main() must not run
# at import time or every worker restarts the whole test.
if __name__ == "__main__":
    main()

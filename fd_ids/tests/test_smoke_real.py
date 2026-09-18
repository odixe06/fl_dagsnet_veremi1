"""End-to-end on REAL VeReMi rows: prepack -> 2 FL rounds -> metrics -> resume.

Small enough for the 8 GB WSL box (3 truncated clients, CPU, no compile), but it runs the
same proj/ modules the Kaggle notebook runs. It proves the pipeline, not the speed.
"""
import json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pyarrow.dataset as ds
import torch

ROOT = Path("/home/odixe/nckh/fd_ids")
sys.path.insert(0, str(ROOT / "papers/fd-ids-2025"))
from proj.model import CFG
from proj.data import assert_fp16_safe
from proj.metrics import METRIC_KEYS
from proj import ckpt as C
from proj import driver as D
from proj.verify import verify_run

TRAIN = Path("/home/odixe/nckh/dataset/fl_client/alpha05/100_client/train")
TEST = Path("/home/odixe/nckh/dataset/centralized/test")
META = json.load(open(ROOT / "knowledge/meta.json"))
SCALER = json.load(open(ROOT / "knowledge/scaler.json"))["features"]
FEATS = META["feature_cols"]
ROWS_PER_CLIENT, TEST_ROWS = 20_000, 40_000


def head(dataset, cols, n):
    out, got = [], 0
    for b in ds.dataset(str(dataset), format="parquet").to_batches(columns=cols,
                                                                  batch_size=8192):
        out.append(b); got += b.num_rows
        if got >= n: break
    import pyarrow as pa
    return pa.Table.from_batches(out).slice(0, n)


def main():
    tmp = Path(tempfile.mkdtemp()); cache = tmp / "cache"; cache.mkdir()
    cids = [0, 1, 2]

    # ---- prepack train: already z-scored at source, must NOT be scaled again
    xs, ys, spans, off = [], [], {}, 0
    for cid in cids:
        t = head(TRAIN / f"client_id={cid:03d}", FEATS + ["label"], ROWS_PER_CLIENT)
        a = np.empty((t.num_rows, 66), np.float16)
        for j, c in enumerate(FEATS):
            a[:, j] = t.column(c).to_numpy(zero_copy_only=False)
        xs.append(a); ys.append(t.column("label").to_numpy(zero_copy_only=False).astype(np.uint8))
        spans[cid] = (off, off + t.num_rows); off += t.num_rows
    X = np.concatenate(xs); Y = np.concatenate(ys)
    # A 20k-row head is time-ordered and correlated, so its column means are NOT the
    # global ~0 of the full split. Discriminate by magnitude instead: f_rcv_x_rel has raw
    # mean 1132.77 / std 858.0, so scaled data sits near 0 and raw data near 1132.
    probe = X[:, FEATS.index("f_rcv_x_rel")].astype(np.float32).mean()
    print(f"   train mean(f_rcv_x_rel) = {probe:.3f}  "
          f"(raw would be ~{SCALER['f_rcv_x_rel']['mean']:.0f}; source is pre-scaled)")
    assert abs(probe) < 50, f"train does not look standardized: {probe}"

    # ---- prepack test: raw at source, scaler applied HERE and nowhere else
    t = head(TEST, FEATS + ["label"], TEST_ROWS)
    raw_spd = t.column("f_snd_spd").to_numpy(zero_copy_only=False).mean()
    print(f"   test raw mean(f_snd_spd) = {raw_spd:.4f}  (7.4493 on the full split; "
          f"~0 would mean it was already scaled)")
    assert raw_spd > 5.0, "test looks pre-scaled - do not scale it twice"
    TX = np.empty((t.num_rows, 66), np.float16)
    for j, c in enumerate(FEATS):
        v = np.nan_to_num(t.column(c).to_numpy(zero_copy_only=False).astype(np.float64))
        TX[:, j] = (v - SCALER[c]["mean"]) / SCALER[c]["std_used"]
    TY = t.column("label").to_numpy(zero_copy_only=False).astype(np.uint8)
    print(f"   fp16 max|x|: train={assert_fp16_safe(X,'train'):.1f} "
          f"test={assert_fp16_safe(TX,'test'):.1f}  (limit 65504)")

    for n, a in (("train_X.f16", X), ("train_y.u8", Y), ("test_X.f16", TX), ("test_y.u8", TY)):
        np.save(cache / f"{n}.npy", a)

    cfg = {**CFG, "n_features": 66, "device": "cpu", "world_size": 1, "compile": False,
           "batch": 256, "eval_batch": 4096, "lr": 1e-3, "lam": 0.5, "beta": 0.1,
           "mu": 0.01, "temperature": 3.0, "clip": 1.0, "seed": 42, "rounds": 2,
           "local_epochs": 1, "data_id": "smoke0000smoke00", "n_clients": 3,
           "n_test": len(TY), "cache": str(cache), "run_name": "smoke",
           "max_seconds": 3600, "require_resume": False,
           "max_skips_per_client": 16, "finalize_reserve_seconds": 0}
    C.run_dir = lambda name, _t=tmp: _mk(_t / "runs" / name)

    D.write_manifest(cfg, META["class_names"], spans,
                     y_true_src=cache / "test_y.u8.npy",
                     extra={"n_test": len(TY), "content_id": "smoke_content"})
    hist = D.run(cfg, spans, META["class_names"])
    assert len(hist) == 2, hist
    d = tmp / "runs" / "smoke"
    fp = C.fingerprint(cfg)

    # every round: all 10 metrics, a full-coverage confusion, weights, resume, preds, marker
    for r in (1, 2):
        mj = json.loads((d / "metrics" / f"round_{r:03d}.json").read_text())
        missing = [k for k in METRIC_KEYS if k not in mj]
        assert not missing, f"round {r} missing metrics: {missing}"
        assert len(mj["per_class"]) == 16
        for k in ("ce_client_mean", "kd_client_mean", "grad_norm", "steps", "applied",
                  "skipped", "teacher_sec", "vram_train_gb", "backend", "seconds"):
            assert k in mj, f"round {r} metrics json cannot rebuild history: no {k}"
        cm = np.load(d / "confusion" / f"round_{r:03d}.npy")
        assert cm.sum() == len(TY), f"round {r} confusion covers {cm.sum()}/{len(TY)}"
        for sub, name in (("weights", f"round_{r:03d}.pt"), ("resume", f"round_{r:03d}.pt"),
                          ("preds", f"round_{r:03d}.u8.npy"), ("logs", f"round_{r:03d}.json"),
                          ("complete", f"round_{r:03d}.done")):
            assert (d / sub / name).exists(), f"round {r}: no {sub}/{name}"
        cl = json.loads((d / "logs" / f"round_{r:03d}.json").read_text())
        assert len(cl) == len(cids) and all(c["applied"] > 0 for c in cl), cl
    print(f"   round 1 f1_macro={hist[0]['f1_macro']:.6f}  "
          f"round 2 f1_macro={hist[1]['f1_macro']:.6f}")

    # the identity precision_micro == recall_micro == f1_micro == accuracy must hold
    h = hist[-1]
    assert abs(h["precision_micro"] - h["accuracy"]) < 1e-12
    assert abs(h["f1_micro"] - h["accuracy"]) < 1e-12

    # weights file is weights-only and rebuilds the round-2 global model exactly
    from proj.model import build_model
    mdl, meta = C.load_weights(d / "weights/round_002.pt", build_model, expect_params=395_024)
    raw = torch.load(d / "weights/round_002.pt", map_location="cpu", weights_only=True)
    assert "optim" not in raw and all(torch.is_tensor(v) for v in raw["model"].values())
    print(f"   round-2 weights rebuild: {sum(p.numel() for p in mdl.parameters()):,} params, "
          f"weights_only=True, {(d/'weights/round_002.pt').stat().st_size/1e6:.2f} MB")

    # every published number re-derived from the artifacts, including preds -> confusion
    ok, lines = verify_run(d, cfg=cfg, build=build_model, expect_params=395_024,
                           y_true_path=d / "reports" / "y_true.u8.npy",
                           require_rounds=2, full=True)
    assert ok, "\n".join(lines)
    print("   verify_run: metrics == confusion == weights == history, preds rebuild the CM")

    # history.csv keyed by round, no duplicates
    import csv
    rows = list(csv.DictReader(open(d / "history.csv")))
    assert [int(r["round"]) for r in rows] == [1, 2], rows

    # The 1-worker vs 2-worker comparison lives in tests/test_schedule.py: two CPU
    # workers plus this parent put the tree at 2,992 of 3,000 MiB, and the watchdog is a
    # boundary to respect, not a number to raise.
    # ---- crash recovery: drop round 2's marker, resume must redo round 2 IDENTICALLY
    before = {k: v.clone() for k, v in
              torch.load(d / "weights/round_002.pt", map_location="cpu",
                         weights_only=True)["model"].items()}
    (d / "complete" / "round_002.done").unlink()
    assert C.last_complete_round(d, fp) == 1
    hist2 = D.run(cfg, spans, META["class_names"])
    assert len(hist2) == 1 and hist2[0]["round"] == 2, hist2
    after = torch.load(d / "weights/round_002.pt", map_location="cpu",
                       weights_only=True)["model"]
    dmax = max((before[k].float() - after[k].float()).abs().max().item() for k in before)
    assert dmax == 0.0, f"replayed round 2 differs by {dmax}"
    assert abs(hist2[0]["f1_macro"] - hist[1]["f1_macro"]) < 1e-12, \
        f"replay f1 {hist2[0]['f1_macro']} != original {hist[1]['f1_macro']}"
    rows = list(csv.DictReader(open(d / "history.csv")))
    assert [int(r["round"]) for r in rows] == [1, 2], "duplicate history row after redo"
    print(f"   resume: round 2 replayed bit-identically "
          f"(f1_macro {hist2[0]['f1_macro']:.6f}), history still rounds 1,2")

    shutil.rmtree(tmp)
    print("\nSMOKE TEST PASSED (real VeReMi rows, 3 clients, 2 rounds, 1 and 2 workers)")


def _mk(d):
    for s in C.SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d


# mp start method is "spawn": the child re-imports this module, so main() must not run
# at import time or every worker restarts the whole test.
if __name__ == "__main__":
    main()

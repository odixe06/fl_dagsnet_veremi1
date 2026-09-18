"""A verifier that only ever says "pass" is indistinguishable from one that checks nothing.

Every tamper below was accepted by an earlier version. The two that matter most are subtle:
a metric overwritten with NaN slips through any check written as `abs(a - b) > tol`, because
that comparison is False for NaN; and a per-class block can be entirely fabricated while its
`support` column stays right, since support comes straight from the confusion row sums.
"""
import csv, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/fd-ids-2025"))
from proj.model import build_model, CFG
from proj.metrics import metrics_from_confusion, per_class_from_confusion
from proj import ckpt as C
from proj.verify import verify_run

K, NT, B = 16, 400, 64
CLIENTS = {0: 200, 1: 300, 2: 150}
NAMES = [f"c{i}" for i in range(K)]
CFGD = {**CFG, "n_features": 66, "lr": 1e-3, "lam": 0.5, "beta": 0.1, "mu": 0.01,
        "temperature": 3.0, "clip": 1.0, "n_clients": len(CLIENTS), "batch": B,
        "local_epochs": 1, "seed": 42, "data_id": "d" * 16, "run_name": "v",
        "rounds": 2, "eval_batch": 4096, "world_size": 1, "compile": False,
        "cache": "/tmp/x", "max_seconds": 60, "require_resume": False, "device": "cpu",
        "n_test": NT}


def build_fixture(root):
    C.run_dir = lambda name, _r=root: _mk(_r / "runs" / name)
    d = C.run_dir("v")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, K, NT, dtype=np.uint8)
    np.save(d / "reports" / "y_true.u8.npy", y_true)
    torch.manual_seed(0)
    m = build_model(CFGD)
    hist = []
    for r in (1, 2):
        with torch.no_grad():
            for p in m.parameters(): p.add_(torch.randn_like(p) * 0.01)
        y_pred = rng.integers(0, K, NT, dtype=np.uint8)
        cm = np.bincount(y_true.astype(np.int64) * K + y_pred.astype(np.int64),
                         minlength=K * K).reshape(K, K)
        met = metrics_from_confusion(cm)
        C.save_round_weights(m, r, CFGD, met, "v")
        np.save(d / "confusion" / f"round_{r:03d}.npy", cm)
        np.save(d / "preds" / f"round_{r:03d}.u8.npy", y_pred)
        logs = [{"round": r, "cid": c, "n_k": n, "rank": 0,
                 "steps": -(-n // B), "applied": -(-n // B), "skipped": 0, "nonfinite": 0,
                 "seed": 1, "ce": 1.0, "kd": 0.5, "gnorm": 0.3, "sec": 1.0,
                 "teacher_sec": 0.1, "vram_gb": 0.0} for c, n in sorted(CLIENTS.items())]
        (d / "logs" / f"round_{r:03d}.json").write_text(json.dumps(logs))
        row = {"round": r, **met, "ce_client_mean": 1.0, "kd_client_mean": 0.5,
               "grad_norm": 0.3, "steps": 12, "applied": 12, "skipped": 0,
               "teacher_sec": 0.3, "vram_train_gb": 0.0, "vram_eval_gb": 0.0,
               "backend": "eager", "seconds": 1.0}
        (d / "metrics" / f"round_{r:03d}.json").write_text(json.dumps(
            {**row, "per_class": per_class_from_confusion(cm, NAMES)}))
        C.append_history(d, row)
        C.mark_complete(d, r)
        hist.append(row)
    (d / "reports" / "manifest.json").write_text(json.dumps(
        {"fingerprint": C.fingerprint(CFGD), "cfg": CFGD, "class_names": NAMES,
         "n_clients": len(CLIENTS), "n_train": sum(CLIENTS.values()),
         "client_rows": {str(c): n for c, n in sorted(CLIENTS.items())},
         "content_id": "c" * 16, "data_id": "d" * 16}))
    return d


def _mk(d):
    for s in C.SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d


def check(d):
    return verify_run(d, cfg=CFGD, build=build_model, expect_params=395_024,
                      y_true_path=d / "reports" / "y_true.u8.npy", require_rounds=2,
                      full=True)


def case(name, mutate):
    with tempfile.TemporaryDirectory() as t:
        d = build_fixture(Path(t))
        mutate(d)
        ok, lines = check(d)
        assert not ok, f"{name}: verifier PASSED it\n" + "\n".join(lines)
        print(f"   rejected: {name}")


def drop_csv_row(d):
    rows = [r for r in csv.DictReader(open(d / "history.csv")) if int(r["round"]) != 2]
    C._write_csv(d / "history.csv", rows)


def dup_csv_row(d):
    rows = list(csv.DictReader(open(d / "history.csv")))
    C._write_csv(d / "history.csv", rows + [rows[0]])


def nan_metric(d):
    p = d / "metrics" / "round_001.json"
    j = json.loads(p.read_text()); j["accuracy"] = float("nan")
    p.write_text(json.dumps(j))


def fake_per_class(d):
    p = d / "metrics" / "round_001.json"
    j = json.loads(p.read_text())
    for e in j["per_class"]: e["f1"] = 999.0        # support left correct on purpose
    p.write_text(json.dumps(j))


def main():
    with tempfile.TemporaryDirectory() as t:
        d = build_fixture(Path(t))
        ok, lines = check(d)
        assert ok, "the untampered fixture must pass\n" + "\n".join(lines)
        print("   baseline passes:", lines[1].strip())

    case("history row for round 2 deleted", drop_csv_row)
    case("history row for round 1 duplicated", dup_csv_row)
    case("accuracy in metrics json set to NaN", nan_metric)
    case("per-class F1 = 999 with support left correct", fake_per_class)
    case("all predictions deleted",
         lambda d: [f.unlink() for f in (d / "preds").glob("*.npy")])
    case("all client logs deleted",
         lambda d: [f.unlink() for f in (d / "logs").glob("*.json")])
    case("resume file overwritten with garbage",
         lambda d: (d / "resume" / "round_001.pt").write_bytes(b"\x00" * 4096))
    case("predictions swapped for a different vector",
         lambda d: np.save(d / "preds" / "round_002.u8.npy",
                           np.random.default_rng(9).integers(0, K, NT, dtype=np.uint8)))
    case("a client log row claims the wrong step count",
         lambda d: (d / "logs" / "round_001.json").write_text(json.dumps(
             [{**e, "steps": 99} for e in
              json.loads((d / "logs" / "round_001.json").read_text())])))
    case("the manifest is missing",
         lambda d: (d / "reports" / "manifest.json").unlink())

    # cfg claiming a different test size must not be reconciled by comparing rounds to
    # each other -- they agree with one another and disagree with the config.
    with tempfile.TemporaryDirectory() as t:
        d = build_fixture(Path(t))
        ok, lines = verify_run(d, cfg={**CFGD, "n_test": 999}, build=build_model,
                               expect_params=395_024,
                               y_true_path=d / "reports" / "y_true.u8.npy",
                               require_rounds=2, full=True)
        assert not ok, "cfg n_test=999 accepted against a 400-row confusion matrix"
        print("   rejected: cfg declares n_test=999, artifacts hold 400")

    # an incomplete run must fail the completion gate even though every round is valid
    with tempfile.TemporaryDirectory() as t:
        d = build_fixture(Path(t))
        (d / "complete" / "round_002.done").unlink()
        # what the next session does on startup: history follows the verified range
        C.rebuild_history(d, C.fingerprint(CFGD))
        ok, _ = verify_run(d, cfg=CFGD, build=build_model, expect_params=395_024,
                           y_true_path=d / "reports" / "y_true.u8.npy",
                           require_rounds=2, full=True)
        assert not ok, "1 of 2 rounds certified as complete"
        ok2, _ = verify_run(d, cfg=CFGD, build=build_model, expect_params=395_024,
                            y_true_path=d / "reports" / "y_true.u8.npy", full=True)
        assert ok2, "a valid 1-round run should verify when completion is not required"
        print("   rejected: 1 of 2 rounds under --require-rounds; accepted without it")

    print("\nALL 13 VERIFIER CHECKS PASSED (1 baseline + 12 tampers rejected)")


if __name__ == "__main__":
    main()

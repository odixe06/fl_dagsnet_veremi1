"""Execute the generated PROBE notebook's cells, in order, against a fake /kaggle tree
built from real parquet rows (with the `.stats.json` sidecars Kaggle ships), on the local
GPU with one worker. Substitutions are listed in SUBS and nothing else is changed, so
this exercises the notebook's own flow: hardware cell, CFG, module writes, prepack with
cache + content_id, the calibration micro-benchmark, training, and the verifier cell."""
import json, shutil, sys, tempfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests.test_smoke_real as S

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "papers/nilm-li-2024/notebook/20c_probe/nilm_20c_probe.ipynb"
CIDS = [0, 1, 2, 3]
ROWS, TROWS = 2_500, 9_000


def fake_input(tmp):
    import pyarrow.parquet as pq
    fl = tmp / "input/datasets/odixe0502/veremi-fl-20client/20_client"
    for cid in CIDS:
        d = fl / "train" / f"client_id={cid:03d}"; d.mkdir(parents=True)
        t = S.head(S.TRAIN / f"client_id={cid:03d}", None, ROWS)
        pq.write_table(t, d / "part-0.parquet")
        (d / "part-0.stats.json").write_text("{}")             # Kaggle's sidecar
    cen = tmp / "input/datasets/odixe0502/veremi-nextgen2026-centralized/upload"
    (cen / "test").mkdir(parents=True)
    pq.write_table(S.head(S.TEST, None, TROWS), cen / "test/part-00000.parquet")
    (cen / "test/part-00000.stats.json").write_text("{}")
    shutil.copy(ROOT / "knowledge/scaler.json", cen / "scaler.json")


def main():
    # GPU mode runs the calibration cell (eager) and stops before training: a CUDA parent
    # plus a CUDA worker exceeds the 8 GB WSL watchdog. CPU mode skips the calibration
    # cell and runs training + verification. Together they cover every cell.
    gpu = "--gpu" in sys.argv
    assert torch.cuda.is_available() or not gpu
    tmp = Path(tempfile.mkdtemp())
    fake_input(tmp)
    (tmp / "working").mkdir(); (tmp / "temp").mkdir()
    nb = json.loads(NB.read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    n_train, n_test = ROWS * len(CIDS), TROWS
    SUBS = [  # (cell must contain, old, new) -- the ONLY edits made to the notebook text
        ("device_count()", "assert n == 2,", "assert n >= 0,"),
        ("device_count()", "assert cap == (7, 5)", "assert cap[0] >= 7"),
        ("device_count()", '"/kaggle/working', f'"{tmp}/working'),
        ("load_clients(", "43_045_415", str(n_train)),
        ("load_clients(", "10_761_343", str(n_test)),
    ]
    ns = {}
    for i, src in enumerate(cells):
        for need, old, new in SUBS:
            if need in src and not src.startswith("%%writefile"):
                assert old in src, (i, old); src = src.replace(old, new)
        if src.startswith("%%writefile"):
            first, _, body = src.partition("\n")
            dst = Path(first.split()[-1].replace("/kaggle/working", str(tmp / "working")))
            dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(body)
            continue
        if "import wandb" in src:
            ns["run"] = None; print(f"-- cell {i}: W&B skipped (run=None)"); continue
        if "PROBE ONLY" in src and not gpu:
            print(f"-- cell {i}: calibration skipped on CPU"); continue
        if "from proj.driver import" in src and gpu:
            print(f"-- cell {i}: training skipped in GPU mode (watchdog)"); break
        print(f"-- cell {i}: exec")
        if "CFG = dict(" in src:
            exec(src, ns)
            # fixture-sized overrides; every other CFG value is the notebook's own
            # compile=False here: Inductor in the parent plus a CUDA worker exceeds the
            # 8 GB WSL watchdog; the compiled paths are covered by tests/test_gpu_local.py
            ns["CFG"].update(n_clients=len(CIDS), batch=64, eval_batch=1024, world_size=1,
                             cache=str(tmp / "temp/veremi_cache"), max_seconds=3600,
                             compile=False, device="cuda" if gpu else "cpu")
            import proj.ckpt as C, proj.data as PD
            C.run_dir.__defaults__ = (str(tmp / "working/runs"),)
            PD.find_root.__defaults__ = ((str(tmp / "input"),),)
            continue
        exec(src, ns)
    d = tmp / "working/runs" / ns["CFG"]["run_name"]
    if gpu:
        cal = json.loads((d / "reports/calibration.json").read_text())
        assert cal["train_backend"] == "eager" and "eval_eager_folded_16384" in cal
        print(f"\ncalibration: {json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in cal.items()})}")
    else:
        assert len(ns["hist"]) == 2 and ns["ok"]
        for f in ("history.csv", "clients.csv", "reports/manifest.json", "reports/y_true.u8.npy",
                  "weights/round_002.pt", "confusion/round_002.npy", "confusion/global_002.npy",
                  "logs/round_002.json", "complete/round_002.done"):
            assert (d / f).exists(), f
    shutil.rmtree(tmp)
    print(f"\nNOTEBOOK SIMULATION PASSED ({'GPU: cells 0-12 incl. calibration' if gpu else 'CPU: all cells but calibration, 2 rounds + verify'})")


if __name__ == "__main__":
    main()

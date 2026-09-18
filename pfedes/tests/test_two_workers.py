"""Schedule independence: 2 CPU workers (dynamic LPT dispatch, eval split by client id)
must produce the SAME weights and metrics as 1 worker, bit for bit. Small fixture so two
worker processes plus the parent stay under the 8 GB WSL watchdog."""
import json, shutil, sys, tempfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/pfedes-yi-2025"))
import tests.test_smoke_real as S
from proj import ckpt as C
from proj import driver as D

S.ROWS_PER_CLIENT, S.TEST_ROWS = 2_000, 4_000


def run_with(world_size, tmp):
    cache = tmp / f"cache{world_size}"; cache.mkdir()
    cids = [0, 1, 2, 3]
    spans, n_test = S.prepack(cache, cids)
    cfg = S.make_cfg(cache, len(cids), n_test, rounds=1, world_size=world_size)
    cfg["participation"] = 1.0; cfg["run_name"] = f"w{world_size}"
    cfg["batch"], cfg["eval_batch"] = 128, 1024      # activations, not data, set the RSS here
    C.run_dir = lambda name, base=None, _t=tmp: S._mk(_t / "runs" / name)
    D.write_manifest(cfg, S.META["class_names"], spans, y_true_src=cache / "test_y.u8.npy",
                     extra={"n_test": n_test, "content_id": "smoke_content"})
    hist = D.run(cfg, spans, S.META["class_names"])
    assert len(hist) == 1
    return hist[0], torch.load(tmp / "runs" / cfg["run_name"] / "weights/round_001.pt",
                               map_location="cpu", weights_only=True)


def main():
    tmp = Path(tempfile.mkdtemp())
    h1, w1 = run_with(1, tmp)
    h2, w2 = run_with(2, tmp)
    dmax = 0.0
    for c in w1["clients"]:
        for k in w1["clients"][c]:
            dmax = max(dmax, (w1["clients"][c][k].float() - w2["clients"][c][k].float()).abs().max().item())
    for k in w1["global"]:
        dmax = max(dmax, (w1["global"][k].float() - w2["global"][k].float()).abs().max().item())
    assert dmax == 0.0, f"1 vs 2 workers differ by {dmax}"
    for k in ("f1_macro", "accuracy", "f1_macro_std", "loss_w_client_mean", "loss_theta_client_mean"):
        assert abs(h1[k] - h2[k]) < 1e-12, (k, h1[k], h2[k])
    shutil.rmtree(tmp)
    print(f"\n1 worker vs 2 workers: max|dweight| = {dmax}, f1_macro {h1['f1_macro']:.6f} == "
          f"{h2['f1_macro']:.6f}\nTWO-WORKER TEST PASSED")


if __name__ == "__main__":
    main()

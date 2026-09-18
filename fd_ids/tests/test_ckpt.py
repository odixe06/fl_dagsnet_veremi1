"""Weights-only checkpoints must rebuild the exact model that wrote them, and a round must
count as done only when every artifact of that round is actually on disk.

These exercise the path FD-IDS runs: save_round_weights + mark_complete + resolve_resume.
"""
import json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/fd-ids-2025"))
from proj.model import build_model, CFG
from proj import ckpt as C

CFGD = {**CFG, "n_features": 66, "lr": 1e-3, "lam": 0.5, "beta": 0.1, "mu": 0.01,
        "temperature": 3.0, "clip": 1.0, "n_clients": 20, "batch": 512,
        "local_epochs": 1, "seed": 42, "data_id": "deadbeefdeadbeef", "run_name": "t",
        "rounds": 50, "eval_batch": 16384, "world_size": 2, "compile": True,
        "cache": "/tmp/x", "max_seconds": 3600, "require_resume": False, "device": "cuda"}
MET = {k: 0.5 for k in ("accuracy", "precision_macro", "precision_micro",
                        "precision_weighted", "recall_macro", "recall_micro",
                        "recall_weighted", "f1_macro", "f1_micro", "f1_weighted")}


def commit(d, r, model, cfg=CFGD):
    """What driver.run does for one round, in the same order."""
    C.save_round_weights(model, r, cfg, MET, cfg["run_name"])
    np.save(d / "confusion" / f"round_{r:03d}.npy", np.eye(16, dtype=np.int64) * 100)
    (d / "metrics" / f"round_{r:03d}.json").write_text(
        json.dumps({"round": r, **MET, "seconds": 1.0, "per_class": []}))
    C.append_history(d, {"round": r, **MET, "seconds": 1.0})
    C.mark_complete(d, r)


def main():
    tmp = Path(tempfile.mkdtemp())
    C.run_dir = lambda name, _t=tmp: _mk(_t / "runs" / name)
    d = C.run_dir("t")

    torch.manual_seed(42)
    model = build_model(CFGD)
    x = torch.randn(64, 66)
    model.train()
    torch.nn.functional.cross_entropy(model(x), torch.randint(0, 16, (64,))).backward()
    torch.optim.Adam(model.parameters(), lr=1e-3).step()
    model.eval()
    with torch.no_grad(): ref = model(x).clone()

    # 1. weights + resume land BEFORE the marker; the marker is the caller's last act
    C.save_round_weights(model, 7, CFGD, MET, "t")
    assert (d / "weights/round_007.pt").exists() and (d / "resume/round_007.pt").exists()
    assert not (d / "complete/round_007.done").exists(), "marker written by the writer"

    # 2. the weights file carries weights ONLY - no optimizer, no module object
    raw = torch.load(d / "weights/round_007.pt", map_location="cpu", weights_only=True)
    assert set(raw) == {"round", "model", "cfg", "fingerprint", "prev_sha",
                        "metrics"}, sorted(raw)
    assert raw["prev_sha"] is None, "round 7 written into an empty tree has no parent"
    assert all(torch.is_tensor(v) for v in raw["model"].values()), "non-tensor in weights"
    assert "optim" not in raw and "scaler" not in raw
    print(f"   weights {(d/'weights/round_007.pt').stat().st_size/1e6:.2f} MB   "
          f"resume {(d/'resume/round_007.pt').stat().st_size/1e6:.2f} MB")

    # 3. BatchNorm buffers are present - strict=True would catch their absence
    assert any("running_mean" in k for k in raw["model"]), "BN buffers filtered out"
    assert any("num_batches_tracked" in k for k in raw["model"])

    # 4. the resume bundle also loads weights_only=True (numpy RNG converted to a tensor)
    r = torch.load(d / "resume/round_007.pt", map_location="cpu", weights_only=True)
    assert r["round"] == 7 and "optim" not in r

    # 5. THE REQUIREMENT: rebuild the model from weights alone, bit-identical logits
    rebuilt, meta = C.load_weights(d / "weights/round_007.pt", build_model,
                                   expect_params=395_024)
    with torch.no_grad(): got = rebuilt(x)
    delta = (got - ref).abs().max().item()
    assert delta == 0.0 and meta["round"] == 7, delta
    print(f"   rebuilt from weights: max|delta logits| = {delta}")

    # 6. a cfg that drifted must be caught, not silently loaded
    try:
        C.load_weights(d / "weights/round_007.pt", build_model, expect_params=999)
        raise AssertionError("wrong parameter count accepted")
    except RuntimeError as e:
        assert "expected" in str(e)

    # 7. a diverged tensor must not load: argmax turns NaN logits into ordinary labels
    bad = dict(raw); bad["model"] = dict(raw["model"])
    k0 = next(k for k, v in bad["model"].items() if v.is_floating_point())
    bad["model"][k0] = bad["model"][k0].clone(); bad["model"][k0].view(-1)[0] = float("nan")
    C.atomic_save(bad, d / "weights/bad.pt")
    try:
        C.load_weights(d / "weights/bad.pt", build_model)
        raise AssertionError("NaN weights accepted")
    except RuntimeError as e:
        assert "non-finite" in str(e)
    (d / "weights/bad.pt").unlink()

    # 8. RNG round-trip
    C.set_rng_state(r["rng"])
    import random
    a = (np.random.rand(), random.random(), torch.rand(1).item())
    C.set_rng_state(r["rng"])
    assert a == (np.random.rand(), random.random(), torch.rand(1).item())

    # 9. a marker whose artifacts are missing must NOT count. This is the failure the
    #    old import produced every time: copy order put `complete` third of eight.
    (d / "weights/round_007.pt").unlink(); (d / "resume/round_007.pt").unlink()
    fp = C.fingerprint(CFGD)
    commit(d, 1, model); commit(d, 2, model); commit(d, 3, model)
    assert C.last_complete_round(d, fp) == 3
    (d / "metrics/round_003.json").unlink()
    assert C.last_complete_round(d, fp) == 2, "marker trusted without its metrics"
    (d / "metrics" / "round_003.json").write_text(
        json.dumps({"round": 3, **MET, "seconds": 1.0, "per_class": []}))

    # 10. a gap ends the run: round 5 is meaningless without round 4 before it
    commit(d, 5, model)
    assert sorted(C._markers(d)) == [1, 2, 3, 5]
    assert C.last_complete_round(d, fp) == 3, "largest marker used despite a gap at 4"
    for s in ("weights", "resume", "metrics", "confusion", "complete"):
        for f in (d / s).glob("round_005.*"): f.unlink()

    # 11. the fingerprint must move for every scientific knob and stand still for the
    #     operational ones. The old key list saw 5 of these and missed all 13.
    base = C.fingerprint(CFGD)
    for k, v in (("lr", 3e-4), ("mu", 0.02), ("beta", 0.2), ("lam", 0.7),
                 ("temperature", 4.0), ("clip", 0.5), ("seed", 1), ("local_epochs", 2),
                 ("batch", 256), ("dropout", 0.2), ("n_clients", 50),
                 ("num_classes", 2), ("n_features", 30), ("data_id", "0" * 16)):
        assert C.fingerprint({**CFGD, k: v}) != base, f"fingerprint blind to {k}"
    for k, v in (("rounds", 80), ("cache", "/other"), ("world_size", 1),
                 ("compile", False), ("max_seconds", 60), ("eval_batch", 4096)):
        assert C.fingerprint({**CFGD, k: v}) == base, f"fingerprint over-tight on {k}"
    try:
        C.fingerprint({k: v for k, v in CFGD.items() if k != "data_id"})
        raise AssertionError("missing fingerprint key silently defaulted")
    except KeyError as e:
        assert "data_id" in str(e)
    print(f"   fingerprint: moves on 14 scientific keys, stable on 6 operational ones")

    # 12. history is derived. Truncate it, delete it - a rebuild restores every round.
    import csv
    (d / "history.csv").write_text("round,f1_macro\n1,0.5\n")     # a crash mid-rewrite
    assert C.rebuild_history(d, fp) == 3
    rows = list(csv.DictReader(open(d / "history.csv")))
    assert [int(x["round"]) for x in rows] == [1, 2, 3], rows
    assert all("seconds" in x for x in rows), "rebuilt history lost a column"

    # 13. import through staging: publish per round, marker last, idempotent on retry.
    #     A source tree is written under its OWN run_name -- run_name is part of the
    #     fingerprint, so a tree copied from another run is correctly refused.
    cfgu = {**CFGD, "run_name": "u"}
    work = C.run_dir("u")
    for rr in (1, 2, 3): commit(work, rr, model, cfgu)
    srcu = tmp / "input" / "prev" / "runs" / "u"
    for sub in C.SUBDIRS: (srcu / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("weights", "resume", "metrics", "confusion", "complete"):
        for f in (work / sub).iterdir(): shutil.copyfile(f, srcu / sub / f.name)
    shutil.rmtree(work)                  # /kaggle/working is wiped when a session starts
    work = C.run_dir("u")
    fpu = C.fingerprint(cfgu)

    real_mark = C.mark_complete
    def die_on_3(dd, rr):
        if rr == 3: raise OSError("disk full")
        real_mark(dd, rr)
    C.mark_complete = die_on_3
    try:
        C.resolve_resume("u", cfgu, attached=tmp / "input")
        raise AssertionError("the injected failure did not fire")
    except OSError:
        pass
    finally:
        C.mark_complete = real_mark
    assert C.last_complete_round(work, fpu) == 2, "partial import was trusted"
    assert (work / "weights/round_003.pt").exists(), "round 3 files should be staged in"
    assert C.resolve_resume("u", cfgu, attached=tmp / "input") == 3, "retry did not finish"
    assert [int(x["round"]) for x in csv.DictReader(open(work / "history.csv"))] == [1, 2, 3], \
        "imported run did not get its history back"
    print("   import: interrupted at round 3 -> 2 trusted; retry completes to 3 with history")

    # 14. a source whose scientific config disagrees must import NOTHING, not resume on
    #     top of it. The old fingerprint saw 5 keys and lr was not one of them.
    cfgv = {**CFGD, "run_name": "v"}
    workv = C.run_dir("v")
    for rr in (1, 2): commit(workv, rr, model, cfgv)
    srcv = tmp / "input2" / "prev" / "runs" / "v"
    for sub in C.SUBDIRS: (srcv / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("weights", "resume", "metrics", "confusion", "complete"):
        for f in (workv / sub).iterdir(): shutil.copyfile(f, srcv / sub / f.name)
    shutil.rmtree(workv); C.run_dir("v")
    assert C.resolve_resume("v", {**cfgv, "lr": 9e-9}, attached=tmp / "input2") is None, \
        "weights trained at a different lr were accepted as a resume point"
    assert C.resolve_resume("v", cfgv, attached=tmp / "input2") == 2, "matching cfg refused"
    print("   import: a changed lr imports nothing; the unchanged cfg still resumes")

    shutil.rmtree(tmp)
    print("\nALL 14 CHECKS PASSED")


def _mk(d):
    for s in C.SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    main()

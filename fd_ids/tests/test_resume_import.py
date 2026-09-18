"""An import must not invent a completion the source never claimed, and must not splice
two training histories into one.

Two runs of the same config share a fingerprint. That is what makes the fingerprint a
config check and not a run identity, and it is why round N carries the hash of the file
round N-1 was read from: same config, different history, different bytes, broken chain.
"""
import json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/fd-ids-2025"))
from proj.model import build_model, CFG
from proj import ckpt as C

CFGD = {**CFG, "n_features": 66, "lr": 1e-3, "lam": 0.5, "beta": 0.1, "mu": 0.01,
        "temperature": 3.0, "clip": 1.0, "n_clients": 3, "batch": 64, "local_epochs": 1,
        "seed": 42, "data_id": "d" * 16, "run_name": "r", "rounds": 10, "eval_batch": 4096,
        "world_size": 1, "compile": False, "cache": "/tmp/x", "max_seconds": 60,
        "require_resume": False, "device": "cpu"}
MET = {k: 0.5 for k in ("accuracy", "precision_macro", "precision_micro",
                        "precision_weighted", "recall_macro", "recall_micro",
                        "recall_weighted", "f1_macro", "f1_micro", "f1_weighted")}
TMP = None


def mk(d):
    for s in C.SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d


def commit(name, r, model, cfg, mark=True):
    d = C.run_dir(name)
    C.save_round_weights(model, r, cfg, MET, name)
    np.save(d / "confusion" / f"round_{r:03d}.npy", np.eye(16, dtype=np.int64) * 100)
    (d / "metrics" / f"round_{r:03d}.json").write_text(
        json.dumps({"round": r, **MET, "seconds": 1.0, "per_class": []}))
    np.save(d / "preds" / f"round_{r:03d}.u8.npy", np.zeros(1600, np.uint8))
    (d / "logs" / f"round_{r:03d}.json").write_text("[]")
    if mark: C.mark_complete(d, r)
    return d


def publish(name, rounds, seed, mark_upto=None, cfg=None):
    """Train-and-commit `rounds` rounds under `name`, then move the tree to an input mount."""
    cfg = cfg or {**CFGD, "run_name": name}
    torch.manual_seed(seed)
    m = build_model(cfg)
    d = None
    for r in range(1, rounds + 1):
        with torch.no_grad():                       # a different history every seed
            for p in m.parameters(): p.add_(torch.randn_like(p) * 0.01)
        d = commit(name, r, m, cfg, mark=(mark_upto is None or r <= mark_upto))
    src = TMP / f"input_{name}_{seed}" / "prev" / "runs" / name
    for s in C.SUBDIRS: (src / s).mkdir(parents=True, exist_ok=True)
    for s in C.SUBDIRS:
        for f in (d / s).iterdir():
            if f.is_file(): shutil.copyfile(f, src / s / f.name)
    shutil.rmtree(d); C.run_dir(name)
    return src.parents[2], d


def main():
    global TMP
    TMP = Path(tempfile.mkdtemp())
    C.run_dir = lambda name, _t=TMP: mk(_t / "runs" / name)

    # 1. The source committed one round and crashed before marking the second. Importing
    #    round 2 would promote a round the source itself does not consider done.
    mount, d = publish("a", 2, seed=1, mark_upto=1, cfg={**CFGD, "run_name": "a"})
    got = C.resolve_resume("a", {**CFGD, "run_name": "a"}, attached=mount)
    assert got == 1, f"source committed 1 round, import returned {got}"
    assert not (d / "complete" / "round_002.done").exists(), "invented a marker"
    print("   source marked 1 of 2 rounds -> imported exactly 1")

    # 2. Working holds run A's round 1; the mount holds run B's rounds 1-2. Same cfg, same
    #    fingerprint, different weights. Keeping A's round 1 and taking B's round 2 makes a
    #    history that never happened.
    cfgb = {**CFGD, "run_name": "b"}
    mount_b, _ = publish("b", 2, seed=7, cfg=cfgb)          # run B on the mount
    torch.manual_seed(99)                                    # run A in working/
    ma = build_model(cfgb)
    with torch.no_grad():
        for p in ma.parameters(): p.add_(torch.randn_like(p) * 0.5)
    commit("b", 1, ma, cfgb)
    try:
        C.resolve_resume("b", cfgb, attached=mount_b)
        raise AssertionError("two different histories were spliced")
    except RuntimeError as e:
        assert "different training runs" in str(e), e
    print("   working=A r1 + mount=B r1..2, same fingerprint -> refused, not spliced")

    # 3. Same source imported into an empty tree still works, and the chain holds.
    shutil.rmtree(TMP / "runs" / "b"); C.run_dir("b")
    assert C.resolve_resume("b", cfgb, attached=mount_b) == 2
    print("   the same mount into an empty tree imports both rounds")

    # 4. A broken chain must not be trusted even inside one tree: swap round 2's weights
    #    for a file trained from something else.
    db = TMP / "runs" / "b"
    shutil.copyfile(TMP / "input_b_7" / "prev" / "runs" / "b" / "weights/round_001.pt",
                    db / "weights/round_002.pt")
    assert C.last_complete_round(db, C.fingerprint(cfgb)) == 1, "broken chain accepted"
    print("   round 2 replaced by a file from elsewhere -> chain check stops at 1")

    # 5. A gap ends the range, and history follows the range rather than the markers.
    cfgc = {**CFGD, "run_name": "c"}
    torch.manual_seed(3); mc = build_model(cfgc)
    for r in (1, 2, 3): commit("c", r, mc, cfgc)
    dc = TMP / "runs" / "c"
    for s in ("weights", "resume", "metrics", "confusion", "complete"):
        for f in (dc / s).glob("round_002.*"): f.unlink()
    assert C.last_complete_round(dc, C.fingerprint(cfgc)) == 1
    assert C.rebuild_history(dc, C.fingerprint(cfgc)) == 1, "history kept a row after a gap"
    rounds = [int(r.split(",")[0]) for r in
              (dc / "history.csv").read_text().splitlines()[1:]]
    assert rounds == [1], rounds
    print("   gap at round 2 -> last_complete=1 and history holds only round 1")

    # 6. When nothing is valid any more, the stale CSV must go with it.
    for s in ("weights", "resume", "metrics", "confusion", "complete"):
        for f in (dc / s).glob("round_001.*"): f.unlink()
    assert C.rebuild_history(dc, C.fingerprint(cfgc)) == 0
    assert not (dc / "history.csv").exists(), "a CSV outlived every round it described"
    print("   no valid rounds left -> history.csv removed, not left behind")

    # 7. A corrupt resume file fails the round: existence and size are not readability.
    cfgd = {**CFGD, "run_name": "e"}
    torch.manual_seed(5); me = build_model(cfgd)
    commit("e", 1, me, cfgd)
    de = TMP / "runs" / "e"
    good = C.last_complete_round(de, C.fingerprint(cfgd))
    (de / "resume" / "round_001.pt").write_bytes(b"\x00" * 4096)
    assert good == 1 and C.last_complete_round(de, C.fingerprint(cfgd)) is None
    print("   resume file overwritten with garbage -> round no longer counts")

    shutil.rmtree(TMP)
    print("\nALL 7 IMPORT/CHAIN CHECKS PASSED")


if __name__ == "__main__":
    main()

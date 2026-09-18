"""Checkpoint contract, resume import, and the verifier's tamper sensitivity.

Fixture: a tiny 3-client run (2 rounds, K = 2) written with the real proj/ckpt.py writers
and real metrics from synthetic confusion matrices. Every tamper case below once passed a
verifier in a sibling project; each must FAIL here.
"""
import copy, json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/pfedes-yi-2025"))
from proj.model import build_model, build_proxy, CFG, N_PARAMS_MODEL, N_PARAMS_PROXY
from proj.metrics import metrics_from_confusion, per_class_from_confusion, METRIC_KEYS
from proj.pfedes import select_clients, expected_steps, lr_at
from proj.verify import verify_run
from proj import ckpt as C

N, NT = 3, 1600
CFGD = {**CFG, "n_features": 66, "lr": 1e-3, "lr_schedule": "cosine", "lr_min": 1e-5,
        "weight_decay": 1e-4, "mu": 0.5, "clip": 1.0,
        "n_clients": N, "participation": 0.67, "batch": 64, "local_epochs": 1,
        "proxy_epochs": 1, "seed": 42, "data_id": "d" * 16, "run_name": "r", "rounds": 2,
        "eval_batch": 4096, "world_size": 1, "compile": False, "cache": "/tmp/x",
        "max_seconds": 60, "require_resume": False, "device": "cpu", "n_test": NT,
        "preds_rounds": [2]}
CLASSES = [f"c{i}" for i in range(16)]
Y_TRUE = np.repeat(np.arange(16, dtype=np.uint8), NT // 16)


def mk(d):
    for s in C.SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d


def write_run(d, cfg, seed=0, rounds=2, mark_upto=None):
    """A run written with the production writers. Confusion matrices come from random
    predictions so the per-client metrics are real numbers, not constants."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    G, Fs = build_proxy(cfg), {c: build_model(cfg) for c in range(N)}
    prev_cm, prev_preds = None, None
    for r in range(1, rounds + 1):
        sel = select_clients(N, cfg["participation"], cfg["seed"], r)
        with torch.no_grad():
            for p in G.parameters(): p.add_(torch.randn_like(p) * 0.01)
            for c in sel:
                for p in Fs[c].parameters(): p.add_(torch.randn_like(p) * 0.01)
        cm = np.zeros((N, 16, 16), np.int64)
        preds = np.zeros((N, NT), np.uint8)
        for c in range(N):
            if r > 1 and c not in sel:
                cm[c], preds[c] = prev_cm[c], prev_preds[c]; continue
            yp = np.where(rng.random(NT) < 0.6, Y_TRUE, rng.integers(0, 16, NT)).astype(np.uint8)
            preds[c] = yp
            cm[c] = np.bincount(Y_TRUE.astype(int) * 16 + yp, minlength=256).reshape(16, 16)
        full = r == 1 or r == rounds
        per_client = [{"cid": c, "selected": c in sel, "evaluated": full or c in sel,
                       **metrics_from_confusion(cm[c]),
                       "per_class": per_class_from_confusion(cm[c], CLASSES)} for c in range(N)]
        row = {"round": r, "selected": len(sel), "evaluated": N if full else len(sel),
               "cache_mismatch": 0}
        for k in METRIC_KEYS:
            v = np.array([pc[k] for pc in per_client])
            row[k], row[f"{k}_std"], row[f"{k}_min"], row[f"{k}_max"] = \
                float(v.mean()), float(v.std()), float(v.min()), float(v.max())
        row.update(loss_w_client_mean=0.5, loss_theta_client_mean=0.3, seconds=1.0)
        C.save_round_weights(C.cpu_sd(G), {c: C.cpu_sd(Fs[c]) for c in range(N)}, r, cfg,
                             {k: row[k] for k in METRIC_KEYS}, d)
        C.atomic_np_save(d / "confusion" / f"round_{r:03d}.npy", cm)
        if r in cfg["preds_rounds"]:
            C.atomic_np_save(d / "preds" / f"round_{r:03d}.u8.npy", preds)
        logs = []
        for c in sel:
            s1, s2 = expected_steps(100, cfg)
            logs.append({"cid": c, "n_k": 100, "lr": lr_at(cfg, r),
                         "steps_w": s1, "applied_w": s1, "skipped_w": 0,
                         "nonfinite_w": 0, "steps_theta": s2, "applied_theta": s2,
                         "skipped_theta": 0, "nonfinite_theta": 0, "loss_w": 0.5,
                         "ce_orig": 0.4, "gnorm_w": 1.0, "loss_theta": 0.3, "gnorm_theta": 1.0})
        (d / "logs" / f"round_{r:03d}.json").write_text(json.dumps({"selected": sel, "clients": logs}))
        (d / "metrics" / f"round_{r:03d}.json").write_text(json.dumps({**row, "clients": per_client}))
        C.append_history(d, row, [{"round": r, **{k: v for k, v in pc.items() if k != "per_class"}}
                                  for pc in per_client])
        if mark_upto is None or r <= mark_upto:
            C.mark_complete(d, r)
        prev_cm, prev_preds = cm, preds
    (d / "reports").mkdir(exist_ok=True)
    (d / "reports" / "manifest.json").write_text(json.dumps(
        {"cfg": cfg, "fingerprint": C.fingerprint(cfg), "n_clients": N,
         "client_rows": {str(c): 100 for c in range(N)}}))
    with open(d / "reports" / "y_true.u8.npy", "wb") as f: np.save(f, Y_TRUE)
    return d


def check(d, cfg, require=2):
    return verify_run(d, cfg=cfg, build_model=build_model, build_proxy=build_proxy,
                      expect_model=N_PARAMS_MODEL, expect_proxy=N_PARAMS_PROXY,
                      y_true_path=d / "reports" / "y_true.u8.npy", require_rounds=require)


def main():
    tmp = Path(tempfile.mkdtemp()); n = 0
    C.run_dir = lambda name, base=None, _t=tmp: mk(_t / "runs" / name)
    cfg = dict(CFGD)
    d = write_run(mk(tmp / "runs/r"), cfg)
    fp = C.fingerprint(cfg)
    ok, lines = check(d, cfg); assert ok, "\n".join(lines)
    print("1. clean fixture passes the verifier"); n += 1

    # ---- weights contract
    raw = torch.load(d / "weights/round_002.pt", map_location="cpu", weights_only=True)
    assert set(raw) == {"round", "global", "clients", "cfg", "fingerprint", "prev_sha", "metrics"}
    assert all(torch.is_tensor(v) for v in raw["global"].values())
    assert all(torch.is_tensor(v) for c in raw["clients"].values() for v in c.values())
    assert "running_mean" in " ".join(raw["global"]) and any("num_batches_tracked" in k for k in raw["global"])
    rs = torch.load(d / "resume/round_002.pt", map_location="cpu", weights_only=True)
    assert rs["round"] == 2 and "rng" in rs
    G, Fs, ck = C.load_weights(d / "weights/round_002.pt", build_model, build_proxy,
                               N_PARAMS_MODEL, N_PARAMS_PROXY)
    assert len(Fs) == N and not G.training and not any(m.training for m in Fs.values())
    G1, _, _ = C.load_weights(d / "weights/round_002.pt", build_model, build_proxy, clients=[1])
    x = torch.randn(4, 66)
    assert torch.equal(G(x), G1(x))
    print(f"2. weights: tensors only, BN buffers included, weights_only=True, strict rebuild of "
          f"G + {N} F_k, {(d/'weights/round_002.pt').stat().st_size/1e6:.2f} MB"); n += 1

    # ---- fingerprint sees every scientific key, is blind to operational ones
    for k, v in (("lr", 0.01), ("mu", 0.3), ("participation", 1.0), ("proxy_epochs", 2),
                 ("weight_decay", 0.0), ("batch", 128), ("seed", 1), ("n_clients", 5),
                 ("data_id", "x" * 16), ("run_name", "other"), ("dropout", 0.2),
                 # the per-round LR schedule spans the run: its shape, floor and horizon
                 ("lr_schedule", "constant"), ("lr_min", 1e-6), ("rounds", 99)):
        assert C.fingerprint({**cfg, k: v}) != fp, k
    for k, v in (("eval_batch", 1), ("max_seconds", 1), ("compile", True),
                 ("world_size", 2), ("eval_all_every", 3), ("preds_rounds", [])):
        assert C.fingerprint({**cfg, k: v}) == fp, k
    try: C.fingerprint({k: v for k, v in cfg.items() if k != "mu"}); raise AssertionError
    except KeyError: pass
    print("3. fingerprint: 14 scientific keys move it, 6 operational keys do not, missing key raises"); n += 1

    # ---- tamper cases: each must FAIL
    def case(label, mutate):
        nonlocal n
        e = mk(tmp / "runs" / f"t_{n}")
        shutil.copytree(d, e, dirs_exist_ok=True)
        mutate(e)
        ok, lines = check(e, cfg)
        assert not ok, f"{label}: verifier passed a tampered run"
        print(f"4. tamper rejected: {label} -> {[l for l in lines if 'FAIL' in l][0].strip()[:90]}"); n += 1

    def m_nan(e):
        j = json.loads((e / "metrics/round_002.json").read_text()); j["clients"][0]["f1_macro"] = float("nan")
        (e / "metrics/round_002.json").write_text(json.dumps(j))
    def m_mean(e):
        j = json.loads((e / "metrics/round_002.json").read_text()); j["f1_macro"] = 0.999
        (e / "metrics/round_002.json").write_text(json.dumps(j))
    def m_csv(e):
        rows = (e / "history.csv").read_text().splitlines(); (e / "history.csv").write_text("\n".join(rows[:2]) + "\n")
    def m_ccsv(e):
        rows = (e / "clients.csv").read_text().splitlines()
        col = rows[0].split(",").index("accuracy"); f = rows[1].split(","); f[col] = "0.999"
        rows[1] = ",".join(f); (e / "clients.csv").write_text("\n".join(rows) + "\n")
    def m_pc(e):
        j = json.loads((e / "metrics/round_002.json").read_text()); j["clients"][2]["per_class"][3]["f1"] = 999
        (e / "metrics/round_002.json").write_text(json.dumps(j))
    def m_carry(e):
        cm = np.load(e / "confusion/round_002.npy"); j = json.loads((e / "metrics/round_002.json").read_text())
        # pretend client 0 was carried forward while its matrix changed
        j["clients"][0]["evaluated"] = False; j["clients"][0]["selected"] = False; j["evaluated"] -= 1
        cm[0, 0, 0] += 1; cm[0, 0, 1] -= 1
        C.atomic_np_save(e / "confusion/round_002.npy", cm)
        for c in j["clients"]:
            if c["cid"] == 0: c.update(metrics_from_confusion(cm[0])); c["per_class"] = per_class_from_confusion(cm[0], CLASSES)
        (e / "metrics/round_002.json").write_text(json.dumps(j))
    def m_sel(e):
        lg = json.loads((e / "logs/round_002.json").read_text()); lg["selected"] = lg["selected"][:1]; lg["clients"] = lg["clients"][:1]
        (e / "logs/round_002.json").write_text(json.dumps(lg))
    def m_steps(e):
        lg = json.loads((e / "logs/round_002.json").read_text()); lg["clients"][0]["applied_w"] = 0; lg["clients"][0]["skipped_w"] = lg["clients"][0]["steps_w"]
        (e / "logs/round_002.json").write_text(json.dumps(lg))
    def m_preds(e):
        p = np.load(e / "preds/round_002.u8.npy"); p[1, :50] = (p[1, :50] + 1) % 16
        C.atomic_np_save(e / "preds/round_002.u8.npy", p)
    def m_nopreds(e): (e / "preds/round_002.u8.npy").unlink()
    def m_nologs(e): (e / "logs/round_002.json").unlink()
    def m_lr(e):
        # a client trained at the constant peak rate in a round whose cosine value is lower
        lg = json.loads((e / "logs/round_002.json").read_text()); lg["clients"][0]["lr"] = cfg["lr"]
        (e / "logs/round_002.json").write_text(json.dumps(lg))
    def m_weight_nan(e):
        w = torch.load(e / "weights/round_002.pt", map_location="cpu", weights_only=True)
        k = next(iter(w["clients"][1])); w["clients"][1][k] = w["clients"][1][k].clone(); w["clients"][1][k].view(-1)[0] = float("nan")
        C.atomic_save(w, e / "weights/round_002.pt")
    def m_short(e): (e / "complete/round_002.done").unlink()
    def m_mm_json(e):
        j = json.loads((e / "metrics/round_002.json").read_text()); j["cache_mismatch"] = 1
        (e / "metrics/round_002.json").write_text(json.dumps(j))
    def m_mm_cm(e):
        # round 2 is a full-eval round: the unselected client was re-evaluated and must
        # reproduce round 1's matrix. Change it while keeping totals and the json consistent.
        cm = np.load(e / "confusion/round_002.npy"); j = json.loads((e / "metrics/round_002.json").read_text())
        u = [c["cid"] for c in j["clients"] if not c["selected"]][0]
        # move every off-diagonal row onto the diagonal: several hundred rows > CACHE_TOL_ROWS
        off = cm[u].copy(); np.fill_diagonal(off, 0); assert off.sum() > 100, off.sum()
        cm[u] = np.diag(cm[u].sum(axis=1))
        C.atomic_np_save(e / "confusion/round_002.npy", cm)
        for c in j["clients"]:
            if c["cid"] == u: c.update(metrics_from_confusion(cm[u])); c["per_class"] = per_class_from_confusion(cm[u], CLASSES)
        (e / "metrics/round_002.json").write_text(json.dumps(j))
    for label, fn in (("NaN per-client metric", m_nan), ("mean overwritten", m_mean),
                      ("history.csv row dropped", m_csv), ("clients.csv value edited", m_ccsv),
                      ("per-class f1 = 999", m_pc), ("carried-forward CM changed", m_carry),
                      ("client dropped from selection log", m_sel),
                      ("client applied 0 steps", m_steps), ("predictions edited", m_preds),
                      ("claimed predictions missing", m_nopreds), ("client log missing", m_nologs),
                      ("client trained at a rate off the schedule", m_lr),
                      ("NaN in a client weight", m_weight_nan), ("run shorter than required", m_short),
                      ("json cache_mismatch = 1 with identical matrices", m_mm_json),
                      ("re-evaluated unselected client's CM moved hundreds of rows", m_mm_cm)):
        case(label, fn)

    # ---- a cross-GPU-sized deviation (5 rows) with a consistent json count is tolerated
    e = mk(tmp / "runs" / "tol"); shutil.copytree(d, e, dirs_exist_ok=True)
    cm = np.load(e / "confusion/round_002.npy"); j = json.loads((e / "metrics/round_002.json").read_text())
    u = [c["cid"] for c in j["clients"] if not c["selected"]][0]
    off = cm[u].copy(); np.fill_diagonal(off, 0); i, k = np.unravel_index(off.argmax(), off.shape)
    cm[u, i, i] += 5; cm[u, i, k] -= 5
    C.atomic_np_save(e / "confusion/round_002.npy", cm)
    pr = np.load(e / "preds/round_002.u8.npy")            # keep the stored predictions consistent
    idx = np.flatnonzero((Y_TRUE == i) & (pr[u] == k))[:5]; assert len(idx) == 5
    pr[u, idx] = i; C.atomic_np_save(e / "preds/round_002.u8.npy", pr)
    for c in j["clients"]:
        if c["cid"] == u: c.update(metrics_from_confusion(cm[u])); c["per_class"] = per_class_from_confusion(cm[u], CLASSES)
    j["cache_mismatch"] = 1
    for k in METRIC_KEYS:
        v = np.array([c[k] for c in j["clients"]])
        j[k], j[f"{k}_std"], j[f"{k}_min"], j[f"{k}_max"] = float(v.mean()), float(v.std()), float(v.min()), float(v.max())
    (e / "metrics/round_002.json").write_text(json.dumps(j))
    C.atomic_save({**torch.load(e / "weights/round_002.pt", weights_only=True), "metrics": {k: j[k] for k in METRIC_KEYS}},
                  e / "weights/round_002.pt")
    C.rebuild_history(e)
    ok, lines = check(e, cfg)
    assert ok and any(l.startswith("cache    : 1 client-round") for l in lines), "\n".join(lines)
    print("4b. tolerated: 5-row cross-GPU deviation with matching json count passes, reported"); n += 1

    # ---- resume import from a READ-ONLY attached source, markers bound the import
    src_root = tmp / "input" / "prev"
    src = src_root / "runs" / "r"
    for s in C.SUBDIRS: (src / s).mkdir(parents=True, exist_ok=True)
    for s in C.SUBDIRS:
        for f in (d / s).iterdir():
            if f.is_file(): shutil.copyfile(f, src / s / f.name)
    (src / "complete/round_002.done").unlink()               # source committed only round 1
    for p in src.rglob("*"): os.chmod(p, 0o444 if p.is_file() else 0o555)
    dest = mk(tmp / "runs2" / "r")
    C.run_dir = lambda name, base=None, _t=tmp: mk(_t / "runs2" / name)
    last = C.resolve_resume("r", cfg, attached=tmp / "input")
    assert last == 1, last
    assert (dest / "weights/round_001.pt").is_file() and not (dest / "complete/round_002.done").exists()
    assert os.access(dest / "weights/round_001.pt", os.W_OK)
    rows = (dest / "history.csv").read_text().splitlines()
    assert len(rows) == 2, rows
    print("5. import from a read-only mount: only the source's committed round 1, writable, history rebuilt"); n += 1

    # a second, different history with the same fingerprint must be refused
    other = mk(tmp / "runs3" / "r")
    C.run_dir = lambda name, base=None, _t=tmp: mk(_t / "runs3" / name)
    write_run(other, cfg, seed=1)
    try:
        C.resolve_resume("r", cfg, attached=tmp / "input"); raise AssertionError("spliced")
    except RuntimeError as e:
        assert "different weights" in str(e)
    print("6. two histories with one fingerprint: refused, not spliced"); n += 1

    for p in src_root.rglob("*"): os.chmod(p, 0o644 if p.is_file() else 0o755)
    shutil.rmtree(tmp)
    print(f"\nALL {n} CHECKPOINT / VERIFIER CHECKS PASSED")


if __name__ == "__main__":
    main()

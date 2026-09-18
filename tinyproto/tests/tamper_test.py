"""Inject known corruptions into a verified run and require verify_run.py to catch each one.

A verifier that only runs on clean artifacts proves nothing: this is the test that the checks
compare stored values against other stored values (verifying-artifacts.md §5).
"""
import csv, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tinyproto_e2e/runs/e2e")
WORK = Path("/tmp/tinyproto_tamper")


def run_verifier(d):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_run.py"), str(d)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip().splitlines()[-3:]


def fresh():
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(SRC, WORK)
    return WORK


def t_history(d):
    p = d / "metrics" / "history.csv"
    rows = list(csv.DictReader(p.open()))
    cols = list(rows[0])
    rows[0]["proto_accuracy"] = "0.999999"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)


def t_nan_weight(d):
    f = sorted((d / "weights").glob("*.pt"))[-1]
    w = torch.load(f, map_location="cpu", weights_only=False)
    w["params"][0, 123] = float("nan")
    torch.save(w, f)


def t_client_log(d):
    p = sorted((d / "client_log").glob("*.csv"))[-1]
    rows = list(csv.DictReader(p.open()))
    cols = list(rows[0])
    rows[1] = dict(rows[0])                     # duplicate client 0's entry over client 1
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)


def t_proto(d):
    f = sorted((d / "protos").glob("*.pt"))[-1]
    pr = torch.load(f, map_location="cpu", weights_only=False)
    pr["local"][0, 0, 0] += 5.0                 # breaks the recomputed Eq. (10) aggregate
    torch.save(pr, f)


def t_resume(d):
    f = sorted((d / "resume").glob("round_*.pt"))
    f = [x for x in f if ".w" not in x.name][-1]
    rb = torch.load(f, map_location="cpu", weights_only=False)
    rb["global_sparse"] = rb["global_sparse"] + 1.0
    torch.save(rb, f)


def t_confusion(d):
    f = sorted((d / "confusion").glob("*.npz"))[-1]
    z = dict(np.load(f))
    z["proto"][0, 0, 0] += 7                    # confusion no longer covers the full test set
    np.savez_compressed(f, **z)


def t_metrics(d):
    f = sorted((d / "metrics").glob("round_*.json"))[-1]
    m = json.loads(f.read_text())
    m["per_client"]["proto"][0]["f1_macro"] = 0.9999
    f.write_text(json.dumps(m, indent=2))


def t_missing_weights(d):
    sorted((d / "weights").glob("*.pt"))[-1].unlink()


def t_stale_resume(d):
    f = [x for x in sorted((d / "resume").glob("round_*.pt")) if ".w" not in x.name][-1]
    shutil.copy(f, d / "resume" / "round_001.pt")


TAMPERS = [("history.csv accuracy", t_history), ("NaN in one weight", t_nan_weight),
           ("duplicated client log row", t_client_log), ("altered local prototype", t_proto),
           ("altered resume global prototype", t_resume), ("altered confusion count", t_confusion),
           ("altered per-client metric", t_metrics), ("deleted weights file", t_missing_weights),
           ("stale resume blob left behind", t_stale_resume)]

code, tail = run_verifier(SRC)
print(f"clean run: exit {code}  {tail[-1] if tail else ''}")
assert code == 0, "the clean run must pass before tampering means anything"

bad = 0
for name, fn in TAMPERS:
    d = fresh(); fn(d)
    code, tail = run_verifier(d)
    ok = code != 0
    bad += 0 if ok else 1
    print(f"  [{'CAUGHT ' if ok else 'MISSED!'}] {name:<34} exit {code}")
    if ok:
        for line in tail:
            if line.strip().startswith("FAIL"):
                print(f"             {line.strip()[:110]}")
                break
print(("\nALL TAMPERS CAUGHT" if bad == 0 else f"\n{bad} TAMPERS MISSED"))
sys.exit(1 if bad else 0)

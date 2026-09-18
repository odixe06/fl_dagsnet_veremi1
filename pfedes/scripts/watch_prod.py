#!/usr/bin/env python
"""Poll W&B (and optionally the Kaggle kernel status) for the production pFedES runs.

    python scripts/watch_prod.py pfedes_20c_cos pfedes_50c_cos pfedes_100c_cos
    python scripts/watch_prod.py pfedes_20c_cos --kernel minhtriethihi/pfedes-veremi-20-clients-cos \
        --every 300

One block per run per poll: the last rows of the W&B history (round = `_step`; the history
has no `round` key), the summary backend, and the kernel status when --kernel is given.
Stop signals to watch for (CONTEXT.md §5): backend `eager`, `cache_mismatch > 0`, skips
growing round over round, f1_macro falling towards 0 or NaN, train_sec +30 % over the
previous round, VRAM > 14 GiB, kernel ERROR.
"""
import argparse, subprocess, sys, time
from pathlib import Path

COLS = ("f1_macro", "f1_macro_min", "f1_macro_max", "accuracy", "lr", "loss_w_client_mean",
        "ce_orig_client_mean", "loss_theta_client_mean", "skipped_w", "skipped_theta",
        "train_sec", "eval_sec", "seconds", "vram_train_gb", "evaluated", "cache_mismatch")
SKILL = Path(__file__).resolve().parents[1] / ".agents/skills/kaggle-training-notebook/scripts"


def kernel_status(slug):
    owner = slug.split("/")[0]
    try:
        out = subprocess.run([sys.executable, str(SKILL / "kaggle_as.py"), owner, "--",
                              "kaggle", "kernels", "status", slug],
                             capture_output=True, text=True, timeout=90)
        return (out.stdout.strip().splitlines() or [out.stderr.strip()])[-1]
    except Exception as e:
        return f"status unavailable: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="W&B run ids (= run_name)")
    ap.add_argument("--project", default="21522798-uit/pfedes-veremi")
    ap.add_argument("--kernel", action="append", default=[],
                    help="owner/slug to poll with `kaggle kernels status`; repeatable")
    ap.add_argument("--every", type=int, default=0, help="seconds between polls; 0 = once")
    ap.add_argument("--last", type=int, default=3, help="rows to show per run")
    a = ap.parse_args()
    import wandb
    api = wandb.Api(timeout=60)
    while True:
        print(time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()), flush=True)
        for name in a.runs:
            try:
                r = api.run(f"{a.project}/{name}")
            except Exception as e:
                print(f"  {name}: {e}"); continue
            h = r.history(samples=2000, pandas=True)
            print(f"  {name}: state={r.state} backend={r.summary.get('backend')}/"
                  f"{r.summary.get('backend_eval')} rows={len(h)}")
            if len(h):
                keep = ["_step"] + [c for c in COLS if c in h.columns]
                tail = h.sort_values("_step")[keep].tail(a.last)
                for _, row in tail.iterrows():
                    print("    " + " ".join(
                        f"{k}={row[k]:.4g}" if isinstance(row[k], float) else f"{k}={row[k]}"
                        for k in keep))
        for slug in a.kernel:
            print(f"  kernel {slug}: {kernel_status(slug)}")
        if not a.every:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()

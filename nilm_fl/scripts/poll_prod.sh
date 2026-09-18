#!/usr/bin/env bash
# One line per (run) whenever its state/rows change; exits when every kernel is terminal.
# usage: scripts/poll_prod.sh <interval_s> owner/slug:run_name [owner/slug:run_name ...]
source /home/odixe/miniforge3/etc/profile.d/conda.sh && conda activate nckh
cd "$(dirname "$0")/.."
H=.agents/skills/kaggle-training-notebook/scripts
every=$1; shift
declare -A prev
while true; do
  alive=0
  for spec in "$@"; do
    slug=${spec%%:*}; rn=${spec##*:}; owner=${slug%%/*}
    st=$(python $H/kaggle_as.py $owner -- kaggle kernels status $slug 2>&1 | tail -1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
    rows=$(RN=$rn python - <<'PY' 2>/dev/null
import os, wandb
try:
    r = wandb.Api(timeout=60).run(f"21522798-uit/nilm-veremi/{os.environ['RN']}")
    h = r.history(samples=100, pandas=True)
    keep = [c for c in ("f1_macro","global_f1_macro","accuracy","train_sec","eval_sec","seconds","skipped","vram_train_gb") if c in h.columns]
    out = f"backend={r.summary.get('backend')}/{r.summary.get('backend_eval')} rows={len(h)}"
    if len(h) and "_step" in h.columns:
        for _, row in h.sort_values("_step")[["_step"]+keep].tail(2).iterrows():
            out += " | r" + " ".join(f"{k}={row[k]:.4g}" if isinstance(row[k], float) else f"{k}={row[k]}" for k in ["_step"]+keep)
    print(out)
except Exception as e:
    print("wandb: " + str(e)[:60])
PY
)
    cur="$st $rows"
    if [ "${prev[$rn]}" != "$cur" ]; then echo "$(date -u +%H:%MZ) $rn: $cur"; prev[$rn]="$cur"; fi
    case "$st" in *COMPLETE*|*ERROR*|*CANCEL*) ;; *) alive=1;; esac
  done
  [ $alive = 0 ] && { echo "ALL TERMINAL"; break; }
  sleep $every
done

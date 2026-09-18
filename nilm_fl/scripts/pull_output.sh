#!/usr/bin/env bash
# Kéo output kernel Kaggle với vòng retry (CLI hay chết/treo giữa chừng — CONTEXT §9).
# usage: scripts/pull_output.sh <acct> <owner/slug> <dest-dir> [max-attempts]
# Mỗi lượt: xoá file 0 byte (trừ *.done, __init__.py — hợp lệ 0 byte) → chạy `kaggle kernels output` (CLI bỏ qua file đã đủ
# kích thước) → giết khi có file 0 byte đứng yên > 4 phút hoặc lượt vượt 40 phút.
set -u
ACCT=$1; KERNEL=$2; DEST=$3; MAX=${4:-12}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
KAS="$ROOT/.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
mkdir -p "$DEST"
for ((i=1; i<=MAX; i++)); do
  find "$DEST" -type f -size 0 ! -name '*.done' ! -name '__init__.py' -delete
  echo "[$(date -u +%H:%M:%SZ)] attempt $i/$MAX: $KERNEL -> $DEST"
  python "$KAS" "$ACCT" -- kaggle kernels output "$KERNEL" -p "$DEST" --page-size 200 \
      > "${DEST}.cli.log" 2>&1 &
  pid=$!
  t0=$(date +%s); stuck=0
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    now=$(date +%s)
    if (( now - t0 > 2400 )); then echo "  lượt quá 40 phút → kill"; pkill -P $pid; kill $pid; break; fi
    if find "$DEST" -type f -size 0 ! -name '*.done' ! -name '__init__.py' -mmin +4 | grep -q .; then
      stuck=1; echo "  file 0 byte đứng > 4 phút → kill"; pkill -P $pid; kill $pid; break
    fi
  done
  wait $pid; rc=$?
  n=$(find "$DEST" -type f ! -name '.pull.log' | wc -l); z=$(find "$DEST" -type f -size 0 ! -name '*.done' ! -name '__init__.py' | wc -l)
  echo "  rc=$rc files=$n zero=$z  $(tail -1 "${DEST}.cli.log")"
  if [[ $rc -eq 0 && $z -eq 0 && $stuck -eq 0 ]]; then echo "PULL_OK $KERNEL"; exit 0; fi
  sleep 20
done
echo "PULL_FAIL $KERNEL"; exit 1

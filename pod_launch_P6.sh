#!/usr/bin/env bash
# ============================================================================
# POD P6 — CWD against AdamW at 124.5M parameters on eighteen new seeds
# (protocols/P6_cwd_124M_seeds.md, fixed before any run). No tuning: the P4 winners.
#   A  seeds 2007-2012, AdamW (lr 4.5e-3, wd 3.75) then CWD (lr 9e-3, wd 7.5), tag p6a
#   B  seeds 2013-2018, same, tag p6b
#   C  seeds 2019-2024, same, tag p6c
# The tuning-seed grid point replays from the seeded journal. SAFE TO STOP AT ANY TIME.
# ============================================================================
set -euo pipefail
TAG=${1:?tag}
BLOCK=${2:?block: A B C}
bash "$(dirname "$0")/pod_launch_dc.sh" "$TAG"             # provision, corpus check, preflight; no runs
source "$(dirname "$0")/pod_common.sh"
cd "$REPO"
LOG="logs/pod_${TAG}.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dc() {   # <method> <tag> <lr> <wd> <final-seeds>; retried, journaled runs replay
  local method=$1 tag=$2 lr=$3 wd=$4 fs=$5 attempt
  for attempt in 1 2 3; do
    if python -u dc_grid.py --size M --unique-tokens 10000000 --epochs 16 --evals-per-epoch 2 \
      --method "$method" --tag "$tag" --lrs "$lr" --grid "wd=$wd" --final-seeds "$fs" 2>&1 | tee -a "$LOG"; then return 0; fi
    echo "[POD P6:$TAG] $(date -u) attempt $attempt FAILED" | tee -a "$LOG"
  done
  return 1
}

case "$BLOCK" in
A) T=p6a; S=2007,2008,2009,2010,2011,2012 ;;
B) T=p6b; S=2013,2014,2015,2016,2017,2018 ;;
C) T=p6c; S=2019,2020,2021,2022,2023,2024 ;;
*) echo "[POD P6:$TAG] unknown block $BLOCK" | tee -a "$LOG"; T=""; S="" ;;
esac
echo "[POD P6:$TAG] $(date -u) block $BLOCK" | tee -a "$LOG"
STATUS=done
if [ -z "$T" ]; then STATUS=FAILED; else
  dc AdamW "$T" 0.0045 3.75 "$S" || STATUS=FAILED
  dc CWD "$T" 0.009 7.5 "$S" || STATUS=FAILED
fi

echo "[POD P6:$TAG] $(date -u) $STATUS (block $BLOCK) -> results/dc/" | tee -a "$LOG"
POD_ID="${RUNPOD_POD_ID:-$(cat /root/.runpod_pod_id 2>/dev/null || true)}"
if [ -s /root/.runpod_key ] && [ -n "$POD_ID" ]; then
  echo "[POD P6:$TAG] stopping pod $POD_ID" | tee -a "$LOG"
  for attempt in 1 2 3; do
    out=$(curl -s -X POST https://api.runpod.io/graphql -H "Authorization: Bearer $(cat /root/.runpod_key)" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }\"}" || true)
    echo "[POD P6:$TAG] graphql podStop -> $out" | tee -a "$LOG"
    echo "$out" | grep -q '"desiredStatus"' && break
    sleep 30
  done
else
  echo "[POD P6:$TAG] self-stop SKIPPED: key=$([ -s /root/.runpod_key ] && echo yes || echo no) pod_id='$POD_ID'" | tee -a "$LOG"
fi

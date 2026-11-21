#!/usr/bin/env bash
# ============================================================================
# POD P5 — cautious weight decay at 355M parameters
# (protocols/P5_cwd_355M.md, fixed before any run). One block per pod; output tag p5.
#   LA  L R16, AdamW, lr {1.125e-3,2.25e-3,4.5e-3} x wd {1.25,2.5,3.75,7.5}, finals 3001-3006
#   LC  L R16, CWD,   lr {2.25e-3,4.5e-3,9e-3} x wd {3.75,7.5,15},          finals 3001-3006
# Journaled points replay after a restart. SAFE TO STOP AT ANY TIME.
# ============================================================================
set -euo pipefail
TAG=${1:?tag}
BLOCK=${2:?block: LA LC}
bash "$(dirname "$0")/pod_launch_dc.sh" "$TAG"             # provision, corpus check, preflight; no runs
source "$(dirname "$0")/pod_common.sh"
cd "$REPO"
LOG="logs/pod_${TAG}.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # the evaluation's 3 GiB block failed on a fragmented 24 GB pool

run() {   # <method> <lrs> <wds>; journaled points replay, so a failed attempt resumes where it stopped
  local method=$1 lrs=$2 wds=$3 attempt
  for attempt in 1 2 3; do
    if python -u dc_grid.py --size L --unique-tokens 10000000 --epochs 16 --evals-per-epoch 2 --micro-seqs "${MICRO:-8}" \
      --method "$method" --tag p5 --lrs "$lrs" --grid "wd=$wds" --final-seeds "$F6" 2>&1 | tee -a "$LOG"; then return 0; fi
    echo "[POD P5:$TAG] $(date -u) attempt $attempt FAILED" | tee -a "$LOG"
  done
  return 1
}

F6=3001,3002,3003,3004,3005,3006
echo "[POD P5:$TAG] $(date -u) block $BLOCK" | tee -a "$LOG"
STATUS=done
case "$BLOCK" in
LA) run AdamW 0.001125,0.00225,0.0045 1.25,2.5,3.75,7.5 || STATUS=FAILED ;;
LC) run CWD 0.00225,0.0045,0.009 3.75,7.5,15 || STATUS=FAILED ;;
*) echo "unknown block $BLOCK"; exit 2 ;;
esac

echo "[POD P5:$TAG] $(date -u) $STATUS (block $BLOCK) -> results/dc/" | tee -a "$LOG"
POD_ID="${RUNPOD_POD_ID:-$(cat /root/.runpod_pod_id 2>/dev/null || true)}"
if [ -s /root/.runpod_key ] && [ -n "$POD_ID" ]; then
  echo "[POD P5:$TAG] stopping pod $POD_ID" | tee -a "$LOG"
  for attempt in 1 2 3; do
    out=$(curl -s -X POST https://api.runpod.io/graphql -H "Authorization: Bearer $(cat /root/.runpod_key)" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }\"}" || true)
    echo "[POD P5:$TAG] graphql podStop -> $out" | tee -a "$LOG"
    echo "$out" | grep -q '"desiredStatus"' && break
    sleep 30
  done
else
  echo "[POD P5:$TAG] self-stop SKIPPED: key=$([ -s /root/.runpod_key ] && echo yes || echo no) pod_id='$POD_ID'" | tee -a "$LOG"
fi

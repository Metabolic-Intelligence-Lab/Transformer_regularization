#!/usr/bin/env bash
# ============================================================================
# POD P5x — closing the edges of P5 at 355M parameters, and a hardware check
# (protocols/P5x_edges_355M.md, fixed before any run). One block per pod; output tags p5x, p5hw.
#   XA  AdamW, lr {1.125e-3,2.25e-3,4.5e-3,9e-3} x wd {1.25,2.5,3.75,7.5}; finals only if the
#       winner moves; then CWD at lr 4.5e-3, wd 15 on seed 3001 (tag p5hw). RTX 4090.
#   XC  CWD, lr {2.25e-3,4.5e-3,9e-3} x wd {3.75,7.5,15,37.5}; finals only if the winner moves. RTX 3090.
# P5 grid points replay from copies of the P5 journals. SAFE TO STOP AT ANY TIME.
# ============================================================================
set -euo pipefail
TAG=${1:?tag}
BLOCK=${2:?block: XA XC}
bash "$(dirname "$0")/pod_launch_dc.sh" "$TAG"             # provision, corpus check, preflight; no runs
source "$(dirname "$0")/pod_common.sh"
cd "$REPO"
LOG="logs/pod_${TAG}.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dc() {   # <method> <tag> <lrs> <wds> <final-seeds or empty>; retried, journaled points replay
  local method=$1 tag=$2 lrs=$3 wds=$4 fs=$5 attempt fin
  if [ -n "$fs" ]; then fin=(--final-seeds "$fs"); else fin=(--finals 0); fi
  for attempt in 1 2 3; do
    if python -u dc_grid.py --size L --unique-tokens 10000000 --epochs 16 --evals-per-epoch 2 --micro-seqs 8 \
      --method "$method" --tag "$tag" --lrs "$lrs" --grid "wd=$wds" "${fin[@]}" 2>&1 | tee -a "$LOG"; then return 0; fi
    echo "[POD P5x:$TAG] $(date -u) attempt $attempt FAILED" | tee -a "$LOG"
  done
  return 1
}

moved() {   # <stem> <lr> <wd>: true if the winner in results/dc/<stem>.json differs from (lr, wd)
  python - "$@" <<'PY'
import json, sys
w = json.load(open(f'results/dc/{sys.argv[1]}.json'))['winner']['hp']
sys.exit(0 if (w['lr'], w['wd']) != (float(sys.argv[2]), float(sys.argv[3])) else 1)
PY
}

F6=3001,3002,3003,3004,3005,3006
echo "[POD P5x:$TAG] $(date -u) block $BLOCK" | tee -a "$LOG"
STATUS=done
case "$BLOCK" in
XA)
  A=(AdamW p5x 0.001125,0.00225,0.0045,0.009 1.25,2.5,3.75,7.5)
  dc "${A[@]}" "" || STATUS=FAILED
  if [ "$STATUS" = done ] && moved L_D10M_E16_AdamW_p5x 0.0045 3.75; then
    echo "[POD P5x:$TAG] AdamW winner moved: finals on $F6" | tee -a "$LOG"
    dc "${A[@]}" "$F6" || STATUS=FAILED
  fi
  dc CWD p5hw 0.0045 15 3001 || STATUS=FAILED ;;
XC)
  C=(CWD p5x 0.00225,0.0045,0.009 3.75,7.5,15,37.5)
  dc "${C[@]}" "" || STATUS=FAILED
  if [ "$STATUS" = done ] && moved L_D10M_E16_CWD_p5x 0.0045 15; then
    echo "[POD P5x:$TAG] CWD winner moved: finals on $F6" | tee -a "$LOG"
    dc "${C[@]}" "$F6" || STATUS=FAILED
  fi ;;
*) echo "unknown block $BLOCK"; exit 2 ;;
esac

echo "[POD P5x:$TAG] $(date -u) $STATUS (block $BLOCK) -> results/dc/" | tee -a "$LOG"
POD_ID="${RUNPOD_POD_ID:-$(cat /root/.runpod_pod_id 2>/dev/null || true)}"
if [ -s /root/.runpod_key ] && [ -n "$POD_ID" ]; then
  echo "[POD P5x:$TAG] stopping pod $POD_ID" | tee -a "$LOG"
  for attempt in 1 2 3; do
    out=$(curl -s -X POST https://api.runpod.io/graphql -H "Authorization: Bearer $(cat /root/.runpod_key)" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }\"}" || true)
    echo "[POD P5x:$TAG] graphql podStop -> $out" | tee -a "$LOG"
    echo "$out" | grep -q '"desiredStatus"' && break
    sleep 30
  done
else
  echo "[POD P5x:$TAG] self-stop SKIPPED: key=$([ -s /root/.runpod_key ] && echo yes || echo no) pod_id='$POD_ID'" | tee -a "$LOG"
fi

#!/usr/bin/env bash
# ============================================================================
# POD P3x — edge extension and added seeds for the re-tuned single-pass arm
# (protocols/P3x_extension.md, fixed before any run). One block per pod; output tag p3x.
#   EX  XS R1, AdamW+EMA, ema {0.9,0.99,0.999}, finals 1001-1006
#   DX  XS R1, AdamW+dropout, dropout {0.01,0.03,0.1}, finals 1001-1006
#   AX  XS R1, AdamW at its P3 grid, finals 1001-1006
# Seed-42 points and finals of P3 replay from copies of the P3 journals. SAFE TO STOP AT ANY TIME.
# ============================================================================
set -euo pipefail
TAG=${1:?tag}
BLOCK=${2:?block: EX DX AX}
bash "$(dirname "$0")/pod_launch_dc.sh" "$TAG"             # provision, corpus check, preflight; no runs
source "$(dirname "$0")/pod_common.sh"
cd "$REPO"
LOG="logs/pod_${TAG}.log"

seed_journal() {   # <tagged stem> <earlier stems>...: start the P3x journal from earlier journals
  local dst="results/journal/dc_$1.jsonl"; shift
  [ -f "$dst" ] && return 0
  : > "$dst"
  for s in "$@"; do [ -f "results/journal/dc_$s.jsonl" ] && cat "results/journal/dc_$s.jsonl" >> "$dst"; done
  return 0
}

run() {   # <method> <lrs> <wds> [extra --grid k=v,...]
  local method=$1 lrs=$2 wds=$3; shift 3
  python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --evals-per-epoch 32 \
    --method "$method" --tag p3x --lrs "$lrs" --grid "wd=$wds" "$@" --final-seeds "$F6" 2>&1 | tee -a "$LOG"
}

F6=1001,1002,1003,1004,1005,1006
P=XS_D159.384M_E1
echo "[POD P3x:$TAG] $(date -u) block $BLOCK" | tee -a "$LOG"
case "$BLOCK" in
EX)
  seed_journal ${P}_AdamW+EMA_p3x ${P}_AdamW+EMA_p3
  run AdamW+EMA 0.0005,0.0015,0.0045 0.1,0.5,2.5 --grid ema=0.9,0.99,0.999 ;;
DX)
  seed_journal ${P}_AdamW+dropout_p3x ${P}_AdamW+dropout_p3
  run AdamW+dropout 0.0005,0.0015,0.0045 0.1,0.5,2.5 --grid dropout=0.01,0.03,0.1 ;;
AX)
  seed_journal ${P}_AdamW_p3x ${P}_AdamW_p3
  run AdamW 0.0005,0.0015,0.0045,0.0135 0.1,0.5,2.5,7.5 ;;
*) echo "unknown block $BLOCK"; exit 2 ;;
esac

echo "[POD P3x:$TAG] $(date -u) done (block $BLOCK) -> results/dc/" | tee -a "$LOG"
POD_ID="${RUNPOD_POD_ID:-$(cat /root/.runpod_pod_id 2>/dev/null || true)}"
if [ -s /root/.runpod_key ] && [ -n "$POD_ID" ]; then
  echo "[POD P3x:$TAG] stopping pod $POD_ID" | tee -a "$LOG"
  for attempt in 1 2 3; do
    out=$(curl -s -X POST https://api.runpod.io/graphql -H "Authorization: Bearer $(cat /root/.runpod_key)" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }\"}" || true)
    echo "[POD P3x:$TAG] graphql podStop -> $out" | tee -a "$LOG"
    echo "$out" | grep -q '"desiredStatus"' && break
    sleep 30
  done
else
  echo "[POD P3x:$TAG] self-stop SKIPPED: key=$([ -s /root/.runpod_key ] && echo yes || echo no) pod_id='$POD_ID'" | tee -a "$LOG"
fi

#!/usr/bin/env bash
# ============================================================================
# POD P4 — cautious weight decay at 124.5M parameters, completed
# (protocols/P4_cwd_size.md, fixed before any run). One block per pod; output tag p4.
#   MA  M R16, AdamW, lr {2.25e-3,4.5e-3,9e-3} x wd {0.5,2.5,3.75,7.5}, finals 2001-2006
#   MC  M R16, CWD,   lr {4.5e-3,9e-3,1.8e-2} x wd {2.5,7.5,37.5},      finals 2001-2006
# Grid points and finals of P2 replay from copies of the P2 journals. SAFE TO STOP AT ANY TIME.
# ============================================================================
set -euo pipefail
TAG=${1:?tag}
BLOCK=${2:?block: MA MC}
bash "$(dirname "$0")/pod_launch_dc.sh" "$TAG"             # provision, corpus check, preflight; no runs
source "$(dirname "$0")/pod_common.sh"
cd "$REPO"
LOG="logs/pod_${TAG}.log"

seed_journal() {   # <tagged stem> <earlier stems>...: start the P4 journal from earlier journals
  local dst="results/journal/dc_$1.jsonl"; shift
  [ -f "$dst" ] && return 0
  : > "$dst"
  for s in "$@"; do [ -f "results/journal/dc_$s.jsonl" ] && cat "results/journal/dc_$s.jsonl" >> "$dst"; done
  return 0
}

run() {   # <method> <lrs> <wds>
  local method=$1 lrs=$2 wds=$3
  python -u dc_grid.py --size M --unique-tokens 10000000 --epochs 16 --evals-per-epoch 2 \
    --method "$method" --tag p4 --lrs "$lrs" --grid "wd=$wds" --final-seeds "$F6" 2>&1 | tee -a "$LOG"
}

F6=2001,2002,2003,2004,2005,2006
P=M_D10M_E16
echo "[POD P4:$TAG] $(date -u) block $BLOCK" | tee -a "$LOG"
case "$BLOCK" in
MA)
  seed_journal ${P}_AdamW_p4 ${P}_AdamW_p2
  run AdamW 0.00225,0.0045,0.009 0.5,2.5,3.75,7.5 ;;
MC)
  seed_journal ${P}_CWD_p4 ${P}_CWD_p2
  run CWD 0.0045,0.009,0.018 2.5,7.5,37.5 ;;
*) echo "unknown block $BLOCK"; exit 2 ;;
esac

echo "[POD P4:$TAG] $(date -u) done (block $BLOCK) -> results/dc/" | tee -a "$LOG"
POD_ID="${RUNPOD_POD_ID:-$(cat /root/.runpod_pod_id 2>/dev/null || true)}"
if [ -s /root/.runpod_key ] && [ -n "$POD_ID" ]; then
  echo "[POD P4:$TAG] stopping pod $POD_ID" | tee -a "$LOG"
  for attempt in 1 2 3; do
    out=$(curl -s -X POST https://api.runpod.io/graphql -H "Authorization: Bearer $(cat /root/.runpod_key)" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }\"}" || true)
    echo "[POD P4:$TAG] graphql podStop -> $out" | tee -a "$LOG"
    echo "$out" | grep -q '"desiredStatus"' && break
    sleep 30
  done
else
  echo "[POD P4:$TAG] self-stop SKIPPED: key=$([ -s /root/.runpod_key ] && echo yes || echo no) pod_id='$POD_ID'" | tee -a "$LOG"
fi

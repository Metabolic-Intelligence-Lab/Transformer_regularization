#!/usr/bin/env bash
# ============================================================================
# POD P1 extra — post-registration sensitivity runs (protocols/P1_minibench.md is NOT changed;
# these are reported as sensitivity analyses, never merged into the pre-registered tables).
# ----------------------------------------------------------------------------
#   A) three more final seeds (1004-1006) at S for AdamW and AdamW+EMA under their pre-registered
#      winners (the S contrast EMA vs AdamW was -0.024 nats with n = 3, Holm 0.21);
#   B) AdamW at S with a second wd extension (7.5): the winner sat on the wd edge (2.5) after the
#      one extension the protocol allows, so the standard may be under-tuned;
#   C) SF-AdamW at XS/R16 beyond the protocol grid (lr 4e-2, 1.2e-1; wd 7.5): under-tuned, or
#      just slow at this horizon (uniform averaging over 2,432 steps)? Tuning points only.
# Every search runs under its own --tag, seeded with a copy of the pre-registered journal, so the
# already-trained grid points and finals are reused and the pre-registered files are never rewritten.
#   bash pod_launch_P1_extra.sh <tag>        # via pod_p1.sh: AUTOSTOP=1 bash pod_p1.sh P1_x pod_launch_P1_extra.sh
# SAFE TO STOP AT ANY TIME: re-running resumes from results/journal/dc_*.jsonl.
# ============================================================================
set -euo pipefail
TAG=${1:?tag}
bash "$(dirname "$0")/pod_launch_dc.sh" "$TAG"             # provision, corpus check, preflight; no runs
source "$(dirname "$0")/pod_common.sh"
cd "$REPO"
LOG="logs/pod_${TAG}.log"

seed_journal() {   # <pre-registered stem> <tagged stem>: start the tagged search from the registered journal
  [ -f "results/journal/dc_$2.jsonl" ] || cp "results/journal/dc_$1.jsonl" "results/journal/dc_$2.jsonl"
}
S_ARGS=(--size S --unique-tokens 10000000 --epochs 16)
XS_ARGS=(--size XS --unique-tokens 10000000 --epochs 16)
LR_ADAM=5e-4,1.5e-3,4.5e-3

echo "[POD P1:$TAG] $(date -u) A) extra seeds at S: AdamW, AdamW+EMA (finals 6)" | tee -a "$LOG"
seed_journal S_D10M_E16_AdamW S_D10M_E16_AdamW_x6
python -u dc_grid.py "${S_ARGS[@]}" --method AdamW --lrs $LR_ADAM --grid wd=0.1,0.5,2.5 \
  --finals 6 --tag x6 2>&1 | tee -a "$LOG"
seed_journal S_D10M_E16_AdamW+EMA S_D10M_E16_AdamW+EMA_x6
python -u dc_grid.py "${S_ARGS[@]}" --method AdamW+EMA --lrs $LR_ADAM --grid ema=0.99,0.999,0.9999 \
  --fixed wd=2.5 --finals 6 --tag x6 2>&1 | tee -a "$LOG"

echo "[POD P1:$TAG] $(date -u) B) AdamW at S, second wd extension (7.5)" | tee -a "$LOG"
seed_journal S_D10M_E16_AdamW S_D10M_E16_AdamW_wdx
python -u dc_grid.py "${S_ARGS[@]}" --method AdamW --lrs $LR_ADAM --grid wd=0.1,0.5,2.5,7.5 \
  --finals 3 --tag wdx 2>&1 | tee -a "$LOG"

echo "[POD P1:$TAG] $(date -u) C) SF-AdamW audit at XS/R16 (tuning points only)" | tee -a "$LOG"
seed_journal XS_D10M_E16_SF-AdamW XS_D10M_E16_SF-AdamW_audit
python -u dc_grid.py "${XS_ARGS[@]}" --method SF-AdamW --lrs 4.5e-3,1.35e-2,4e-2,1.2e-1 --grid wd=2.5,7.5 \
  --finals 0 --tag audit 2>&1 | tee -a "$LOG"

echo "[POD P1:$TAG] $(date -u) done -> results/dc/" | tee -a "$LOG"
# Self-stop (opt-in: pod_p1.sh uploads /root/.runpod_key and /root/.runpod_pod_id when AUTOSTOP=1).
POD_ID="${RUNPOD_POD_ID:-$(cat /root/.runpod_pod_id 2>/dev/null || true)}"
if [ -s /root/.runpod_key ] && [ -n "$POD_ID" ]; then
  echo "[POD P1:$TAG] stopping pod $POD_ID" | tee -a "$LOG"
  for attempt in 1 2 3; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
      -H "Authorization: Bearer $(cat /root/.runpod_key)" -H 'User-Agent: regdecay-pod-create/1.0' || echo 000)
    echo "[POD P1:$TAG] stop attempt $attempt -> HTTP $code" | tee -a "$LOG"
    case "$code" in 2*) break;; esac; sleep 30
  done
else
  echo "[POD P1:$TAG] self-stop SKIPPED: key=$([ -s /root/.runpod_key ] && echo yes || echo no) pod_id='$POD_ID'" | tee -a "$LOG"
fi

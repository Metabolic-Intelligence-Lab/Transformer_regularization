#!/usr/bin/env bash
# ============================================================================
# POD P1 matrix — protocols/P1_minibench.md, run through p1_matrix.py (edge rule and
# stacking methods applied mechanically; every run journaled by dc_grid.py).
# ----------------------------------------------------------------------------
#   bash pod_launch_P1_matrix.sh <tag> <size XS|S> <methods: all|a,b,c> <phase> [<phase>...]
#   e.g.  bash pod_launch_P1_matrix.sh P1_xs XS all r16 r1
#         bash pod_launch_P1_matrix.sh P1_s  S  all s          # needs every XS R16 result in results/dc
# Via pod_p1.sh:  RUNPOD_API_KEY=... bash pod_p1.sh P1_xs pod_launch_P1_matrix.sh XS all r16 r1
# The corpus must be on the pod at /workspace/data/dc (pod_p1.sh uploads it).
# SAFE TO STOP AT ANY TIME: re-running resumes from results/journal/dc_*.jsonl.
# ============================================================================
set -euo pipefail
TAG=${1:?tag}; SIZE=${2:?size XS|S}; METHODS=${3:?methods: all or comma list}; shift 3
[ $# -gt 0 ] || { echo "usage: $0 <tag> <size> <methods> <phase...>"; exit 1; }
bash "$(dirname "$0")/pod_launch_dc.sh" "$TAG"             # provision, corpus check, preflight; no runs
source "$(dirname "$0")/pod_common.sh"
cd "$REPO"
for ph in "$@"; do
  echo "[POD P1:$TAG] $(date -u) phase $ph size $SIZE methods $METHODS"
  if [ "$METHODS" = all ]; then
    python -u p1_matrix.py --phase "$ph" --size "$SIZE" 2>&1 | tee -a "logs/pod_${TAG}.log"
  else
    python -u p1_matrix.py --phase "$ph" --size "$SIZE" --methods "$METHODS" 2>&1 | tee -a "logs/pod_${TAG}.log"
  fi
done
echo "[POD P1:$TAG] $(date -u) done -> results/dc/"
# Self-stop (opt-in: pod_p1.sh uploads /root/.runpod_key when AUTOSTOP=1). A STOPPED pod keeps
# /workspace (results, journals, corpus) and bills storage only; resume with pod_power.py start.
# The pod id comes from the container env or, when the tmux/ssh session does not inherit it (P1_xs,
# 2026-10-22: the stop silently never ran and the pod idled 19 days), from /root/.runpod_pod_id
# written by pod_p1.sh. Every branch logs, so a skipped stop is visible in the log.
POD_ID="${RUNPOD_POD_ID:-$(cat /root/.runpod_pod_id 2>/dev/null || true)}"
if [ -s /root/.runpod_key ] && [ -n "$POD_ID" ]; then
  echo "[POD P1:$TAG] stopping pod $POD_ID"
  for attempt in 1 2 3; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
      -H "Authorization: Bearer $(cat /root/.runpod_key)" -H 'User-Agent: regdecay-pod-create/1.0' || echo 000)
    echo "[POD P1:$TAG] stop attempt $attempt -> HTTP $code"
    case "$code" in 2*) break;; esac; sleep 30
  done
else
  echo "[POD P1:$TAG] self-stop SKIPPED: key=$([ -s /root/.runpod_key ] && echo yes || echo no) pod_id='$POD_ID'"
fi

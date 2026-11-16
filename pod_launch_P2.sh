#!/usr/bin/env bash
# ============================================================================
# POD P2 — confirmation of the Cautious Weight Decay result (protocols/P2_cwd_confirmation.md,
# fixed before any run). One block per pod; output tag p2; held-out final seeds 2001-2006.
#   S   H1: S, AdamW and CWD, lr {1.5e-3,4.5e-3,1.35e-2} x wd {2.5,3.75,5,7.5}, 6 finals   ~8.8 GPU-h
#   X1  H2: XS R1 (both methods) and the R16 finals                                       ~6.9 GPU-h
#   X2  H2: XS R4 (39.8M tokens x 4 epochs, both methods)                                 ~6.4 GPU-h
#   M1  H3: M, AdamW, lr {4.5e-3,9e-3} x wd {2.5,7.5}, 3 finals                           ~5.5 GPU-h
#   M2  H3: M, CWD, same grid                                                             ~6.4 GPU-h
#   AUTOSTOP=1 bash pod_p1.sh P2_s pod_launch_P2.sh S
# Each search runs without finals first; if the winner sits on an lr edge, the grid gets one
# more lr (factor 3, M: 2) beyond that edge (P1 edge rule, once); then the finals run on the
# final winner. Grid points already run on seed 42 in P1 are replayed from copies of the P1
# journals. SAFE TO STOP AT ANY TIME: re-running resumes from the journals.
# ============================================================================
set -euo pipefail
TAG=${1:?tag}
BLOCK=${2:?block: S X1 X2 M1 M2}
bash "$(dirname "$0")/pod_launch_dc.sh" "$TAG"             # provision, corpus check, preflight; no runs
source "$(dirname "$0")/pod_common.sh"
cd "$REPO"
LOG="logs/pod_${TAG}.log"

seed_journal() {   # <tagged stem> <P1 stem>...: start the P2 journal from the P1 journals (seed-42 points replay)
  local dst="results/journal/dc_$1.jsonl"; shift
  [ -f "$dst" ] && return 0
  : > "$dst"
  for s in "$@"; do [ -f "results/journal/dc_$s.jsonl" ] && cat "results/journal/dc_$s.jsonl" >> "$dst"; done
  return 0
}

# search <size> <unique tokens> <epochs> <evals per epoch> <method> <lrs> <wds> <final seeds> <edge factor>
search() {
  local size=$1 tok=$2 ep=$3 epe=$4 method=$5 lrs=$6 wds=$7 seeds=$8 fac=$9
  local common=(--size "$size" --unique-tokens "$tok" --epochs "$ep" --evals-per-epoch "$epe"
                --method "$method" --grid "wd=$wds" --tag p2)
  python -u dc_grid.py "${common[@]}" --lrs "$lrs" --finals 0 2>&1 | tee -a "$LOG"
  local stem; stem=$(python - "$size" "$tok" "$ep" "$method" <<'PY'
import sys; s, t, e, m = sys.argv[1:]; print(f"{s}_D{int(t) / 1e6:g}M_E{e}_{m}_p2")
PY
)
  lrs=$(python - "results/dc/$stem.json" "$lrs" "$fac" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); lrs = [float(x) for x in sys.argv[2].split(',')]; f = float(sys.argv[3])
w = d['winner']['hp']['lr']
if w == max(lrs): lrs.append(round(max(lrs) * f, 10)); print(f"[P2] edge: winner at the largest lr, adding {lrs[-1]:g}", file=sys.stderr)
elif w == min(lrs): lrs.insert(0, round(min(lrs) / f, 10)); print(f"[P2] edge: winner at the smallest lr, adding {lrs[0]:g}", file=sys.stderr)
print(','.join(f"{x:g}" for x in sorted(lrs)))
PY
)
  python -u dc_grid.py "${common[@]}" --lrs "$lrs" --final-seeds "$seeds" 2>&1 | tee -a "$LOG"
}

S6=2001,2002,2003,2004,2005,2006
S3=2001,2002,2003
R4TOK=39845889     # 608 steps per epoch x 4 epochs = 2,432 steps
R1TOK=159383553
echo "[POD P2:$TAG] $(date -u) block $BLOCK" | tee -a "$LOG"
case "$BLOCK" in
S)
  seed_journal S_D10M_E16_AdamW_p2 S_D10M_E16_AdamW S_D10M_E16_AdamW_wdx
  search S 10000000 16 2 AdamW 0.0015,0.0045,0.0135 2.5,3.75,5,7.5 $S6 3
  seed_journal S_D10M_E16_CWD_p2 S_D10M_E16_CWD S_D10M_E16_CWD_cwd
  search S 10000000 16 2 CWD 0.0015,0.0045,0.0135 2.5,3.75,5,7.5,37.5 $S6 3 ;;
X1)
  seed_journal XS_D159.384M_E1_AdamW_p2 XS_D159.384M_E1_AdamW
  search XS $R1TOK 1 32 AdamW 0.0015,0.0045,0.0135 0.5,2.5,7.5 $S3 3
  seed_journal XS_D159.384M_E1_CWD_p2 XS_D159.384M_E1_CWD
  search XS $R1TOK 1 32 CWD 0.0015,0.0045,0.0135 0.5,2.5,7.5 $S3 3
  seed_journal XS_D10M_E16_AdamW_p2 XS_D10M_E16_AdamW XS_D10M_E16_AdamW_edge
  search XS 10000000 16 2 AdamW 0.0015,0.0045,0.0135 0.5,2.5,7.5 $S3 3
  seed_journal XS_D10M_E16_CWD_p2 XS_D10M_E16_CWD XS_D10M_E16_CWD_edge XS_D10M_E16_CWD_cwd
  search XS 10000000 16 2 CWD 0.0015,0.0045,0.0135 0.5,2.5,7.5 $S3 3 ;;
X2)
  search XS $R4TOK 4 8 AdamW 0.0015,0.0045,0.0135 0.5,2.5,7.5 $S3 3
  search XS $R4TOK 4 8 CWD 0.0015,0.0045,0.0135 0.5,2.5,7.5 $S3 3 ;;
M1)
  seed_journal M_D10M_E16_AdamW_p2 M_D10M_E16_AdamW_m
  search M 10000000 16 2 AdamW 0.0045,0.009 2.5,7.5 $S3 2 ;;
M2)
  search M 10000000 16 2 CWD 0.0045,0.009 2.5,7.5 $S3 2 ;;
*) echo "unknown block $BLOCK"; exit 2 ;;
esac

echo "[POD P2:$TAG] $(date -u) done (block $BLOCK) -> results/dc/" | tee -a "$LOG"
POD_ID="${RUNPOD_POD_ID:-$(cat /root/.runpod_pod_id 2>/dev/null || true)}"
if [ -s /root/.runpod_key ] && [ -n "$POD_ID" ]; then
  echo "[POD P2:$TAG] stopping pod $POD_ID" | tee -a "$LOG"
  for attempt in 1 2 3; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
      -H "Authorization: Bearer $(cat /root/.runpod_key)" -H 'User-Agent: regdecay-pod-create/1.0' || echo 000)
    echo "[POD P2:$TAG] stop attempt $attempt -> HTTP $code" | tee -a "$LOG"
    case "$code" in 2*) break;; esac; sleep 30
  done
else
  echo "[POD P2:$TAG] self-stop SKIPPED: key=$([ -s /root/.runpod_key ] && echo yes || echo no) pod_id='$POD_ID'" | tee -a "$LOG"
fi

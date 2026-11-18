#!/usr/bin/env bash
# ============================================================================
# POD P3 — mask controls for CWD and a re-tuned single-pass arm (protocols/P3_controls.md, fixed
# before any run). One block per pod; output tag p3.
#   RW  S, RandWD (random 40 % mask), lr {1.5e-3,4.5e-3,1.35e-2} x wd {2.5,7.5,12.5}, finals 2001-2006
#   AC  S, AntiCWD (u*p < 0 mask),    lr {1.5e-3,4.5e-3,1.35e-2} x wd {2.5,5,7.5},    finals 2001-2006
#   RA  XS R1, AdamW (wd 0.1..7.5) and AdamW+EMA (wd x ema), finals 1001-1003
#   RD  XS R1, AdamW+dropout (wd x dropout), finals 1001-1003
#   AUTOSTOP=1 bash pod_p1.sh P3_rw pod_launch_P3.sh RW
# Each search runs without finals first; if the winner sits on an edge of lr or wd, that axis gets
# one more value beyond the edge (factor 3, once per axis); then the finals run on the final winner.
# Seed-42 points of P1 and P2 replay from copies of their journals. SAFE TO STOP AT ANY TIME.
# ============================================================================
set -euo pipefail
TAG=${1:?tag}
BLOCK=${2:?block: RW AC RA RD}
bash "$(dirname "$0")/pod_launch_dc.sh" "$TAG"             # provision, corpus check, preflight; no runs
source "$(dirname "$0")/pod_common.sh"
cd "$REPO"
LOG="logs/pod_${TAG}.log"

seed_journal() {   # <tagged stem> <earlier stems>...: start the P3 journal from earlier journals
  local dst="results/journal/dc_$1.jsonl"; shift
  [ -f "$dst" ] && return 0
  : > "$dst"
  for s in "$@"; do [ -f "results/journal/dc_$s.jsonl" ] && cat "results/journal/dc_$s.jsonl" >> "$dst"; done
  return 0
}

# search <size> <unique tokens> <epochs> <evals per epoch> <method> <lrs> <wds> <final seeds> [extra --grid k=v,...]
search() {
  local size=$1 tok=$2 ep=$3 epe=$4 method=$5 lrs=$6 wds=$7 seeds=$8; shift 8
  local extra=("$@")
  local stem; stem=$(python - "$size" "$tok" "$ep" "$method" <<'PY'
import sys; s, t, e, m = sys.argv[1:]; print(f"{s}_D{int(t) / 1e6:g}M_E{e}_{m}_p3")
PY
)
  local pass
  for pass in 1 2 3; do       # pass 1: base grid; passes 2-3: at most one extension per axis
    python -u dc_grid.py --size "$size" --unique-tokens "$tok" --epochs "$ep" --evals-per-epoch "$epe" \
      --method "$method" --tag p3 --lrs "$lrs" --grid "wd=$wds" "${extra[@]}" --finals 0 2>&1 | tee -a "$LOG"
    read -r lrs wds changed < <(python - "results/dc/$stem.json" "$lrs" "$wds" "$pass" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); w = d['winner']['hp']
axes = {'lr': [float(x) for x in sys.argv[2].split(',')], 'wd': [float(x) for x in sys.argv[3].split(',')]}
done = json.load(open(sys.argv[1] + '.ext')) if sys.argv[4] != '1' else {}
changed = 0
for k, vals in axes.items():
    if done.get(k):
        continue
    if w[k] == max(vals):
        vals.append(round(max(vals) * 3, 10)); done[k] = 1; changed = 1
        print(f"[P3] edge: {k} winner at the largest value, adding {vals[-1]:g}", file=sys.stderr)
    elif w[k] == min(vals):
        vals.insert(0, round(min(vals) / 3, 10)); done[k] = 1; changed = 1
        print(f"[P3] edge: {k} winner at the smallest value, adding {vals[0]:g}", file=sys.stderr)
json.dump(done, open(sys.argv[1] + '.ext', 'w'))
f = lambda v: ','.join(f"{x:g}" for x in sorted(v))
print(f(axes['lr']), f(axes['wd']), changed)
PY
)
    [ "$changed" = 1 ] || break
  done
  python -u dc_grid.py --size "$size" --unique-tokens "$tok" --epochs "$ep" --evals-per-epoch "$epe" \
    --method "$method" --tag p3 --lrs "$lrs" --grid "wd=$wds" "${extra[@]}" --final-seeds "$seeds" 2>&1 | tee -a "$LOG"
}

S6=2001,2002,2003,2004,2005,2006
F3=1001,1002,1003
R1TOK=159383553
echo "[POD P3:$TAG] $(date -u) block $BLOCK" | tee -a "$LOG"
case "$BLOCK" in
RW)
  search S 10000000 16 2 RandWD 0.0015,0.0045,0.0135 2.5,7.5,12.5 $S6 ;;
AC)
  search S 10000000 16 2 AntiCWD 0.0015,0.0045,0.0135 2.5,5,7.5 $S6 ;;
RA)
  seed_journal XS_D159.384M_E1_AdamW_p3 XS_D159.384M_E1_AdamW XS_D159.384M_E1_AdamW_x3 XS_D159.384M_E1_AdamW_p2
  search XS $R1TOK 1 32 AdamW 0.0005,0.0015,0.0045,0.0135 0.1,0.5,2.5,7.5 $F3
  seed_journal XS_D159.384M_E1_AdamW+EMA_p3 XS_D159.384M_E1_AdamW+EMA XS_D159.384M_E1_AdamW+EMA_x3
  search XS $R1TOK 1 32 AdamW+EMA 0.0015,0.0045 0.1,0.5,2.5 $F3 --grid ema=0.9,0.99 ;;
RD)
  seed_journal XS_D159.384M_E1_AdamW+dropout_p3 XS_D159.384M_E1_AdamW+dropout XS_D159.384M_E1_AdamW+dropout_x3
  search XS $R1TOK 1 32 AdamW+dropout 0.0015,0.0045 0.1,0.5,2.5 $F3 --grid dropout=0.03,0.1 ;;
*) echo "unknown block $BLOCK"; exit 2 ;;
esac

echo "[POD P3:$TAG] $(date -u) done (block $BLOCK) -> results/dc/" | tee -a "$LOG"
POD_ID="${RUNPOD_POD_ID:-$(cat /root/.runpod_pod_id 2>/dev/null || true)}"
if [ -s /root/.runpod_key ] && [ -n "$POD_ID" ]; then
  echo "[POD P3:$TAG] stopping pod $POD_ID" | tee -a "$LOG"
  for attempt in 1 2 3; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
      -H "Authorization: Bearer $(cat /root/.runpod_key)" -H 'User-Agent: regdecay-pod-create/1.0' || echo 000)
    echo "[POD P3:$TAG] stop attempt $attempt -> HTTP $code" | tee -a "$LOG"
    case "$code" in 2*) break;; esac
    # fallback: the GraphQL API (rest.runpod.io stopped resolving on 2026-11-17)
    out=$(curl -s -X POST https://api.runpod.io/graphql -H "Authorization: Bearer $(cat /root/.runpod_key)" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }\"}" || true)
    echo "[POD P3:$TAG] graphql podStop -> $out" | tee -a "$LOG"
    echo "$out" | grep -q '"desiredStatus"' && break
    sleep 30
  done
else
  echo "[POD P3:$TAG] self-stop SKIPPED: key=$([ -s /root/.runpod_key ] && echo yes || echo no) pod_id='$POD_ID'" | tee -a "$LOG"
fi

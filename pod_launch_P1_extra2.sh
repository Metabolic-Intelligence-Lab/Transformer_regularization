#!/usr/bin/env bash
# ============================================================================
# POD P1 extra 2 — second package of post-registration runs (protocols/P1_minibench.md is NOT
# changed; everything is reported under "Sensitivity analyses", never merged into the registered
# tables). Generated from the registered result files by scratch gen_extra2.py on 2026-11-14: every
# replayed search reproduces the registered grid exactly, so the new runs are only the added seeds
# or points. Blocks and cost on one RTX 3090 (XS run 16 min, S run 21 min):
#   A: 21 new runs (XS) ~ 5.6 GPU-h
#   B: 36 new runs (XS) ~ 9.6 GPU-h
#   C: 13 new runs (XS) ~ 3.5 GPU-h
#   D: 28 new runs (XS) ~ 7.5 GPU-h
#   E: 12 new runs (S) ~ 4.2 GPU-h
#   F: 19 new runs (XS) ~ 5.1 GPU-h
#   G: 18 runs at M (~55 min each on a 3090) ~ 17 GPU-h; a third pod: pod_launch_P1_extra2.sh G
#   part 1 = A C D (~17 GPU-h), part 2 = B E F (~19 GPU-h); two community pods run them in parallel.
#   AUTOSTOP=1 bash pod_p1.sh P1_y1 pod_launch_P1_extra2.sh 1
#   AUTOSTOP=1 bash pod_p1.sh P1_y2 pod_launch_P1_extra2.sh 2
# Tags: x6 (six seeds), tune43 (second tuning seed), x3 (third R1 seed), edge (second extension),
# r1s (R1 at S), add (AdEMAMix), m (124.5M spot check), cwd (CWD follow-up, block H), add2 (AdEMAMix with alpha and beta3 tuned, block I). SAFE TO STOP AT ANY TIME: re-running resumes from the journals.
# ============================================================================
set -euo pipefail
TAG=${1:?tag}
PART=${2:-all}
bash "$(dirname "$0")/pod_launch_dc.sh" "$TAG"             # provision, corpus check, preflight; no runs
source "$(dirname "$0")/pod_common.sh"
cd "$REPO"
LOG="logs/pod_${TAG}.log"

seed_journal() {   # <registered stem> <tagged stem>: start the tagged search from the registered journal
  [ -f "results/journal/dc_$2.jsonl" ] || cp "results/journal/dc_$1.jsonl" "results/journal/dc_$2.jsonl"
}
block_on() {
  case "$PART" in all) return 0;; 1) [[ " A C D " == *" $1 "* ]];; 2) [[ " B E F " == *" $1 "* ]];; *) [[ " $PART " == *" $1 "* ]];; esac
}
run_block() { block_on "$1" && echo "[POD P1:$TAG] $(date -u) $2" | tee -a "$LOG"; }

run_block A "A) XS R16: six final seeds for the top group (AdamW, EMA, Muon, AdamS, Muon+Const, dropout, CWD)" || true
if block_on A; then
seed_journal XS_D10M_E16_AdamW XS_D10M_E16_AdamW_x6
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method AdamW \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid wd=0.1,0.5,2.5 \
  --finals 6 --tag x6 2>&1 | tee -a "$LOG"
seed_journal XS_D10M_E16_AdamW+EMA XS_D10M_E16_AdamW+EMA_x6
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method AdamW+EMA \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid ema=0.9,0.99,0.999 \
  --fixed wd=2.5 \
  --finals 6 --tag x6 2>&1 | tee -a "$LOG"
seed_journal XS_D10M_E16_Muon XS_D10M_E16_Muon_x6
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method Muon \
  --lrs 0.007,0.02,0.06,0.18 \
  --grid wd=0.005,0.02,0.1 \
  --fixed aux_lr=0.0015 \
  --fixed aux_wd=0.1 \
  --finals 6 --tag x6 2>&1 | tee -a "$LOG"
seed_journal XS_D10M_E16_AdamS XS_D10M_E16_AdamS_x6
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method AdamS \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid wd=1e-05,0.0001,0.001 \
  --finals 6 --tag x6 2>&1 | tee -a "$LOG"
seed_journal XS_D10M_E16_Muon+Const XS_D10M_E16_Muon+Const_x6
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method Muon+Const \
  --lrs 0.007,0.02,0.06,0.18 \
  --grid rho=0.0001,0.0005,0.0025 \
  --fixed aux_lr=0.0015 \
  --finals 6 --tag x6 2>&1 | tee -a "$LOG"
seed_journal XS_D10M_E16_AdamW+dropout XS_D10M_E16_AdamW+dropout_x6
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method AdamW+dropout \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid dropout=0.03,0.1,0.3 \
  --fixed wd=2.5 \
  --finals 6 --tag x6 2>&1 | tee -a "$LOG"
seed_journal XS_D10M_E16_CWD XS_D10M_E16_CWD_x6
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method CWD \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid wd=0.1,0.5,2.5 \
  --finals 6 --tag x6 2>&1 | tee -a "$LOG"
fi

run_block B "B) XS R16: second tuning seed (43) on the registered grids of AdamW, EMA, Muon; no finals" || true
if block_on B; then
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method AdamW \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid wd=0.1,0.5,2.5 \
  --finals 0 --tag tune43 \
  --tune-seed 43 2>&1 | tee -a "$LOG"
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method AdamW+EMA \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid ema=0.9,0.99,0.999 \
  --fixed wd=2.5 \
  --finals 0 --tag tune43 \
  --tune-seed 43 2>&1 | tee -a "$LOG"
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method Muon \
  --lrs 0.007,0.02,0.06,0.18 \
  --grid wd=0.005,0.02,0.1 \
  --fixed aux_lr=0.0015 \
  --fixed aux_wd=0.1 \
  --finals 0 --tag tune43 \
  --tune-seed 43 2>&1 | tee -a "$LOG"
fi

run_block C "C) XS R1: a third final seed for all thirteen methods" || true
if block_on C; then
seed_journal XS_D159.384M_E1_Baseline XS_D159.384M_E1_Baseline_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method Baseline \
  --lrs 0.0015 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
seed_journal XS_D159.384M_E1_AdamW XS_D159.384M_E1_AdamW_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method AdamW \
  --lrs 0.0045 \
  --grid wd=0.5,2.5 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
seed_journal XS_D159.384M_E1_AdamW+dropout XS_D159.384M_E1_AdamW+dropout_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method AdamW+dropout \
  --lrs 0.0045 \
  --fixed dropout=0.03 \
  --fixed wd=2.5 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
seed_journal XS_D159.384M_E1_AdamW+EMA XS_D159.384M_E1_AdamW+EMA_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method AdamW+EMA \
  --lrs 0.0045 \
  --grid ema=0.9,0.99 \
  --fixed wd=2.5 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
seed_journal XS_D159.384M_E1_CWD XS_D159.384M_E1_CWD_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method CWD \
  --lrs 0.0045 \
  --grid wd=0.5,2.5 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
seed_journal XS_D159.384M_E1_AdamS XS_D159.384M_E1_AdamS_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method AdamS \
  --lrs 0.0135 \
  --grid wd=2e-05,0.0001 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
seed_journal XS_D159.384M_E1_CPR XS_D159.384M_E1_CPR_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method CPR \
  --lrs 0.0015 \
  --grid warm_start=250,1000 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
seed_journal XS_D159.384M_E1_SF-AdamW XS_D159.384M_E1_SF-AdamW_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method SF-AdamW \
  --lrs 0.0045 \
  --grid wd=0.5,2.5 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
seed_journal XS_D159.384M_E1_Muon XS_D159.384M_E1_Muon_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method Muon \
  --lrs 0.06 \
  --grid wd=0.02,0.1 \
  --fixed aux_lr=0.0015 \
  --fixed aux_wd=0.1 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
seed_journal XS_D159.384M_E1_Muon+Const XS_D159.384M_E1_Muon+Const_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method Muon+Const \
  --lrs 0.06 \
  --grid rho=0.0005,0.0025 \
  --fixed aux_lr=0.0015 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
seed_journal XS_D159.384M_E1_Const XS_D159.384M_E1_Const_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method Const \
  --lrs 0.0045 \
  --grid rho=0.0005,0.0025 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
seed_journal XS_D159.384M_E1_Const-Fair XS_D159.384M_E1_Const-Fair_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method Const-Fair \
  --lrs 0.0045 \
  --grid rho=0.0005,0.0025 \
  --fixed delta=0.1 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
seed_journal XS_D159.384M_E1_Const-Huber XS_D159.384M_E1_Const-Huber_x3
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method Const-Huber \
  --lrs 0.0045 \
  --grid rho=0.0005,0.0025 \
  --fixed delta=0.1 \
  --finals 3 --tag x3 \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
fi

run_block D "D) XS R16: the second extension on the axis where each winner still sits on the edge (finals follow a moved winner)" || true
if block_on D; then
seed_journal XS_D10M_E16_AdamW XS_D10M_E16_AdamW_edge
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method AdamW \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid wd=0.1,0.5,2.5,7.5 \
  --finals 3 --tag edge 2>&1 | tee -a "$LOG"
seed_journal XS_D10M_E16_CWD XS_D10M_E16_CWD_edge
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method CWD \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid wd=0.1,0.5,2.5,7.5 \
  --finals 3 --tag edge 2>&1 | tee -a "$LOG"
seed_journal XS_D10M_E16_Const XS_D10M_E16_Const_edge
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method Const \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid rho=0.0001,0.0005,0.0025,0.0125 \
  --finals 3 --tag edge 2>&1 | tee -a "$LOG"
seed_journal XS_D10M_E16_Const-Fair XS_D10M_E16_Const-Fair_edge
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method Const-Fair \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid rho=0.0001,0.0005,0.0025,0.0125 \
  --fixed delta=0.1 \
  --finals 3 --tag edge 2>&1 | tee -a "$LOG"
seed_journal XS_D10M_E16_Const-Huber XS_D10M_E16_Const-Huber_edge
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method Const-Huber \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid rho=0.0001,0.0005,0.0025,0.0125 \
  --fixed delta=0.1 \
  --finals 3 --tag edge 2>&1 | tee -a "$LOG"
seed_journal XS_D10M_E16_Muon XS_D10M_E16_Muon_edge
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method Muon \
  --lrs 0.007,0.02,0.06,0.18 \
  --grid wd=0.005,0.02,0.1,0.5 \
  --fixed aux_lr=0.0015 \
  --fixed aux_wd=0.1 \
  --finals 3 --tag edge 2>&1 | tee -a "$LOG"
seed_journal XS_D10M_E16_Muon+Const XS_D10M_E16_Muon+Const_edge
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method Muon+Const \
  --lrs 0.007,0.02,0.06,0.18 \
  --grid rho=0.0001,0.0005,0.0025,0.0125 \
  --fixed aux_lr=0.0015 \
  --finals 3 --tag edge 2>&1 | tee -a "$LOG"
fi

run_block E "E) S R1: control arm at 51.5M for AdamW, EMA, Muon (R16-S winner and its weaker variant, two finals)" || true
if block_on E; then
python -u dc_grid.py --size S --unique-tokens 159383553 --epochs 1 --method AdamW \
  --lrs 0.0045 \
  --grid wd=0.5,2.5 \
  --finals 2 --tag r1s \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
python -u dc_grid.py --size S --unique-tokens 159383553 --epochs 1 --method AdamW+EMA \
  --lrs 0.0045 \
  --grid ema=0.99,0.9999 \
  --fixed wd=2.5 \
  --finals 2 --tag r1s \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
python -u dc_grid.py --size S --unique-tokens 159383553 --epochs 1 --method Muon \
  --lrs 0.06 \
  --grid wd=0.02,0.1 \
  --fixed aux_lr=0.0015 \
  --fixed aux_wd=0.1 \
  --finals 2 --tag r1s \
  --evals-per-epoch 32 2>&1 | tee -a "$LOG"
fi

run_block F "F) AdEMAMix (in the registry, outside the protocol): XS R16 on the extended AdamW grid, then R1" || true
if block_on F; then
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method AdEMAMix \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid wd=0.1,0.5,2.5 \
  --finals 3 --tag add 2>&1 | tee -a "$LOG"
W=$(python - <<'EOF'
import json; w = json.load(open('results/dc/XS_D10M_E16_AdEMAMix_add.json'))['winner']['hp']; print(f"{w['lr']:g} {w['wd']:g} {w['wd']/5:g}")
EOF
)
read -r ALR AWD AWDW <<< "$W"
echo "[POD P1:$TAG] AdEMAMix R16 winner lr=$ALR wd=$AWD -> R1 at wd $AWDW,$AWD" | tee -a "$LOG"
python -u dc_grid.py --size XS --unique-tokens 159383553 --epochs 1 --method AdEMAMix \
  --lrs $ALR --grid wd=$AWDW,$AWD --finals 2 --tag add --evals-per-epoch 32 2>&1 | tee -a "$LOG"
fi

run_block G "G) M (124.5M parameters) R16 spot check: AdamW, EMA, Muon at their S-tuned second axis, learning rate re-tuned (3 values), 3 finals" || true
if block_on G; then
python -u dc_grid.py --size M --unique-tokens 10000000 --epochs 16 --method AdamW \
  --lrs 0.0005,0.0015,0.0045 \
  --fixed wd=2.5 \
  --finals 3 --tag m 2>&1 | tee -a "$LOG"
python -u dc_grid.py --size M --unique-tokens 10000000 --epochs 16 --method AdamW+EMA \
  --lrs 0.0005,0.0015,0.0045 \
  --fixed ema=0.9999 \
  --fixed wd=2.5 \
  --finals 3 --tag m 2>&1 | tee -a "$LOG"
python -u dc_grid.py --size M --unique-tokens 10000000 --epochs 16 --method Muon \
  --lrs 0.007,0.02,0.06 \
  --fixed wd=0.1 \
  --fixed aux_lr=0.0015 \
  --fixed aux_wd=0.1 \
  --finals 3 --tag m 2>&1 | tee -a "$LOG"
fi

run_block H "H) CWD follow-up: XS grid closed with wd 37.5 (from the edge journal); S search with wd up to 37.5, three finals" || true
if block_on H; then
seed_journal XS_D10M_E16_CWD_edge XS_D10M_E16_CWD_cwd
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method CWD \
  --lrs 0.0005,0.0015,0.0045,0.0135 \
  --grid wd=0.1,0.5,2.5,7.5,37.5 \
  --finals 3 --tag cwd 2>&1 | tee -a "$LOG"
python -u dc_grid.py --size S --unique-tokens 10000000 --epochs 16 --method CWD \
  --lrs 0.0015,0.0045,0.0135 \
  --grid wd=2.5,7.5,37.5 \
  --finals 3 --tag cwd 2>&1 | tee -a "$LOG"
fi

run_block I "I) AdEMAMix with alpha {2,4} and beta3 {0.999,0.9999} tuned, wd 2.5, lr {1.5e-4,5e-4,1.5e-3}, three finals (tag add2)" || true
if block_on I; then
python -u dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method AdEMAMix-T \
  --lrs 0.00015,0.0005,0.0015 \
  --grid alpha=2,4 \
  --grid beta3=0.999,0.9999 \
  --fixed wd=2.5 \
  --finals 3 --tag add2 2>&1 | tee -a "$LOG"
fi

echo "[POD P1:$TAG] $(date -u) done (part $PART) -> results/dc/" | tee -a "$LOG"
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

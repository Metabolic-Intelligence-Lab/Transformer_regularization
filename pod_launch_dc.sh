#!/usr/bin/env bash
# ============================================================================
# POD DC — data-constrained LM study (dcstudy/, dc_grid.py)
# ----------------------------------------------------------------------------
#   bash pod_launch_dc.sh <tag> -- <dc_grid.py arguments> [-- <dc_grid.py arguments> ...]
# Runs one or more dc_grid.py invocations in sequence (each "--" starts a new one).
# Requires data/dc/{train,val,test}.bin + meta.json uploaded beforehand (scp); their
# SHA-256 is checked against meta.json before anything runs.
#
# SAFE TO STOP AT ANY TIME: re-running resumes from results/journal/dc_*.jsonl.
# ============================================================================
set -euo pipefail
TAG=${1:?tag}; shift
source "$(dirname "$0")/pod_common.sh"
pod_provision
pip install -q "schedulefree==1.4.1" "pytorch-cpr==0.3.1"
cd "$REPO"

# The token files live on the persistent volume, OUTSIDE the repo directory that
# pod_bootstrap.sh wipes and re-creates: upload them once to /workspace/data/dc
# (scp -r ~/regdecay_data/dc160/ root@<ip>:/workspace/data/dc) and they are linked in here.
if [ ! -e data/dc ] && [ -d /workspace/data/dc ]; then mkdir -p data && ln -s /workspace/data/dc data/dc; fi
[ -f data/dc/meta.json ] || { echo '[DATA] data/dc missing: upload the corpus to /workspace/data/dc first'; exit 1; }

python - <<'EOF'
import hashlib, json, sys
meta = json.load(open('data/dc/meta.json'))
pinned = json.load(open('dcstudy/data_meta.json'))     # the corpus the protocol pins
for split, info in meta['files'].items():
    h = hashlib.sha256(open(f'data/dc/{split}.bin', 'rb').read()).hexdigest()
    if h != info['sha256']:
        sys.exit(f'[DATA] {split}.bin sha256 mismatch with data/dc/meta.json')
    if h != pinned['files'][split]['sha256']:
        sys.exit(f'[DATA] {split}.bin is not the corpus pinned in dcstudy/data_meta.json')
print('[DATA] train/val/test sha256 OK and pinned, revision', meta['revision'])
EOF

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  python -m pytest tests/test_dcstudy.py -q
  SM=/workspace/dcsmoke; rm -rf "$SM"; mkdir -p "$SM"
  ( cd "$SM" && PYTHONPATH="$REPO" python -u "$REPO/dc_grid.py" --smoke --method Const-Fair \
      --lrs 3e-3 --grid rho=1e-4 --grid delta=0.1 --finals 1 )
  rm -rf "$SM"
  echo "[POD DC:$TAG] preflight OK $(date -u)"
fi

[ "${1:-}" = "--" ] && shift
while [ $# -gt 0 ]; do
  ARGS=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do ARGS+=("$1"); shift; done
  [ $# -gt 0 ] && shift
  echo "[POD DC:$TAG] $(date -u) dc_grid.py ${ARGS[*]}"
  python -u dc_grid.py "${ARGS[@]}" 2>&1 | tee -a "logs/pod_${TAG}.log"
done
echo "[POD DC:$TAG] $(date -u) done -> results/dc/, results/journal/dc_*.jsonl"

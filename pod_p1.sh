#!/usr/bin/env bash
# ============================================================================
# P1 (protocols/P1_minibench.md): create a pod, upload the pinned corpus, launch a step.
# ----------------------------------------------------------------------------
#   RUNPOD_API_KEY=... bash pod_p1.sh pilot                 # step P1a: pod_launch_P1_pilot.sh
#   RUNPOD_API_KEY=... bash pod_p1.sh <tag> <launch script> [args...]   # e.g. pod_launch_P1_matrix.sh XS all r16 r1
#
# 1. the working tree must be committed: the pod runs `git archive HEAD` (protocol first, runs after);
# 2. creates one RTX 4090 pod (falls back to the other GPUs below) and records it in pods.conf;
# 3. uploads ~/regdecay_data/dc160 (the corpus pinned in dcstudy/data_meta.json) to /workspace/data/dc
#    on the pod's persistent volume, outside the repo directory that pod_bootstrap.sh re-creates;
# 4. bootstraps the launch script in tmux. Idempotent: a tag already in pods.conf is reused.
# AUTOSTOP=1 also uploads ~/.runpod_key to /root/.runpod_key (mode 600) so the launch script can
# STOP the pod when it is done (no idle billing); the volume with results and corpus survives.
# Afterwards: bash pod_status.sh --pull <tag>; python analysis/p1_gate.py; bash pod_finish.sh <tag> dc '{...}'
# ============================================================================
set -euo pipefail
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
STEP=${1:?pilot | <tag>}
if [ "$STEP" = "pilot" ]; then TAG=P1_pilot; SCRIPT=pod_launch_P1_pilot.sh; shift; else TAG=$STEP; SCRIPT=${2:?launch script}; shift 2; fi
# anything left is passed to the launch script after the tag
DATA=${DATA:-$HOME/regdecay_data/dc160}
# Priority order; community availability changes by the minute, so the list is long and the
# caller may loop (bash pod_p1.sh is idempotent). CLOUD=SECURE selects the secure cloud.
GPUS=(--gpu "NVIDIA GeForce RTX 4090" --gpu "NVIDIA GeForce RTX 3090" --gpu "NVIDIA RTX A5000"
      --gpu "NVIDIA RTX A6000" --gpu "NVIDIA RTX A4500" --gpu "NVIDIA L4" --gpu "NVIDIA A40"
      --gpu "NVIDIA RTX 4000 Ada Generation" --gpu "NVIDIA GeForce RTX 4080")
CLOUD=${CLOUD:-COMMUNITY}
SO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30"

[ -f "$SCRIPT" ] || { echo "[P1] no $SCRIPT"; exit 1; }
[ -f "$DATA/meta.json" ] || { echo "[P1] no corpus at $DATA (python -m dcstudy.prepare_data ...)"; exit 1; }
python3 - "$DATA" <<'PY'
import json, sys
a = json.load(open(sys.argv[1] + '/meta.json'))['files']; b = json.load(open('dcstudy/data_meta.json'))['files']
bad = [s for s in a if a[s]['sha256'] != b[s]['sha256']]
sys.exit(f"[P1] corpus at {sys.argv[1]} is not the one pinned in dcstudy/data_meta.json: {bad}" if bad else 0)
PY
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "[P1] uncommitted changes: commit first (the pod runs the committed tree)"; exit 1
fi
SHA=$(git rev-parse --short HEAD); TGZ=/tmp/repo_${SHA}.tgz
# pod_bootstrap.sh extracts the tarball in /workspace and expects the top-level directory
# Regularization_Decay/ (the name pod_common.sh uses as $REPO).
git archive --format=tar.gz --prefix=Regularization_Decay/ -o "$TGZ" HEAD \
  && echo "[P1] tarball $TGZ ($(du -h "$TGZ" | cut -f1), commit $SHA)"

if grep -q "^$TAG " pods.conf 2>/dev/null; then
  echo "[P1] $TAG already in pods.conf - reusing the pod"
else
  python3 pod_create.py --tag "$TAG" --stem dc --cloud "$CLOUD" "${GPUS[@]}"
fi
line=$(grep "^$TAG " pods.conf | tail -1); ip=$(awk '{print $2}' <<<"$line"); port=$(awk '{print $3}' <<<"$line")
[ -n "$ip" ] && [ -n "$port" ] || { echo "[P1] no ip/port for $TAG in pods.conf"; exit 1; }

ok=0
for i in 1 2 3 4 5 6 7 8 9 10; do
  ssh $SO -o BatchMode=yes -p "$port" "root@$ip" 'echo ssh-ok' && { ok=1; break; }
  echo "[P1] ssh not ready (attempt $i), retrying in 30 s"; sleep 30
done
[ "$ok" -eq 1 ] || { echo "[P1] cannot ssh to root@$ip:$port"; exit 1; }
# Upload the corpus once (330 MB); skipped when the pod already holds the pinned files.
if ! ssh $SO -p "$port" "root@$ip" "cmp -s /workspace/data/dc/meta.json - " < "$DATA/meta.json"; then
  echo "[P1] uploading corpus to /workspace/data/dc ..."
  ssh $SO -p "$port" "root@$ip" 'mkdir -p /workspace/data/dc'
  scp $SO -P "$port" "$DATA"/train.bin "$DATA"/val.bin "$DATA"/test.bin "$DATA"/meta.json "root@$ip:/workspace/data/dc/"
else
  echo "[P1] corpus already on the pod"
fi
if [ "${AUTOSTOP:-0}" = "1" ]; then
  scp $SO -P "$port" "$HOME/.runpod_key" "root@$ip:/root/.runpod_key" && ssh $SO -p "$port" "root@$ip" 'chmod 600 /root/.runpod_key' \
    && echo "[P1] auto-stop enabled (key on the pod)"
  # the pod id too: RUNPOD_POD_ID is not guaranteed inside the ssh/tmux session (P1_xs never stopped)
  pid=$(grep "^# $TAG id=" pods.conf | tail -1 | sed -E 's/.*id=([a-z0-9]+).*/\1/')
  [ -n "$pid" ] && ssh $SO -p "$port" "root@$ip" "echo $pid > /root/.runpod_pod_id" && echo "[P1] pod id $pid on the pod"
fi
BOOT_TAG="$TAG" bash pod_bootstrap.sh "$ip" "$port" "$TGZ" "$SCRIPT" "$TAG" "$@"
echo "[P1] $TAG launched on $ip:$port ($(date -u)); follow: ssh -p $port root@$ip 'tail -f /workspace/Regularization_Decay/logs/pod_${TAG}_full.log'"

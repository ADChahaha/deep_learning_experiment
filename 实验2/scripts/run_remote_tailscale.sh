#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_remote_tailscale.sh <tailscale-host> [remote-python]

Examples:
  ./scripts/run_remote_tailscale.sh gpu-box.tailnet.ts.net
  ./scripts/run_remote_tailscale.sh 100.101.102.103 /opt/venvs/dl/bin/python

Environment variables:
  REMOTE_BASE_TMP   Remote parent temp dir. Default: /tmp
  REMOTE_SSH_PORT   SSH port. Default: 22
  REMOTE_ACTIVATE   Remote shell snippet executed before training, e.g.:
                    'source ~/miniconda3/etc/profile.d/conda.sh && conda activate torch'
  LOCAL_RETURN_DIR  Local folder used to store returned artifacts.
                    Default: <experiment>/outputs/remote_runs/<timestamp>
  RSYNC_RSH         Custom remote shell for rsync. Default uses ssh with StrictHostKeyChecking=accept-new

What it does:
  1. Creates a temporary remote working directory.
  2. Copies code, config, data, and pretrained checkpoint to the remote machine.
  3. Runs `python -m src.train` inside that temporary directory.
  4. Pulls back logs, outputs, and checkpoints.
  5. Deletes the remote temporary directory.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_HOST="$1"
REMOTE_PYTHON="${2:-python3}"
REMOTE_BASE_TMP="${REMOTE_BASE_TMP:-/tmp}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-22}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOCAL_RETURN_DIR="${LOCAL_RETURN_DIR:-${EXP_DIR}/outputs/remote_runs/${TIMESTAMP}}"
SSH_OPTS=(
  -p "${REMOTE_SSH_PORT}"
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=10
)
SSH_DEST="${REMOTE_HOST}"
RSYNC_RSH="${RSYNC_RSH:-ssh ${SSH_OPTS[*]}}"

mkdir -p "${LOCAL_RETURN_DIR}"

echo "[1/6] Creating remote temporary directory on ${REMOTE_HOST}..."
REMOTE_TMPDIR="$(
  ssh "${SSH_OPTS[@]}" "${SSH_DEST}" \
    "mktemp -d '${REMOTE_BASE_TMP%/}/exp2_tailscale_XXXXXX'"
)"
echo "Remote temp dir: ${REMOTE_TMPDIR}"

cleanup_remote() {
  if [[ -n "${REMOTE_TMPDIR:-}" ]]; then
    ssh "${SSH_OPTS[@]}" "${SSH_DEST}" "rm -rf '${REMOTE_TMPDIR}'" >/dev/null 2>&1 || true
  fi
}

trap cleanup_remote EXIT

echo "[2/6] Syncing experiment files..."
rsync -az --delete -e "${RSYNC_RSH}" \
  --exclude '.DS_Store' \
  --exclude '__pycache__/' \
  --exclude 'outputs/' \
  --exclude '*.pyc' \
  "${EXP_DIR}/src" \
  "${EXP_DIR}/scripts" \
  "${EXP_DIR}/config.json" \
  "${EXP_DIR}/generate_report.py" \
  "${EXP_DIR}/requirements-remote.txt" \
  "${EXP_DIR}/data" \
  "${EXP_DIR}/checkpoints" \
  "${SSH_DEST}:${REMOTE_TMPDIR}/"

echo "[3/6] Writing remote runner..."
ssh "${SSH_OPTS[@]}" "${SSH_DEST}" "mkdir -p '${REMOTE_TMPDIR}/outputs' '${REMOTE_TMPDIR}/checkpoints'"
ssh "${SSH_OPTS[@]}" "${SSH_DEST}" "cat > '${REMOTE_TMPDIR}/run_train.sh' <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_TMPDIR}'
export PYTHONUNBUFFERED=1
if [[ -n \"\${REMOTE_ACTIVATE:-}\" ]]; then
  eval \"\${REMOTE_ACTIVATE}\"
fi
echo \"Host: \$(hostname)\"
echo \"Workdir: \$(pwd)\"
echo \"Python: $('${REMOTE_PYTHON}' -V 2>&1)\"
echo \"Start: \$(date '+%Y-%m-%d %H:%M:%S')\"
'${REMOTE_PYTHON}' -m src.train 2>&1 | tee outputs/remote_train.log
echo \"End: \$(date '+%Y-%m-%d %H:%M:%S')\" | tee -a outputs/remote_train.log
EOF
chmod +x '${REMOTE_TMPDIR}/run_train.sh'"

echo "[4/6] Starting remote training..."
ssh "${SSH_OPTS[@]}" "${SSH_DEST}" "REMOTE_ACTIVATE='${REMOTE_ACTIVATE:-}' '${REMOTE_TMPDIR}/run_train.sh'"

echo "[5/6] Pulling back logs, checkpoints, and outputs..."
mkdir -p "${LOCAL_RETURN_DIR}/outputs" "${LOCAL_RETURN_DIR}/checkpoints"
rsync -az -e "${RSYNC_RSH}" \
  "${SSH_DEST}:${REMOTE_TMPDIR}/outputs/" \
  "${LOCAL_RETURN_DIR}/outputs/"
rsync -az -e "${RSYNC_RSH}" \
  "${SSH_DEST}:${REMOTE_TMPDIR}/checkpoints/" \
  "${LOCAL_RETURN_DIR}/checkpoints/"
rsync -az -e "${RSYNC_RSH}" \
  "${SSH_DEST}:${REMOTE_TMPDIR}/config.json" \
  "${LOCAL_RETURN_DIR}/config.remote.json"

echo "[6/6] Cleaning up remote files..."
cleanup_remote
trap - EXIT

cat <<EOF
Finished.

Returned files:
  ${LOCAL_RETURN_DIR}/outputs
  ${LOCAL_RETURN_DIR}/checkpoints
  ${LOCAL_RETURN_DIR}/config.remote.json

Useful log:
  ${LOCAL_RETURN_DIR}/outputs/remote_train.log
EOF

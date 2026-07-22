#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_remote_tailscale_asus_gitbash.sh [start|fetch|cleanup]

Modes:
  start    Sync code/config to Asus and start training in background.
  fetch    Pull outputs/checkpoints from a saved remote run directory.
  cleanup  Delete a saved remote run directory after successful fetch.

Environment variables:
  REMOTE_HOST        Default: asus.tailc3e9be.ts.net
  REMOTE_USER        Default: asus
  REMOTE_GIT_BASH    Default: C:\Program Files\Git\bin\bash.exe
  REMOTE_PYTHON      Default: /c/Users/14195/miniconda3/envs/segexp/python.exe
  REMOTE_DATA_ROOT   Default: C:/Users/14195/claude_segexp/data
  REMOTE_RUNS_ROOT   Default: /c/Users/14195/claude_segexp/remote_runs
  REMOTE_RUN_DIR     Required for fetch/cleanup
  LOCAL_RETURN_DIR   Default: <experiment>/outputs/remote_runs/<timestamp> for fetch

Notes:
  - Data is not re-uploaded; the Asus cache is reused.
  - Results are kept on Asus until cleanup is run manually.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODE="${1:-start}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
REMOTE_HOST="${REMOTE_HOST:-asus.tailc3e9be.ts.net}"
REMOTE_USER="${REMOTE_USER:-asus}"
REMOTE_GIT_BASH="${REMOTE_GIT_BASH:-C:\\Program Files\\Git\\bin\\bash.exe}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/c/Users/14195/miniconda3/envs/segexp/python.exe}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-C:/Users/14195/claude_segexp/data}"
REMOTE_RUNS_ROOT="${REMOTE_RUNS_ROOT:-/c/Users/14195/claude_segexp/remote_runs}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-}"
LOCAL_RETURN_DIR="${LOCAL_RETURN_DIR:-${EXP_DIR}/outputs/remote_runs/${TIMESTAMP}}"
LOCAL_STAGE_DIR="$(mktemp -d /tmp/exp2_remote_stage_XXXXXX)"
SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=8
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=10
)
SSH_DEST="${REMOTE_USER}@${REMOTE_HOST}"

remote_gitbash() {
  local script="$1"
  ssh "${SSH_OPTS[@]}" "${SSH_DEST}" "\"${REMOTE_GIT_BASH}\" -lc \"$script\""
}

cleanup_local() {
  rm -rf "${LOCAL_STAGE_DIR}"
}

trap cleanup_local EXIT

case "${MODE}" in
  start)
    echo "[1/4] Creating persistent remote run directory on ${SSH_DEST}..."
    REMOTE_RUN_DIR="$(
      remote_gitbash "mkdir -p '${REMOTE_RUNS_ROOT}' && mktemp -d '${REMOTE_RUNS_ROOT}/exp2_XXXXXX'"
    )"
    echo "Remote run dir: ${REMOTE_RUN_DIR}"

    echo "[2/4] Preparing lightweight sync bundle..."
    mkdir -p "${LOCAL_STAGE_DIR}"
    cp -R "${EXP_DIR}/src" "${LOCAL_STAGE_DIR}/src"
    cp -R "${EXP_DIR}/scripts" "${LOCAL_STAGE_DIR}/scripts"
    cp "${EXP_DIR}/generate_report.py" "${LOCAL_STAGE_DIR}/generate_report.py"
    cp "${EXP_DIR}/requirements-remote.txt" "${LOCAL_STAGE_DIR}/requirements-remote.txt"
    cat > "${LOCAL_STAGE_DIR}/run_train_remote.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_RUN_DIR}'
mkdir -p outputs checkpoints
rm -f outputs/train.exitcode
export PYTHONUNBUFFERED=1
'${REMOTE_PYTHON}' -m src.train > outputs/remote_train.log 2>&1
printf '%s' \$? > outputs/train.exitcode
EOF
    chmod +x "${LOCAL_STAGE_DIR}/run_train_remote.sh"
    cat > "${LOCAL_STAGE_DIR}/launch_train_remote.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_RUN_DIR}'
mkdir -p outputs checkpoints
rm -f outputs/train.pid outputs/train.exitcode
nohup bash ./run_train_remote.sh >/dev/null 2>&1 &
printf '%s' "\$!" | tee outputs/train.pid >/dev/null
EOF
    chmod +x "${LOCAL_STAGE_DIR}/launch_train_remote.sh"
    python3 - "${EXP_DIR}/config.json" "${LOCAL_STAGE_DIR}/config.json" "${REMOTE_DATA_ROOT}" <<'PY'
import json
import sys

src, dst, data_root = sys.argv[1:]
with open(src, "r", encoding="utf-8") as f:
    config = json.load(f)
config["data"]["root"] = data_root
config["data"]["download"] = False
with open(dst, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
PY

    echo "[3/4] Streaming code/config files to remote..."
    COPYFILE_DISABLE=1 tar -C "${LOCAL_STAGE_DIR}" \
      --exclude='._*' \
      --exclude='__pycache__' \
      --exclude='.DS_Store' \
      -cf - . \
      | ssh "${SSH_OPTS[@]}" "${SSH_DEST}" "\"${REMOTE_GIT_BASH}\" -lc \"mkdir -p '${REMOTE_RUN_DIR}' && tar -C '${REMOTE_RUN_DIR}' -xf -\""

    echo "[4/4] Starting remote training in background..."
    remote_gitbash "cd '${REMOTE_RUN_DIR}' && bash ./launch_train_remote.sh"

    cat <<EOF
Started.

Remote run dir:
  ${REMOTE_RUN_DIR}

Useful files on Asus:
  ${REMOTE_RUN_DIR}/outputs/remote_train.log
  ${REMOTE_RUN_DIR}/outputs/train.exitcode
  ${REMOTE_RUN_DIR}/checkpoints

When training finishes, fetch with:
  REMOTE_RUN_DIR='${REMOTE_RUN_DIR}' bash scripts/run_remote_tailscale_asus_gitbash.sh fetch

After verifying local files, delete remote copy with:
  REMOTE_RUN_DIR='${REMOTE_RUN_DIR}' bash scripts/run_remote_tailscale_asus_gitbash.sh cleanup
EOF
    ;;

  fetch)
    if [[ -z "${REMOTE_RUN_DIR}" ]]; then
      echo "REMOTE_RUN_DIR is required for fetch." >&2
      exit 1
    fi
    echo "[1/2] Pulling outputs and checkpoints from ${REMOTE_RUN_DIR}..."
    mkdir -p "${LOCAL_RETURN_DIR}"
    ssh "${SSH_OPTS[@]}" "${SSH_DEST}" "\"${REMOTE_GIT_BASH}\" -lc \"cd '${REMOTE_RUN_DIR}' && tar -cf - outputs checkpoints config.json\"" \
      | tar -C "${LOCAL_RETURN_DIR}" -xf -
    echo "[2/2] Verifying local files..."
    test -f "${LOCAL_RETURN_DIR}/outputs/remote_train.log"
    test -f "${LOCAL_RETURN_DIR}/outputs/train.exitcode"
    echo "Fetched to: ${LOCAL_RETURN_DIR}"
    ;;

  cleanup)
    if [[ -z "${REMOTE_RUN_DIR}" ]]; then
      echo "REMOTE_RUN_DIR is required for cleanup." >&2
      exit 1
    fi
    echo "Deleting remote run dir: ${REMOTE_RUN_DIR}"
    remote_gitbash "rm -rf '${REMOTE_RUN_DIR}'"
    ;;

  *)
    usage >&2
    exit 1
    ;;
esac

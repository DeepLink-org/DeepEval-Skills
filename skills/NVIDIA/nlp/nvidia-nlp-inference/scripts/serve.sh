#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${MODEL_NAME:-deepseek-r1-0528}"
CONFIG_FILE="${SCRIPT_DIR}/configs/${MODEL_NAME}.sh"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[serve.sh] Unknown MODEL_NAME=${MODEL_NAME}. Available configs:" >&2
  ls "${SCRIPT_DIR}/configs" | sed 's/\.sh$//' >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

: "${MODEL_PATH:?MODEL_PATH not set in ${CONFIG_FILE}}"
: "${TP:?TP not set in ${CONFIG_FILE}}"
: "${PORT:?PORT not set in ${CONFIG_FILE}}"

echo "[serve.sh] MODEL_NAME=${MODEL_NAME}  TP=${TP}  PORT=${PORT}"
echo "[serve.sh] MODEL_PATH=${MODEL_PATH}"

mkdir -p logs

# shellcheck disable=SC2086   # EXTRA_SERVE_ARGS is intentionally word-split
python3 -m sglang.launch_server \
  --model "${MODEL_PATH}" \
  --tp "${TP}" \
  --port "${PORT}" \
  ${EXTRA_SERVE_ARGS:-} 2>&1 | tee "./logs/serve_${MODEL_NAME}.log"

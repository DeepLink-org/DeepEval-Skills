#!/bin/bash
# Open-Sora v2 text-to-video inference, 1 GPU.
# Switch resolution with MM_T2V_RESOLUTION=256px or 768px.

set -e

RESOLUTION=${MM_T2V_RESOLUTION:-256px}
if [ "$#" -gt 0 ]; then
    if [ "$1" = "--resolution" ]; then
        RESOLUTION=${2:-$RESOLUTION}
    else
        RESOLUTION=$1
    fi
fi
case "$RESOLUTION" in
    256) RESOLUTION=256px ;;
    768) RESOLUTION=768px ;;
    256px|768px) ;;
    *)
        echo "Unsupported resolution: ${RESOLUTION}. Use 256px or 768px." >&2
        exit 1
        ;;
esac

OPENSORA_DIR=${MM_T2V_OPENSORA_DIR:-/workspace/code/Open-Sora}
LOG_DIR=${MM_T2V_LOG_DIR:-/workspace/logs}
PROMPT=${MM_T2V_PROMPT:-raining, sea}
OFFLOAD=${MM_T2V_OFFLOAD:-True}

mkdir -p "$LOG_DIR"
cd "$OPENSORA_DIR"

LOG_FILE="${LOG_DIR}/opensora_${RESOLUTION}_gpus1.log"

torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py "configs/diffusion/inference/${RESOLUTION}.py" \
--prompt "$PROMPT" \
--offload "$OFFLOAD" > "$LOG_FILE" 2>&1

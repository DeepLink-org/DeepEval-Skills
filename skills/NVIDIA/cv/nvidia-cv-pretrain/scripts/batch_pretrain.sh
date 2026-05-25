#!/usr/bin/env bash
set -euo pipefail

# Pretrain/classification benchmark. GPU count/model/precision can be controlled by env.

CV_PRE_MMPRE_DIR="${CV_PRE_MMPRE_DIR:-/workspace/code/onedl-mmpretrain}"
CV_PRE_MMCV_DIR="${CV_PRE_MMCV_DIR:-/workspace/code/onedl-mmcv}"
CV_PRE_LOG_DIR="${CV_PRE_LOG_DIR:-/workspace/logs}"
CV_PRE_DATASET_DIR="${CV_PRE_DATASET_DIR:-/workspace/datasets/imagenet}"
CV_PRE_NGPU="${CV_PRE_NGPU:-${CARD_COUNT:-1}}"
CV_PRE_MODELS="${CV_PRE_MODELS:-resnet50}"
CV_PRE_PRECISIONS="${CV_PRE_PRECISIONS:-fp16,fp32}"
PYTHON="${PYTHON:-/usr/bin/python3}"

cd "$CV_PRE_MMPRE_DIR"

export MMPRE_PATH="$CV_PRE_MMPRE_DIR"
export MMCV_PATH="$CV_PRE_MMCV_DIR"
export SYSTEM_PACKAGES="${SYSTEM_PACKAGES:-/usr/local/lib/python3.10/dist-packages}"
export PYTHONPATH="$MMPRE_PATH:$MMCV_PATH:$SYSTEM_PACKAGES:${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

if [ -f /workspace/tools/custom_iter_timer_hook.py ]; then
    cp /workspace/tools/custom_iter_timer_hook.py "$CV_PRE_MMPRE_DIR/custom_iter_timer_hook.py"
fi

mkdir -p "$CV_PRE_LOG_DIR"
if [ -d "$CV_PRE_DATASET_DIR" ]; then
    mkdir -p data
    ln -sfn "$CV_PRE_DATASET_DIR" data/imagenet
fi

model_config() {
    case "$1" in
        resnet50) echo "configs/resnet/resnet50_8xb32_in1k.py" ;;
        inception_v3) echo "configs/inception_v3/inception-v3_8xb32_in1k.py" ;;
        seresnet50) echo "configs/seresnet/seresnet50_8xb32_in1k.py" ;;
        mobilenet_v2) echo "configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py" ;;
        shufflenet_v2) echo "configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py" ;;
        densenet121) echo "configs/densenet/densenet121_4xb256_in1k.py" ;;
        swin_large) echo "configs/swin_transformer/swin-large_16xb64_in1k.py" ;;
        efficientnet_b2) echo "configs/efficientnet/efficientnet-b2_8xb32_in1k.py" ;;
        *) echo "Unsupported model: $1" >&2; return 1 ;;
    esac
}

IFS=',' read -r -a MODELS <<< "$CV_PRE_MODELS"
IFS=',' read -r -a PRECISIONS <<< "$CV_PRE_PRECISIONS"
TOTAL=${#MODELS[@]}
COUNT=0

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_NAME="$(echo "$MODEL_NAME" | xargs)"
    CONFIG="$(model_config "$MODEL_NAME")"
    COUNT=$((COUNT + 1))

    test -f "$CONFIG"

    for PRECISION in "${PRECISIONS[@]}"; do
        PRECISION="$(echo "$PRECISION" | xargs)"
        if [ "$PRECISION" = "fp16" ]; then
            AMP_OPT="optim_wrapper.type=AmpOptimWrapper"
        elif [ "$PRECISION" = "fp32" ]; then
            AMP_OPT="optim_wrapper.type=OptimWrapper"
        else
            echo "Unsupported precision: $PRECISION" >&2
            exit 1
        fi

        echo ""
        echo "============================================================"
        echo "  [${COUNT}/${TOTAL}] ${MODEL_NAME} - ${PRECISION} - ${CV_PRE_NGPU} GPUs"
        echo "============================================================"

        "$PYTHON" -m torch.distributed.launch \
            --nnodes="${NODE_COUNT:-1}" \
            --node-rank="${NODE_RANK:-0}" \
            --master-addr="${MASTER_ADDR:-127.0.0.1}" \
            --nproc_per_node="${CV_PRE_NGPU}" \
            --master-port="${MASTER_PORT:-29500}" \
            tools/train.py "$CONFIG" \
            --launcher pytorch \
            --work-dir "${CV_PRE_LOG_DIR}/${MODEL_NAME}_gpus${CV_PRE_NGPU}_${PRECISION}" \
            --cfg-options \
                "$AMP_OPT"

        echo "[${COUNT}/${TOTAL}] ${MODEL_NAME} ${PRECISION} done."
    done
done

echo ""
echo "========== All pretrain tests finished =========="

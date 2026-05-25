#!/usr/bin/env bash
set -euo pipefail

# Segmentation benchmark. GPU count/model/precision can be controlled by env.

CV_SEG_MMSEG_DIR="${CV_SEG_MMSEG_DIR:-/workspace/code/onedl-mmsegmentation}"
CV_SEG_MMCV_DIR="${CV_SEG_MMCV_DIR:-/workspace/code/onedl-mmcv}"
CV_SEG_LOG_DIR="${CV_SEG_LOG_DIR:-/workspace/logs}"
CV_SEG_WEIGHT_PATH="${CV_SEG_WEIGHT_PATH:-/workspace/weight/resnet50_v1c-2cccc1ad.pth}"
CV_SEG_NGPU="${CV_SEG_NGPU:-${CARD_COUNT:-1}}"
CV_SEG_MODELS="${CV_SEG_MODELS:-fcn}"
CV_SEG_PRECISIONS="${CV_SEG_PRECISIONS:-fp16,fp32}"
PYTHON="${PYTHON:-/usr/bin/python3}"

cd "$CV_SEG_MMSEG_DIR"

export MMSEG_PATH="$CV_SEG_MMSEG_DIR"
export MMCV_PATH="$CV_SEG_MMCV_DIR"
export SYSTEM_PACKAGES="${SYSTEM_PACKAGES:-/usr/local/lib/python3.10/dist-packages}"
export PYTHONPATH="$MMSEG_PATH:$MMCV_PATH:$SYSTEM_PACKAGES:${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

mkdir -p "$CV_SEG_LOG_DIR"

model_config() {
    case "$1" in
        deeplabv3) echo "configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py" ;;
        fcn) echo "configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py" ;;
        pspnet) echo "configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py" ;;
        apcnet) echo "configs/apcnet/apcnet_r50-d8_4xb2-40k_cityscapes-512x1024.py" ;;
        *) echo "Unsupported model: $1" >&2; return 1 ;;
    esac
}

IFS=',' read -r -a MODELS <<< "$CV_SEG_MODELS"
IFS=',' read -r -a PRECISIONS <<< "$CV_SEG_PRECISIONS"
TOTAL=${#MODELS[@]}
COUNT=0

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_NAME="$(echo "$MODEL_NAME" | xargs)"
    CONFIG="$(model_config "$MODEL_NAME")"
    COUNT=$((COUNT + 1))

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
        echo "  [${COUNT}/${TOTAL}] ${MODEL_NAME} - ${PRECISION} - ${CV_SEG_NGPU} GPUs"
        echo "============================================================"

        "$PYTHON" -m torch.distributed.launch \
            --nnodes=1 \
            --nproc_per_node="${CV_SEG_NGPU}" \
            --master_port="${MASTER_PORT:-29500}" \
            tools/train.py "$CONFIG" \
            --launcher pytorch \
            --work-dir "${CV_SEG_LOG_DIR}/${MODEL_NAME}_gpus${CV_SEG_NGPU}_${PRECISION}" \
            --cfg-options \
                model.backbone.init_cfg.type=Pretrained \
                model.backbone.init_cfg.checkpoint="${CV_SEG_WEIGHT_PATH}" \
                model.pretrained=None \
                "$AMP_OPT"

        echo "[${COUNT}/${TOTAL}] ${MODEL_NAME} ${PRECISION} done."
    done
done

echo ""
echo "========== All segmentation tests finished =========="

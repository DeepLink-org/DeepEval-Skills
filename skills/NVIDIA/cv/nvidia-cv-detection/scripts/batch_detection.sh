#!/usr/bin/env bash
set -euo pipefail

# Detection benchmark. GPU count/model/precision can be controlled by env.

CV_DET_MMDET_DIR="${CV_DET_MMDET_DIR:-/workspace/code/onedl-mmdetection}"
CV_DET_MMCV_DIR="${CV_DET_MMCV_DIR:-/workspace/code/onedl-mmcv}"
CV_DET_LOG_DIR="${CV_DET_LOG_DIR:-/workspace/logs}"
CV_DET_WEIGHT_DIR="${CV_DET_WEIGHT_DIR:-/workspace/weight}"
CV_DET_DATASET_DIR="${CV_DET_DATASET_DIR:-/workspace/datasets/coco}"
CV_DET_NGPU="${CV_DET_NGPU:-${CARD_COUNT:-1}}"
CV_DET_MODELS="${CV_DET_MODELS:-faster_rcnn}"
CV_DET_PRECISIONS="${CV_DET_PRECISIONS:-fp16,fp32}"
PYTHON="${PYTHON:-/usr/bin/python3}"

cd "$CV_DET_MMDET_DIR"

export MMDET_PATH="$CV_DET_MMDET_DIR"
export MMCV_PATH="$CV_DET_MMCV_DIR"
export SYSTEM_PACKAGES="${SYSTEM_PACKAGES:-/usr/local/lib/python3.10/dist-packages}"
export PYTHONPATH="$MMDET_PATH:$MMCV_PATH:$SYSTEM_PACKAGES:${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

if [ -f /workspace/tools/custom_iter_timer_hook.py ]; then
    cp /workspace/tools/custom_iter_timer_hook.py "$CV_DET_MMDET_DIR/custom_iter_timer_hook.py"
fi

mkdir -p "$CV_DET_LOG_DIR"
if [ -d "$CV_DET_DATASET_DIR" ]; then
    mkdir -p data
    ln -sfn "$CV_DET_DATASET_DIR" data/coco
fi

model_config() {
    case "$1" in
        faster_rcnn) echo "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py" ;;
        mask_rcnn) echo "configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py" ;;
        cascade_rcnn) echo "configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py" ;;
        retinanet) echo "configs/retinanet/retinanet_r50_fpn_1x_coco.py" ;;
        yolov3) echo "configs/yolo/yolov3_d53_8xb8-320-273e_coco.py" ;;
        fcos) echo "configs/fcos/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py" ;;
        ssd300) echo "configs/ssd/ssd300_coco.py" ;;
        centernet) echo "configs/centernet/centernet_r18_8xb16-crop512-140e_coco.py" ;;
        solo) echo "configs/solo/decoupled-solo_r50_fpn_1x_coco.py" ;;
        swin_mask_rcnn) echo "configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py" ;;
        *) echo "Unsupported model: $1" >&2; return 1 ;;
    esac
}

model_weight() {
    case "$1" in
        faster_rcnn|mask_rcnn|cascade_rcnn|retinanet|solo) echo "${CV_DET_WEIGHT_DIR}/resnet50-0676ba61.pth" ;;
        yolov3) echo "${CV_DET_WEIGHT_DIR}/darknet53-a628ea1b.pth" ;;
        fcos) echo "${CV_DET_WEIGHT_DIR}/resnet50_msra-5891d200.pth" ;;
        ssd300) echo "${CV_DET_WEIGHT_DIR}/vgg16_caffe-292e1171.pth" ;;
        centernet) echo "${CV_DET_WEIGHT_DIR}/resnet18-f37072fd.pth" ;;
        swin_mask_rcnn) echo "${CV_DET_WEIGHT_DIR}/swin_tiny_patch4_window7_224.pth" ;;
        *) echo "Unsupported model: $1" >&2; return 1 ;;
    esac
}

IFS=',' read -r -a MODELS <<< "$CV_DET_MODELS"
IFS=',' read -r -a PRECISIONS <<< "$CV_DET_PRECISIONS"
TOTAL=${#MODELS[@]}
COUNT=0

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_NAME="$(echo "$MODEL_NAME" | xargs)"
    CONFIG="$(model_config "$MODEL_NAME")"
    WEIGHT="$(model_weight "$MODEL_NAME")"
    COUNT=$((COUNT + 1))

    test -f "$CONFIG"
    test -f "$WEIGHT"

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
        echo "  [${COUNT}/${TOTAL}] ${MODEL_NAME} - ${PRECISION} - ${CV_DET_NGPU} GPUs"
        echo "============================================================"

        "$PYTHON" -m torch.distributed.launch \
            --nnodes="${NODE_COUNT:-1}" \
            --node-rank="${NODE_RANK:-0}" \
            --master-addr="${MASTER_ADDR:-127.0.0.1}" \
            --nproc_per_node="${CV_DET_NGPU}" \
            --master-port="${MASTER_PORT:-29500}" \
            tools/train.py "$CONFIG" \
            --launcher pytorch \
            --work-dir "${CV_DET_LOG_DIR}/${MODEL_NAME}_gpus${CV_DET_NGPU}_${PRECISION}" \
            --cfg-options \
                model.backbone.init_cfg.type=Pretrained \
                model.backbone.init_cfg.checkpoint="${WEIGHT}" \
                "$AMP_OPT"

        echo "[${COUNT}/${TOTAL}] ${MODEL_NAME} ${PRECISION} done."
    done
done

echo ""
echo "========== All detection tests finished =========="

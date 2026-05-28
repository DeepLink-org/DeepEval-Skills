#!/usr/bin/env bash
set -euo pipefail

# Detection benchmark. GPU count and model can be controlled by env. Precision is fixed to fp16,fp32.

CV_DET_PROJECT_ROOT="${CV_DET_PROJECT_ROOT:-/workspace/code}"
CV_DET_MMDET_DIR="${CV_DET_MMDET_DIR:-}"
CV_DET_MMCV_DIR="${CV_DET_MMCV_DIR:-}"
CV_DET_LOGS_DIR="${CV_DET_LOGS_DIR:-${CV_DET_LOG_DIR:-/workspace/logs}}"
CV_DET_WEIGHT_DIR="${CV_DET_WEIGHT_DIR:-/workspace/weight}"
CV_DET_DATA_DIR="${CV_DET_DATA_DIR:-${CV_DET_DATASET_DIR:-/workspace/datasets/coco}}"
CV_DET_NGPU="${CV_DET_NGPU:-${CARD_COUNT:-1}}"
CV_DET_MODELS="${CV_DET_MODELS:-faster_rcnn}"
CV_DET_PRECISIONS="fp16,fp32"
PYTHON="${PYTHON:-/usr/bin/python3}"

if [ -z "$CV_DET_MMDET_DIR" ]; then
    for candidate in \
        "${CV_DET_PROJECT_ROOT}/onedl-mmdetection" \
        "/workspace/code/onedl-mmdetection" \
        "/workspace/onedl-mmdetection" \
        "/opt/onedl-mmdetection" \
        "/workspace/mmdetection" \
        "/opt/mmdetection"; do
        if [ -f "$candidate/tools/train.py" ] && [ -d "$candidate/configs" ]; then
            CV_DET_MMDET_DIR="$candidate"
            break
        fi
    done
fi

if [ -z "$CV_DET_MMDET_DIR" ]; then
    echo "ERROR: mmdetection source tree not found. Set CV_DET_MMDET_DIR to a directory containing tools/train.py and configs/." >&2
    exit 1
fi

if [ -z "$CV_DET_MMCV_DIR" ] && [ -d "${CV_DET_PROJECT_ROOT}/onedl-mmcv" ]; then
    CV_DET_MMCV_DIR="${CV_DET_PROJECT_ROOT}/onedl-mmcv"
fi

cd "$CV_DET_MMDET_DIR"

export MMDET_PATH="$CV_DET_MMDET_DIR"
export CV_DET_LOGS_DIR CV_DET_WEIGHT_DIR CV_DET_DATA_DIR CV_DET_NGPU CV_DET_MODELS CV_DET_PRECISIONS
export CV_DET_LOG_DIR="$CV_DET_LOGS_DIR"
export CV_DET_DATASET_DIR="$CV_DET_DATA_DIR"
export SYSTEM_PACKAGES="${SYSTEM_PACKAGES:-/usr/local/lib/python3.10/dist-packages}"
if [ -n "$CV_DET_MMCV_DIR" ]; then
    export MMCV_PATH="$CV_DET_MMCV_DIR"
    export PYTHONPATH="$MMDET_PATH:$MMCV_PATH:$SYSTEM_PACKAGES:${PYTHONPATH:-}"
else
    export PYTHONPATH="$MMDET_PATH:$SYSTEM_PACKAGES:${PYTHONPATH:-}"
fi
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

mkdir -p "$CV_DET_LOGS_DIR"
if [ -d "$CV_DET_DATA_DIR" ]; then
    mkdir -p data
    ln -sfn "$CV_DET_DATA_DIR" data/coco
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
            --work-dir "${CV_DET_LOGS_DIR}/${MODEL_NAME}_gpus${CV_DET_NGPU}_${PRECISION}" \
            --cfg-options \
                model.backbone.init_cfg.type=Pretrained \
                model.backbone.init_cfg.checkpoint="${WEIGHT}" \
                "$AMP_OPT"

        echo "[${COUNT}/${TOTAL}] ${MODEL_NAME} ${PRECISION} done."
    done
done

echo ""
echo "========== All detection tests finished =========="

"$PYTHON" - <<'CV_DET_PARSE'
import glob
import json
import os
import re

models = [m.strip() for m in os.environ.get("CV_DET_MODELS", "faster_rcnn").split(",") if m.strip()]
precisions = [p.strip() for p in os.environ.get("CV_DET_PRECISIONS", "fp16,fp32").split(",") if p.strip()]
gpu = os.environ.get("CV_DET_NGPU", "1")
log_dir = os.environ.get("CV_DET_LOGS_DIR", "/workspace/logs")
marker = os.environ.get("CV_DET_RUN_MARKER")
marker_mtime = os.path.getmtime(marker) if marker and os.path.exists(marker) else 0

results = {}
for model in models:
    for precision in precisions:
        key = f"{model}_gpus{gpu}_{precision}"
        work_dir = os.path.join(log_dir, key)
        logs = [p for p in glob.glob(os.path.join(work_dir, "*", "*.log")) if os.path.getmtime(p) > marker_mtime]
        logs.sort(key=os.path.getmtime, reverse=True)
        log = logs[0] if logs else ""
        text = open(log, "r", encoding="utf-8", errors="ignore").read() if log else ""
        rows = re.findall(r"AVG_ITER_TIME:\s*([0-9.]+)s\s*\|\s*DATA:\s*([0-9.]+)s\s*\|\s*OP:\s*([0-9.]+)s", text)
        last = rows[-1] if rows else None
        item = {
            "model": model,
            "gpu_count": int(gpu),
            "precision": precision,
            "log": log,
            "avg_iter_time": float(last[0]) if last else None,
            "data_time": float(last[1]) if last else None,
            "op_time": float(last[2]) if last else None,
        }
        results[key] = item

aggregate = {
    "status": "success" if results and all(v.get("avg_iter_time") is not None for v in results.values()) else "partial",
    "task": "cv_detection",
    "model": ",".join(models),
    "gpu_count": int(gpu),
    "precisions": precisions,
    "results": results,
}
with open(os.path.join(log_dir, "eval_result.json"), "w", encoding="utf-8") as f:
    json.dump(aggregate, f, ensure_ascii=False, indent=2)
print(f"eval result json written: {os.path.join(log_dir, 'eval_result.json')}")
CV_DET_PARSE

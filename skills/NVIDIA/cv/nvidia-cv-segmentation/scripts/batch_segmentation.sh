#!/usr/bin/env bash
set -euo pipefail

# Segmentation benchmark. GPU count and model can be controlled by env. Precision is fixed to fp16,fp32.

CV_SEG_PROJECT_ROOT="${CV_SEG_PROJECT_ROOT:-/workspace/code}"
CV_SEG_MMSEG_DIR="${CV_SEG_MMSEG_DIR:-}"
CV_SEG_MMCV_DIR="${CV_SEG_MMCV_DIR:-}"
CV_SEG_LOGS_DIR="${CV_SEG_LOGS_DIR:-${CV_SEG_LOG_DIR:-/workspace/logs}}"
CV_SEG_WEIGHT_DIR="${CV_SEG_WEIGHT_DIR:-/workspace/weight}"
CV_SEG_WEIGHT_PATH="${CV_SEG_WEIGHT_PATH:-${CV_SEG_WEIGHT_DIR}/resnet50_v1c-2cccc1ad.pth}"
CV_SEG_DATA_DIR="${CV_SEG_DATA_DIR:-${CV_SEG_DATASET_DIR:-/workspace/datasets/cityscapes}}"
CV_SEG_NGPU="${CV_SEG_NGPU:-${CARD_COUNT:-1}}"
CV_SEG_MODELS="${CV_SEG_MODELS:-fcn}"
CV_SEG_PRECISIONS="fp16,fp32"
PYTHON="${PYTHON:-/usr/bin/python3}"

if [ -z "$CV_SEG_MMSEG_DIR" ]; then
    for candidate in         "${CV_SEG_PROJECT_ROOT}/onedl-mmsegmentation"         "/workspace/code/onedl-mmsegmentation"         "/workspace/onedl-mmsegmentation"         "/opt/onedl-mmsegmentation"         "/workspace/mmsegmentation"         "/opt/mmsegmentation"; do
        if [ -f "$candidate/tools/train.py" ] && [ -d "$candidate/configs" ]; then
            CV_SEG_MMSEG_DIR="$candidate"
            break
        fi
    done
fi

if [ -z "$CV_SEG_MMSEG_DIR" ]; then
    echo "ERROR: mmsegmentation source tree not found. Set CV_SEG_MMSEG_DIR to a directory containing tools/train.py and configs/." >&2
    exit 1
fi

if [ -z "$CV_SEG_MMCV_DIR" ] && [ -d "${CV_SEG_PROJECT_ROOT}/onedl-mmcv" ]; then
    CV_SEG_MMCV_DIR="${CV_SEG_PROJECT_ROOT}/onedl-mmcv"
fi

cd "$CV_SEG_MMSEG_DIR"

export MMSEG_PATH="$CV_SEG_MMSEG_DIR"
export CV_SEG_LOGS_DIR CV_SEG_WEIGHT_DIR CV_SEG_WEIGHT_PATH CV_SEG_DATA_DIR CV_SEG_NGPU CV_SEG_MODELS CV_SEG_PRECISIONS
export CV_SEG_LOG_DIR="$CV_SEG_LOGS_DIR"
export CV_SEG_DATASET_DIR="$CV_SEG_DATA_DIR"
export SYSTEM_PACKAGES="${SYSTEM_PACKAGES:-/usr/local/lib/python3.10/dist-packages}"
if [ -n "$CV_SEG_MMCV_DIR" ]; then
    export MMCV_PATH="$CV_SEG_MMCV_DIR"
    export PYTHONPATH="$MMSEG_PATH:$MMCV_PATH:$SYSTEM_PACKAGES:${PYTHONPATH:-}"
else
    export PYTHONPATH="$MMSEG_PATH:$SYSTEM_PACKAGES:${PYTHONPATH:-}"
fi
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

mkdir -p "$CV_SEG_LOGS_DIR"

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

    test -f "$CONFIG"
    test -f "$CV_SEG_WEIGHT_PATH"

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

        "$PYTHON" -m torch.distributed.launch             --nnodes="${NODE_COUNT:-1}"             --node-rank="${NODE_RANK:-0}"             --master-addr="${MASTER_ADDR:-127.0.0.1}"             --nproc_per_node="${CV_SEG_NGPU}"             --master-port="${MASTER_PORT:-29500}"             tools/train.py "$CONFIG"             --launcher pytorch             --work-dir "${CV_SEG_LOGS_DIR}/${MODEL_NAME}_gpus${CV_SEG_NGPU}_${PRECISION}"             --cfg-options                 model.backbone.init_cfg.type=Pretrained                 model.backbone.init_cfg.checkpoint="${CV_SEG_WEIGHT_PATH}"                 model.pretrained=None                 "$AMP_OPT"

        echo "[${COUNT}/${TOTAL}] ${MODEL_NAME} ${PRECISION} done."
    done
done

echo ""
echo "========== All segmentation tests finished =========="

"$PYTHON" - <<'CV_SEG_PARSE'
import glob
import json
import os
import re

models = [m.strip() for m in os.environ.get("CV_SEG_MODELS", "fcn").split(",") if m.strip()]
precisions = [p.strip() for p in os.environ.get("CV_SEG_PRECISIONS", "fp16,fp32").split(",") if p.strip()]
gpu = os.environ.get("CV_SEG_NGPU", "1")
log_dir = os.environ.get("CV_SEG_LOGS_DIR", "/workspace/logs")
marker = os.environ.get("CV_SEG_RUN_MARKER")
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
    "task": "cv_segmentation",
    "model": ",".join(models),
    "gpu_count": int(gpu),
    "precisions": precisions,
    "results": results,
}
with open(os.path.join(log_dir, "eval_result.json"), "w", encoding="utf-8") as f:
    json.dump(aggregate, f, ensure_ascii=False, indent=2)
print(f"eval result json written: {os.path.join(log_dir, 'eval_result.json')}")
CV_SEG_PARSE

#!/usr/bin/env bash
set -euo pipefail

# Pretrain/classification benchmark. GPU count and model can be controlled by env. Precision is fixed to fp16,fp32.

CV_PRE_PROJECT_ROOT="${CV_PRE_PROJECT_ROOT:-/workspace/code}"
CV_PRE_MMPRE_DIR="${CV_PRE_MMPRE_DIR:-}"
CV_PRE_MMCV_DIR="${CV_PRE_MMCV_DIR:-}"
CV_PRE_LOGS_DIR="${CV_PRE_LOGS_DIR:-${CV_PRE_LOG_DIR:-/workspace/logs}}"
CV_PRE_DATA_DIR="${CV_PRE_DATA_DIR:-${CV_PRE_DATASET_DIR:-/workspace/datasets/imagenet}}"
CV_PRE_NGPU="${CV_PRE_NGPU:-${CARD_COUNT:-1}}"
CV_PRE_MODELS="${CV_PRE_MODELS:-resnet50}"
CV_PRE_PRECISIONS="fp16,fp32"
PYTHON="${PYTHON:-/usr/bin/python3}"

if [ -z "$CV_PRE_MMPRE_DIR" ]; then
    for candidate in         "${CV_PRE_PROJECT_ROOT}/onedl-mmpretrain"         "/workspace/code/onedl-mmpretrain"         "/workspace/onedl-mmpretrain"         "/opt/onedl-mmpretrain"         "/workspace/mmpretrain"         "/opt/mmpretrain"; do
        if [ -f "$candidate/tools/train.py" ] && [ -d "$candidate/configs" ]; then
            CV_PRE_MMPRE_DIR="$candidate"
            break
        fi
    done
fi

if [ -z "$CV_PRE_MMPRE_DIR" ]; then
    echo "ERROR: mmpretrain source tree not found. Set CV_PRE_MMPRE_DIR to a directory containing tools/train.py and configs/." >&2
    exit 1
fi

if [ -z "$CV_PRE_MMCV_DIR" ] && [ -d "${CV_PRE_PROJECT_ROOT}/onedl-mmcv" ]; then
    CV_PRE_MMCV_DIR="${CV_PRE_PROJECT_ROOT}/onedl-mmcv"
fi

cd "$CV_PRE_MMPRE_DIR"

export MMPRE_PATH="$CV_PRE_MMPRE_DIR"
export CV_PRE_LOGS_DIR CV_PRE_DATA_DIR CV_PRE_NGPU CV_PRE_MODELS CV_PRE_PRECISIONS
export CV_PRE_LOG_DIR="$CV_PRE_LOGS_DIR"
export CV_PRE_DATASET_DIR="$CV_PRE_DATA_DIR"
export SYSTEM_PACKAGES="${SYSTEM_PACKAGES:-/usr/local/lib/python3.10/dist-packages}"
if [ -n "$CV_PRE_MMCV_DIR" ]; then
    export MMCV_PATH="$CV_PRE_MMCV_DIR"
    export PYTHONPATH="$MMPRE_PATH:$MMCV_PATH:$SYSTEM_PACKAGES:${PYTHONPATH:-}"
else
    export PYTHONPATH="$MMPRE_PATH:$SYSTEM_PACKAGES:${PYTHONPATH:-}"
fi
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

mkdir -p "$CV_PRE_LOGS_DIR"
if [ -d "$CV_PRE_DATA_DIR" ]; then
    mkdir -p data
    ln -sfn "$CV_PRE_DATA_DIR" data/imagenet
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

        "$PYTHON" -m torch.distributed.launch             --nnodes="${NODE_COUNT:-1}"             --node-rank="${NODE_RANK:-0}"             --master-addr="${MASTER_ADDR:-127.0.0.1}"             --nproc_per_node="${CV_PRE_NGPU}"             --master-port="${MASTER_PORT:-29500}"             tools/train.py "$CONFIG"             --launcher pytorch             --work-dir "${CV_PRE_LOGS_DIR}/${MODEL_NAME}_gpus${CV_PRE_NGPU}_${PRECISION}"             --cfg-options                 "$AMP_OPT"

        echo "[${COUNT}/${TOTAL}] ${MODEL_NAME} ${PRECISION} done."
    done
done

echo ""
echo "========== All pretrain tests finished =========="

"$PYTHON" - <<'CV_PRE_PARSE'
import glob
import json
import os
import re

models = [m.strip() for m in os.environ.get("CV_PRE_MODELS", "resnet50").split(",") if m.strip()]
precisions = [p.strip() for p in os.environ.get("CV_PRE_PRECISIONS", "fp16,fp32").split(",") if p.strip()]
gpu = os.environ.get("CV_PRE_NGPU", "1")
log_dir = os.environ.get("CV_PRE_LOGS_DIR", "/workspace/logs")
marker = os.environ.get("CV_PRE_RUN_MARKER")
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
    "task": "cv_pretrain",
    "model": ",".join(models),
    "gpu_count": int(gpu),
    "precisions": precisions,
    "results": results,
}
with open(os.path.join(log_dir, "eval_result.json"), "w", encoding="utf-8") as f:
    json.dump(aggregate, f, ensure_ascii=False, indent=2)
print(f"eval result json written: {os.path.join(log_dir, 'eval_result.json')}")
CV_PRE_PARSE

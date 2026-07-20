#!/usr/bin/env bash
set -euo pipefail

# alpaca-lora LoRA finetune benchmark, single entry point.
#
# This script merges the old finetune.sh (training launcher) and the wrapper/parser
# into one file, so the skill no longer depends on the project's finetune.sh.
# Defaults are container-side mount targets (match the docker run -v in SKILL.md);
# all overridable via env.

NLP_FIN_PROJECT_ROOT="${NLP_FIN_PROJECT_ROOT:-/workspace/code/alpaca_finetune}"
NLP_FIN_ALPACA_DIR="${NLP_FIN_ALPACA_DIR:-${NLP_FIN_PROJECT_ROOT}/alpaca-lora}"
NLP_FIN_LOGS_DIR="${NLP_FIN_LOGS_DIR:-${LOGS_DIR:-/workspace/logs}}"
NLP_FIN_RESULTS_DIR="${NLP_FIN_RESULTS_DIR:-${RESULTS_DIR:-/workspace/results}}"
NLP_FIN_NGPU="${NLP_FIN_NGPU:-${CARD_COUNT:-8}}"
NLP_FIN_BATCH_SIZE="${NLP_FIN_BATCH_SIZE:-128}"
NLP_FIN_MICRO_BATCH_SIZE="${NLP_FIN_MICRO_BATCH_SIZE:-4}"
NLP_FIN_LOG_FILE="${NLP_FIN_LOG_FILE:-${NLP_FIN_LOGS_DIR}/finetune_${NLP_FIN_BATCH_SIZE}_${NLP_FIN_MICRO_BATCH_SIZE}_closeint8.log}"
NLP_FIN_RESULT_FILE="${NLP_FIN_RESULT_FILE:-${NLP_FIN_RESULTS_DIR}/result.json}"
PYTHON="${PYTHON:-python3}"

# Container-side mount targets (SKILL.md docker run -v targets).
MODEL_DIR="${MODEL_DIR:-/data/models/llama-7b-hf}"
DATASET_DIR="${DATASET_DIR:-/data/datasets/alpaca-cleaned}"
OUTPUT_DIR="${OUTPUT_DIR:-${NLP_FIN_ALPACA_DIR}/lora-adapter}"

if ! [[ "$NLP_FIN_NGPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NLP_FIN_NGPU must be a positive integer, got: $NLP_FIN_NGPU" >&2
    exit 1
fi

if [ ! -d "$NLP_FIN_ALPACA_DIR" ]; then
    echo "ERROR: Alpaca-LoRA directory not found: $NLP_FIN_ALPACA_DIR" >&2
    exit 1
fi

if [ ! -f "${NLP_FIN_ALPACA_DIR}/finetune_int8_close.py" ]; then
    echo "ERROR: finetune_int8_close.py not found in ${NLP_FIN_ALPACA_DIR}" >&2
    exit 1
fi

mkdir -p "$NLP_FIN_LOGS_DIR" "$NLP_FIN_RESULTS_DIR" "$OUTPUT_DIR"

NLP_FIN_RUN_MARKER="${NLP_FIN_RUN_MARKER:-/tmp/nlp_fin_run_marker.$$.${RANDOM}}"
touch "$NLP_FIN_RUN_MARKER"
export NLP_FIN_LOGS_DIR NLP_FIN_RESULTS_DIR NLP_FIN_NGPU
export NLP_FIN_BATCH_SIZE NLP_FIN_MICRO_BATCH_SIZE NLP_FIN_LOG_FILE
export NLP_FIN_RESULT_FILE NLP_FIN_RUN_MARKER
export LOGS_DIR="$NLP_FIN_LOGS_DIR"
export RESULTS_DIR="$NLP_FIN_RESULTS_DIR"
export BATCH_SIZE="$NLP_FIN_BATCH_SIZE"
export MICRO_BATCH_SIZE="$NLP_FIN_MICRO_BATCH_SIZE"

# 用项目自带的 trainer.py / trainer_utils.py 覆盖容器已安装的 transformers 对应文件。
# 自定义版本带 train_tokens_per_second 等性能采集（见 code/README.md），不覆盖的话
# 汇总行没有这些字段，下面的解析会报 "No final training summary containing
# train_tokens_per_second found"。
#
# 版本兼容：自定义 trainer_utils.py 基于 transformers 4.32.0，镜像里是新版（>=4.35）
# 新版 utils 里删除了 is_torch_tpu_available（TPU 支持废弃）。覆盖后 import 会崩：
#   ImportError: cannot import name 'is_torch_tpu_available' from 'transformers.utils'
# TPU 在 NVIDIA 镜像里走不到，sed 删掉 import 行 + 把 2 处 call site 改成 `if False:`。
TRANSFORMERS_DIR=$("${PYTHON}" -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
for fname in trainer.py trainer_utils.py; do
    src="${NLP_FIN_PROJECT_ROOT}/${fname}"
    if [ -f "$src" ]; then
        cp "$src" "${TRANSFORMERS_DIR}/${fname}"
        echo "Patched transformers/${fname} from ${src}"
    else
        echo "WARNING: ${src} not found, using installed transformers/${fname} (no perf instrumentation)" >&2
    fi
done
# trainer_utils.py 兼容新版 transformers：去掉 is_torch_tpu_available
sed -i '/is_torch_tpu_available,/d; s/if is_torch_tpu_available(check_device=True):/if False:/g' \
    "${TRANSFORMERS_DIR}/trainer_utils.py"
echo "Stripped is_torch_tpu_available from ${TRANSFORMERS_DIR}/trainer_utils.py"

cd "$NLP_FIN_ALPACA_DIR"
echo "Running finetune_int8_close.py with ${NLP_FIN_NGPU} GPUs"

# Clear stale log so the -nt check below is a reliable "this run wrote it" signal.
rm -f "$NLP_FIN_LOG_FILE"

"${PYTHON}" -m torch.distributed.run --nproc_per_node="$NLP_FIN_NGPU" finetune_int8_close.py \
    --base_model "${MODEL_DIR}" \
    --data_path "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "$NLP_FIN_BATCH_SIZE" \
    --micro_batch_size "$NLP_FIN_MICRO_BATCH_SIZE" \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length > "${NLP_FIN_LOG_FILE}" 2>&1

if [ ! -f "$NLP_FIN_LOG_FILE" ]; then
    echo "ERROR: expected training log was not generated: $NLP_FIN_LOG_FILE" >&2
    exit 1
fi

if [ ! "$NLP_FIN_LOG_FILE" -nt "$NLP_FIN_RUN_MARKER" ]; then
    echo "ERROR: training log was not updated by this run: $NLP_FIN_LOG_FILE" >&2
    exit 1
fi

"$PYTHON" - <<'NLP_FIN_PARSE'
import ast
import json
import os
import re

log_path = os.environ["NLP_FIN_LOG_FILE"]
result_path = os.environ["NLP_FIN_RESULT_FILE"]
nproc = int(os.environ["NLP_FIN_NGPU"])

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

summaries = []
for line in text.splitlines():
    if "train_tokens_per_second" not in line:
        continue
    for candidate in re.findall(r"\{[^{}]+\}", line):
        try:
            value = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict) and "train_tokens_per_second" in value:
            summaries.append(value)

if not summaries:
    raise SystemExit(
        f"No final training summary containing train_tokens_per_second found in {log_path}"
    )

summary = summaries[-1]

def rounded(name, digits):
    value = summary.get(name)
    return round(float(value), digits) if value is not None else None

total_tps = rounded("train_tokens_per_second", 2)
metrics = {
    "train_tokens_per_second": total_tps,
    "tokens_per_sec_per_gpu": round(total_tps / nproc, 2),
    "train_samples_per_second": rounded("train_samples_per_second", 2),
    "train_steps_per_second": rounded("train_steps_per_second", 4),
    "train_runtime": rounded("train_runtime", 2),
    "train_loss": rounded("train_loss", 4),
}
result = {
    "status": "success",
    "task": "nlp_finetune",
    "gpu_count": nproc,
    "log": log_path,
    "metrics": metrics,
}

os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"train_tokens_per_second (total): {total_tps:.2f}")
print(f"tokens_per_sec_per_gpu ({nproc} GPUs): {metrics['tokens_per_sec_per_gpu']:.2f}")
print(f"result.json: {json.dumps(result, ensure_ascii=False)}")
NLP_FIN_PARSE

echo "Fine-tuning result written to: $NLP_FIN_RESULT_FILE"
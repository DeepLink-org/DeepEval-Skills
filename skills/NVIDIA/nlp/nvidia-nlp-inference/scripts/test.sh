#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-2048}"
NUM_PROMPTS="${NUM_PROMPTS:-2000}"

TRANSFORMERS_OFFLINE=1 \
python3 -m sglang.bench_serving \
  --model /data/models/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 \
  --random-range-ratio 1 \
  --backend sglang \
  --dataset-name random \
  --dataset-path /data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --output-file /workspace/logs/bench.csv \
  --seed 42 2>&1 | tee /workspace/logs/bench.log

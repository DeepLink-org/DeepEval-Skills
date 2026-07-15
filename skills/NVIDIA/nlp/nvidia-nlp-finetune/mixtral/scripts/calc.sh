#!/usr/bin/env bash
set -euo pipefail

FILENAME="${1:?Usage: collect_metrics.sh TRAIN_LOG [START_ITER] [END_ITER]}"
START_ITER="${2:-5}"
END_ITER="${3:-22}"
CODE_ROOT="${MIXTRAL_CODE_ROOT:-/workspace/code}"
LOG_ROOT="${MIXTRAL_LOG_ROOT:-/workspace/logs}"
RESULT_ROOT="${MIXTRAL_RESULT_ROOT:-/workspace/results}"
CALC_LOG="${MIXTRAL_CALC_LOG:-$LOG_ROOT/calc.log}"

test -f "$CODE_ROOT/calc.py"

# 第一个参数必须是 finetune.sh 本次生成的 rank0 训练日志，而不是 launcher 日志。
if [ ! -s "$FILENAME" ]; then
    echo "Training log does not exist or is empty: $FILENAME" >&2
    exit 1
fi
case "$FILENAME" in
    "$LOG_ROOT"/train_Full_*_node0_*.log) ;;
    *)
        echo "Unexpected rank0 training log path: $FILENAME" >&2
        exit 1
        ;;
esac

mkdir -p "$RESULT_ROOT" "$(dirname "$CALC_LOG")"
rm -f "$RESULT_ROOT/result.json"
RAW_OUTPUT="$(python "$CODE_ROOT/calc.py" "$FILENAME" "$START_ITER" "$END_ITER" | tee "$CALC_LOG")"

# calc.py 的合法输出必须仅为一个有限数值，该数值表示吞吐量。
python - "$RAW_OUTPUT" "$RESULT_ROOT/result.json" <<'PY'
import json
import math
import sys

raw = sys.argv[1].strip()
result_path = sys.argv[2]

try:
    throughput = float(raw)
except ValueError as exc:
    raise SystemExit(f"calc.py output is not a single number: {raw!r}") from exc

if not math.isfinite(throughput):
    raise SystemExit(f"calc.py output is not finite: {raw!r}")

result = {"status": "success", "metrics": {"throughput": throughput}}
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("result.json:", json.dumps(result, ensure_ascii=False))
PY

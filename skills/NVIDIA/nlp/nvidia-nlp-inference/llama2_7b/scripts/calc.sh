#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="${1:-/workspace/logs/bench.log}"
TP="${2:-${TP:-1}}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/results}"

test -s "$LOG_PATH"
case "$TP" in ''|*[!0-9]*) echo "TP must be a positive integer" >&2; exit 1 ;; esac
if [ "$TP" -lt 1 ]; then
  echo "TP must be a positive integer" >&2
  exit 1
fi
mkdir -p "$RESULT_ROOT"

python3 - "$LOG_PATH" "$TP" "$RESULT_ROOT/result.json" <<'PY'
import json
import math
import re
import sys

log_path, tp_text, result_path = sys.argv[1:]
with open(log_path, encoding="utf-8", errors="replace") as f:
    text = f.read()

patterns = {
    "output_token_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "total_token_throughput": r"Total token throughput \(tok/s\):\s+([\d.]+)",
    "concurrency": r"Concurrency:\s+([\d.]+)",
    "mean_e2e_latency_ms": r"Mean E2E Latency \(ms\):\s+([\d.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([\d.]+)",
}

metrics = {}
for name, pattern in patterns.items():
    matches = re.findall(pattern, text)
    if not matches:
        raise SystemExit(f"missing benchmark metric: {name}")
    value = float(matches[-1])
    if not math.isfinite(value):
        raise SystemExit(f"non-finite benchmark metric: {name}={value!r}")
    metrics[name] = round(value, 2)

tp = int(tp_text)
metrics["output_tokens_per_sec_per_gpu"] = round(metrics["output_token_throughput"] / tp, 2)
result = {"status": "success", "metrics": metrics}
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("result.json:", json.dumps(result, ensure_ascii=False))
PY

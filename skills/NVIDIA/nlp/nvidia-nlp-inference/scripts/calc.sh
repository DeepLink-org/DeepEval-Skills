#!/usr/bin/env bash
set -euo pipefail
LOG_PATH="${1:-/workspace/logs/bench.log}"
TP="${2:-${TP:-1}}"
# SGLang 0.5.6 writes a single JSON summary despite this legacy filename.
SUMMARY_PATH="${3:-$(dirname "$LOG_PATH")/bench.csv}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/results}"
SCHEMA_VERSION="${SCHEMA_VERSION:-1.2}"
TASK_ID="${TASK_ID:-NVIDIA_nlp_inference}"
WORKLOAD_FINGERPRINT="${WORKLOAD_FINGERPRINT:?WORKLOAD_FINGERPRINT is required}"

test -s "$LOG_PATH"
test -s "$SUMMARY_PATH"
case "$TP" in ''|*[!0-9]*) echo 'TP must be a positive integer' >&2; exit 1;; esac
(( TP >= 1 )) || { echo 'TP must be at least 1' >&2; exit 1; }
[[ "$SCHEMA_VERSION" == "1.2" ]] || { echo 'SCHEMA_VERSION must be 1.2' >&2; exit 1; }
mkdir -p "$RESULT_ROOT"

python3 - "$LOG_PATH" "$TP" "$SUMMARY_PATH" "$RESULT_ROOT/result.json.tmp" \
  "$RESULT_ROOT/result.json" "$SCHEMA_VERSION" "$TASK_ID" "$WORKLOAD_FINGERPRINT" <<'PY'
import json
import math
import os
import re
import sys

log_path, tp_text, summary_path, tmp_path, result_path, schema, task_id, fingerprint = sys.argv[1:]
text = open(log_path, encoding="utf-8", errors="replace").read()
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
    found = re.findall(pattern, text)
    if not found:
        raise SystemExit(f"missing benchmark metric: {name}")
    value = float(found[-1])
    if not math.isfinite(value):
        raise SystemExit(f"non-finite benchmark metric: {name}={value!r}")
    metrics[name] = round(value, 2)

tp = int(tp_text)
metrics["output_tokens_per_sec_per_gpu"] = round(metrics["output_token_throughput"] / tp, 2)

try:
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    measurement_count = int(summary["completed"])
    duration_seconds = float(summary["duration"])
except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid SGLang benchmark summary {summary_path}: {exc}") from exc

if measurement_count <= 0:
    raise SystemExit(f"invalid completed count: {measurement_count}")
if not math.isfinite(duration_seconds) or duration_seconds <= 0:
    raise SystemExit(f"invalid duration: {duration_seconds!r}")

result = {
    "schema_version": schema,
    "task_id": task_id,
    "status": "success",
    "metrics": metrics,
    "metadata": {
        "workload_fingerprint": fingerprint,
        "measurement_count": measurement_count,
        "duration_seconds": duration_seconds,
        "source": f"{log_path}, {summary_path}",
    },
}

with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
os.replace(tmp_path, result_path)
print("result.json:", json.dumps(result, ensure_ascii=False, separators=(",", ":")))
PY

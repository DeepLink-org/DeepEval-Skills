#!/usr/bin/env bash
set -euo pipefail
LOG_PATH="${1:-/workspace/logs/bench.log}"; TP="${2:-${TP:-1}}"; RESULT_ROOT="${RESULT_ROOT:-/workspace/results}"
# The task-level NVIDIA/nlp/inference Skill has a stable Agent task id. Callers
# may override it for an explicitly renamed task, but the default keeps the
# standalone Skill workflow compatible with the Agent result contract.
SCHEMA_VERSION="${SCHEMA_VERSION:-1.0}"; TASK_ID="${TASK_ID:-NVIDIA_nlp_inference}"
test -s "$LOG_PATH"; case "$TP" in ''|*[!0-9]*) echo 'TP must be a positive integer' >&2; exit 1;; esac; (( TP >= 1 )) || exit 1; mkdir -p "$RESULT_ROOT"
python3 - "$LOG_PATH" "$TP" "$RESULT_ROOT/result.json" "$SCHEMA_VERSION" "$TASK_ID" <<'PY'
import json, math, re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
patterns = {"output_token_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)", "total_token_throughput": r"Total token throughput \(tok/s\):\s+([\d.]+)", "concurrency": r"Concurrency:\s+([\d.]+)", "mean_e2e_latency_ms": r"Mean E2E Latency \(ms\):\s+([\d.]+)", "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)", "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)", "mean_itl_ms": r"Mean ITL \(ms\):\s+([\d.]+)"}
metrics = {}
for name, pattern in patterns.items():
    found = re.findall(pattern, text)
    if not found: raise SystemExit(f"missing benchmark metric: {name}")
    value = float(found[-1])
    if not math.isfinite(value): raise SystemExit(f"non-finite benchmark metric: {name}={value!r}")
    metrics[name] = round(value, 2)
metrics["output_tokens_per_sec_per_gpu"] = round(metrics["output_token_throughput"] / int(sys.argv[2]), 2)
result = {
    "schema_version": sys.argv[4],
    "task_id": sys.argv[5],
    "status": "success",
    "metrics": metrics,
}
with open(sys.argv[3], "w", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)
print("result.json:", json.dumps(result, ensure_ascii=False))
PY

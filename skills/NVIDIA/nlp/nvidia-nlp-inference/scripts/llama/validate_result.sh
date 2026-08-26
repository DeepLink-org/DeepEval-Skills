#!/usr/bin/env bash
# Validate the Agent result contract without asking a generated task script to
# embed fragile Python/heredoc quoting.  Identity values are supplied by the
# current Generator result contract through the caller's environment.
set -euo pipefail

RESULT_PATH="${1:-${RESULT_ROOT:-/workspace/results}/result.json}"
: "${TASK_ID:?TASK_ID is required}"
: "${WORKLOAD_FINGERPRINT:?WORKLOAD_FINGERPRINT is required}"
: "${SCHEMA_VERSION:?SCHEMA_VERSION is required}"

python3 - "$RESULT_PATH" "$SCHEMA_VERSION" "$TASK_ID" "$WORKLOAD_FINGERPRINT" <<'PY'
import json
import math
import sys

path, schema, task_id, fingerprint = sys.argv[1:]
try:
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid result.json {path}: {exc}") from exc

if not isinstance(result, dict):
    raise SystemExit("result.json must be an object")
if result.get("schema_version") != schema:
    raise SystemExit("result.json schema_version does not match SCHEMA_VERSION")
if result.get("task_id") != task_id:
    raise SystemExit("result.json task_id does not match TASK_ID")
if result.get("status") not in {"success", "failed", "skipped"}:
    raise SystemExit("result.json has an invalid status")

metrics = result.get("metrics")
metadata = result.get("metadata")
if not isinstance(metrics, dict) or not isinstance(metadata, dict):
    raise SystemExit("result.json metrics and metadata must be objects")
if result["status"] == "success":
    if not metrics:
        raise SystemExit("successful result.json requires metrics")
    if metadata.get("workload_fingerprint") != fingerprint:
        raise SystemExit("result.json workload_fingerprint does not match WORKLOAD_FINGERPRINT")
    if not isinstance(metadata.get("measurement_count"), int) or metadata["measurement_count"] <= 0:
        raise SystemExit("result.json has an invalid measurement_count")
    duration = metadata.get("duration_seconds")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
        raise SystemExit("result.json has an invalid duration_seconds")
    if not isinstance(metadata.get("source"), str) or not metadata["source"]:
        raise SystemExit("result.json has an invalid source")
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise SystemExit(f"result.json metric is not finite: {name}")

print("result.json:", json.dumps(result, ensure_ascii=False, separators=(",", ":")))
PY

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
TARGET="${1:-all}"
require_choice TARGET "$TARGET" gemm gemm-conv longtail all
prepare_operator_dirs
python3 - "$TARGET" "$OPERATOR_RESULTS_DIR" "$OPERATOR_LOGS_DIR" <<'PY'
import csv
import json
import math
import re
import sys
from pathlib import Path

target, results_dir, logs_dir = sys.argv[1:]
results_dir, logs_dir = Path(results_dir), Path(logs_dir)
result = {"status": "success", "backend": "DTK-26.04/HIP-PyTorch-2.10", "artifacts": {}, "errors": []}
names = []
if target in {"gemm", "gemm-conv", "all"}:
    names += ["gemm_fp16", "gemm_fp32"]
if target in {"gemm-conv", "all"}:
    names += ["conv_fp16", "conv_fp32"]
if target in {"longtail", "all"}:
    names += ["longtail_fp16", "longtail_fp32"]
expected_rows = {"gemm_fp16": 224, "gemm_fp32": 224, "conv_fp16": 63, "conv_fp32": 63, "longtail_fp16": 40, "longtail_fp32": 40}
for name in names:
    path = results_dir / f"{name}.csv"
    if not path.is_file():
        result["errors"].append(name)
        continue
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    try:
        numeric_valid = all(math.isfinite(float(row["baseline"])) and float(row["baseline"]) > 0 for row in rows)
    except (KeyError, TypeError, ValueError):
        numeric_valid = False
    row_valid = len(rows) in {39, 40} if name.startswith("longtail_") else len(rows) == expected_rows[name]
    valid = row_valid and numeric_valid
    artifact = {"path": str(path), "rows": len(rows), "valid": valid}
    if rows and numeric_valid:
        artifact["mean_baseline"] = sum(float(row["baseline"]) for row in rows) / len(rows)
    result["artifacts"][name] = artifact
    if not valid:
        result["errors"].append(name)
if target == "all":
    path = logs_dir / "transformer_block.log"
    values = []
    if path.is_file():
        text = path.read_text(errors="replace")
        for name in ("encoder", "decoder"):
            match = re.search(rf"AIBENCH_TRANSFORMER_{name.upper()}_SECONDS=([0-9.eE+-]+)", text)
            if not match:
                match = re.search(rf"time\s+per\s+iteration\s+of\s+{name}\s*:\s*([0-9.eE+-]+)", text, re.IGNORECASE)
            if match:
                values.append(float(match.group(1)))
    valid = len(values) == 2 and all(math.isfinite(value) and value > 0 for value in values)
    result["artifacts"]["transformer_block"] = {"path": str(path), "seconds_per_iteration": values, "valid": valid}
    if not valid:
        result["errors"].append("transformer_block")
if result["errors"]:
    result["status"] = "failed"
(results_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if result["errors"]:
    raise SystemExit(1)
PY
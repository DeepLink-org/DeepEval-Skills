#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"

# Usage: collect_results.sh [gemm-conv|longtail|all]
TARGET="${1:-all}"
require_choice TARGET "$TARGET" gemm-conv longtail all
prepare_operator_dirs

python3 - "$TARGET" "$OPERATOR_PROJECT_ROOT" "$OPERATOR_RESULTS_DIR" "$OPERATOR_LOGS_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd

target, project_root, results_dir, logs_dir = sys.argv[1:]
result = {"status": "success", "metrics": {}, "artifacts": {}}

def load_csv(path):
    if not path.is_file():
        return
    df = pd.read_csv(path)
    numeric = {}
    for column in ("baseline", "time", "score"):
        if column in df:
            values = [float(v) for v in df[column].dropna() if math.isfinite(float(v))]
            if values:
                numeric[column] = values
    result["artifacts"][path.stem] = {"path": str(path), "rows": len(df), "metrics": numeric}

if target in {"gemm-conv", "all"}:
    for name in ("gemm_f16.csv", "gemm_f32.csv", "conv_f16.csv", "conv_f32.csv"):
        load_csv(Path(project_root) / name)
if target in {"longtail", "all"}:
    for name in ("ltout_gpu.csv", "ltout_fp16.csv"):
        load_csv(Path(results_dir) / name)

transformer_log = Path(logs_dir) / "transformer_block.log"
if target == "all" and transformer_log.is_file():
    import re
    values = [float(v) for v in re.findall(r"Time per iteration of (?:encoder|decoder):\\s*([0-9.eE+-]+)", transformer_log.read_text(errors="replace"))]
    if values:
        result["artifacts"]["transformer_block"] = {"path": str(transformer_log), "seconds_per_iteration": values}

if not result["artifacts"]:
    raise SystemExit("no result artifacts found for the selected target")
out = Path(results_dir) / "result.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"result.json written to {out}")
PY

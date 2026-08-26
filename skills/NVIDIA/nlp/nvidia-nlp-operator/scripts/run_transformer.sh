#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
prepare_operator_dirs

TRANSFORMER_DIR="$OPERATOR_PROJECT_ROOT/transformer_block"
test -f "$TRANSFORMER_DIR/test.py"
cd "$TRANSFORMER_DIR"
python3 test.py 2>&1 | tee "$OPERATOR_LOGS_DIR/transformer_block.log"
grep -q "Time per iteration" "$OPERATOR_LOGS_DIR/transformer_block.log"

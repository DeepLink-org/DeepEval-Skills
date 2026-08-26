#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
prepare_operator_dirs; require_dcu
TRANSFORMER_DIR="$OPERATOR_PROJECT_ROOT/speed_test/transformer_block"
LOG="$OPERATOR_LOGS_DIR/transformer_block.log"
test -f "$TRANSFORMER_DIR/test.py"
cd "$TRANSFORMER_DIR"
python3 test.py 2>&1 | tee "$LOG"
grep -q '^AIBENCH_TRANSFORMER_ENCODER_SECONDS=' "$LOG"
grep -q '^AIBENCH_TRANSFORMER_DECODER_SECONDS=' "$LOG"
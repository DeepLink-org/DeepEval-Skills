#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
PRECISION="${1:-fp32}"; require_choice PRECISION "$PRECISION" fp32 fp16
prepare_operator_dirs; require_dcu
SPEED_DIR="$OPERATOR_PROJECT_ROOT/speed_test"
if [[ "$PRECISION" == fp16 ]]; then
  BENCH_ROOT="$SPEED_DIR/LongTail-Bench-fp16"
  CASES="$SPEED_DIR/longtail_perf_gpu_fp16.csv"
else
  BENCH_ROOT="$SPEED_DIR/LongTail-Bench"
  CASES="$SPEED_DIR/longtail_perf_gpu.csv"
fi
OUTPUT="$OPERATOR_RESULTS_DIR/longtail_${PRECISION}.csv"
LOG="$OPERATOR_LOGS_DIR/longtail_${PRECISION}.log"
FILTERED_CASES="/tmp/longtail_${PRECISION}_supported_cases.csv"
test -f "$BENCH_ROOT/long_tail_bench/api/api.py"; test -f "$CASES"
cd "$BENCH_ROOT"; export PYTHONPATH="$BENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Probe the actual backend instead of statically excluding batched_nms.
# If a future image lacks this operator, run every remaining case and leave an
# auditable skip record in the log.
RUN_CASES="$CASES"
if python3 ./long_tail_bench/api/api.py --cases batched_nms >>"$LOG" 2>&1; then
  echo "batched_nms supported on this DCU backend; running all 40 cases." | tee -a "$LOG"
else
  echo "batched_nms unsupported on this DCU backend; skipping it and running 39 cases." | tee -a "$LOG"
  python3 - "$CASES" "$FILTERED_CASES" <<'PY'
import csv, sys
src, dst = sys.argv[1:]
with open(src, newline='', encoding='utf-8-sig') as f:
    r = csv.DictReader(f); fields = r.fieldnames
    rows = [row for row in r if row.get('op') != 'batched_nms']
if len(rows) != 39:
    raise SystemExit(f'expected 39 remaining longtail cases, got {len(rows)}')
with open(dst, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
PY
  RUN_CASES="$FILTERED_CASES"
fi

python3 ./long_tail_bench/api/api.py -f "$RUN_CASES" --outcsv "$OUTPUT" 2>&1 | tee -a "$LOG"
test -s "$OUTPUT"
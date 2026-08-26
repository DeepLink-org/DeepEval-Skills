#!/usr/bin/env bash

# All paths are container paths. The executor mounts the three OPERATOR_*
# paths described in SKILL.md before invoking these scripts.
OPERATOR_PROJECT_ROOT="${OPERATOR_PROJECT_ROOT:-/workspace/operators}"
OPERATOR_RESULTS_DIR="${OPERATOR_RESULTS_DIR:-/workspace/results}"
OPERATOR_LOGS_DIR="${OPERATOR_LOGS_DIR:-/workspace/logs}"

# The Hygon-adapted project is mounted directly in normal use; accept the
# older parent-directory layout as well.
if [[ -d "$OPERATOR_PROJECT_ROOT/operator/speed_test" ]]; then
  OPERATOR_PROJECT_ROOT="$OPERATOR_PROJECT_ROOT/operators"
fi
export OPERATOR_PROJECT_ROOT OPERATOR_RESULTS_DIR OPERATOR_LOGS_DIR

require_choice() {
  local name="$1" value="$2"; shift 2
  local candidate
  for candidate in "$@"; do [[ "$value" == "$candidate" ]] && return 0; done
  echo "$name must be one of: $* (got: $value)" >&2
  return 2
}

prepare_operator_dirs() {
  test -d "$OPERATOR_PROJECT_ROOT"
  mkdir -p "$OPERATOR_RESULTS_DIR" "$OPERATOR_LOGS_DIR"
}

require_dcu() {
  python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("DCU/HIP device unavailable")
if torch.cuda.device_count() < 1:
    raise SystemExit("No DCU/HIP device visible")
print(f"DCU devices: {torch.cuda.device_count()}; active: {torch.cuda.get_device_name(0)}")
PY
}

#!/usr/bin/env bash

# All paths are container paths.  Executor mounts the three OPERATOR_* paths
# described in SKILL.md before invoking these scripts.
OPERATOR_PROJECT_ROOT="${OPERATOR_PROJECT_ROOT:-/workspace/operators}"
OPERATOR_RESULTS_DIR="${OPERATOR_RESULTS_DIR:-/workspace/results}"
OPERATOR_LOGS_DIR="${OPERATOR_LOGS_DIR:-/workspace/logs}"

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

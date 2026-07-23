#!/usr/bin/env bash

# All paths are container paths. Do not resolve host-side run directories here.
MODEL_ROOT="${MODEL_ROOT:-/data/models}"
DATASET_ROOT="${DATASET_ROOT:-/data/datasets}"
DATASET_PREFER="${DATASET_PREFER:-}"

resolve_model_path() {
  local requested="${1:-}" config_path
  if [[ -n "$requested" ]]; then test -f "$requested/config.json"; printf '%s\n' "$requested"; return 0; fi
  config_path="$(find "$MODEL_ROOT" -type f -path '*/snapshots/*/config.json' -print -quit 2>/dev/null || true)"
  [[ -n "$config_path" ]] || config_path="$(find "$MODEL_ROOT" -type f -name config.json -print -quit 2>/dev/null || true)"
  [[ -n "$config_path" ]] || { echo "ERROR: no model config.json below $MODEL_ROOT" >&2; return 1; }
  dirname "$config_path"
}

resolve_dataset_path() {
  local requested="${1:-}" name path
  if [[ -n "$requested" ]]; then test -f "$requested"; printf '%s\n' "$requested"; return 0; fi
  for name in $DATASET_PREFER ShareGPT_V3_unfiltered_cleaned_split.json; do
    path="$DATASET_ROOT/$name"; [[ -f "$path" ]] && { printf '%s\n' "$path"; return 0; }
  done
  path="$(find "$DATASET_ROOT" -type f -name '*.json' -print -quit 2>/dev/null || true)"
  [[ -n "$path" ]] || { echo "ERROR: no JSON dataset below $DATASET_ROOT" >&2; return 1; }
  printf '%s\n' "$path"
}

require_positive_integer() {
  local name="$1" value="$2"
  case "$value" in ''|*[!0-9]*) echo "$name must be a positive integer" >&2; return 1;; esac
  (( value >= 1 )) || { echo "$name must be a positive integer" >&2; return 1; }
}

validate_gpu_ids() {
  local ids="${1:-}" expected="$2" count
  [[ -z "$ids" ]] && return 0
  [[ "$ids" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "GPU_IDS must be comma-separated GPU indices" >&2; return 1; }
  count=$(awk -F, '{print NF}' <<< "$ids")
  [[ "$count" == "$expected" ]] || { echo "GPU_IDS count ($count) must equal TP/GPUS_PER_NODE ($expected)" >&2; return 1; }
}

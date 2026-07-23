#!/usr/bin/env bash

# Resolve resources from the container mounts. Host-side run directories are
# deliberately not referenced here: only their Docker mount points exist in
# the container.
MODEL_ROOT="${MODEL_ROOT:-/data/models}"
DATASET_ROOT="${DATASET_ROOT:-/data/datasets}"

resolve_model_path() {
  local requested="${1:-}"
  local config_path

  if [[ -n "$requested" ]]; then
    test -f "$requested/config.json"
    printf '%s\n' "$requested"
    return 0
  fi

  # A Hugging Face cache normally has snapshots/<revision>/config.json. The
  # fallback also supports a model directory mounted directly at MODEL_ROOT.
  config_path="$(find "$MODEL_ROOT" -type f -path '*/snapshots/*/config.json' -print -quit 2>/dev/null || true)"
  if [[ -z "$config_path" ]]; then
    config_path="$(find "$MODEL_ROOT" -type f -name config.json -print -quit 2>/dev/null || true)"
  fi
  if [[ -z "$config_path" ]]; then
    echo "ERROR: no model config.json found below $MODEL_ROOT" >&2
    return 1
  fi
  dirname "$config_path"
}

resolve_dataset_path() {
  local requested="${1:-}"
  local dataset_path

  if [[ -n "$requested" ]]; then
    test -f "$requested"
    printf '%s\n' "$requested"
    return 0
  fi

  dataset_path="$DATASET_ROOT/ShareGPT_V3_unfiltered_cleaned_split.json"
  if [[ -f "$dataset_path" ]]; then
    printf '%s\n' "$dataset_path"
    return 0
  fi
  dataset_path="$(find "$DATASET_ROOT" -type f -name '*.json' -print -quit 2>/dev/null || true)"
  if [[ -z "$dataset_path" ]]; then
    echo "ERROR: no JSON dataset found below $DATASET_ROOT" >&2
    return 1
  fi
  printf '%s\n' "$dataset_path"
}

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
prepare_operator_dirs

CUDA_OPS_DIR="$OPERATOR_PROJECT_ROOT/cuda_ops"
test -f "$CUDA_OPS_DIR/CMakeLists.txt"
mkdir -p "$CUDA_OPS_DIR/build"

cmake --fresh -S "$CUDA_OPS_DIR" -B "$CUDA_OPS_DIR/build" \
  -DCUDNN_INCLUDE_DIR=/usr/local/corex-4.4.0/include \
  -DCUDNN_LIBRARIES=/usr/local/corex-4.4.0/lib64/libcudnn.so \
  2>&1 | tee "$OPERATOR_LOGS_DIR/compile.log"

cmake --build "$CUDA_OPS_DIR/build" -j"${BUILD_JOBS:-$(nproc)}" \
  2>&1 | tee -a "$OPERATOR_LOGS_DIR/compile.log"

test -x "$CUDA_OPS_DIR/build/gemm"
test -x "$CUDA_OPS_DIR/build/conv"

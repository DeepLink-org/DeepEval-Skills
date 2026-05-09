# DeepSeek-R1 (snapshot 0528) on sglang
#
# Each field uses ${VAR:-default} so pre-set env vars from the caller win over
# the config default. serve.sh / test.sh source this file and then invoke sglang.

export MODEL_PATH="${MODEL_PATH:-/data/models/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52}"
export DOCKER_IMAGE="${DOCKER_IMAGE:-sglang:nightly-dev-20251208-5e2cda61}"
export TP="${TP:-8}"
export PORT="${PORT:-30000}"
export EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:---trust-remote-code}"

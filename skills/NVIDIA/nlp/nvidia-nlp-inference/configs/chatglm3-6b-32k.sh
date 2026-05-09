# ChatGLM3-6B-32K on sglang
#
# Each field uses ${VAR:-default} so pre-set env vars from the caller win over
# the config default. serve.sh / test.sh source this file and then invoke sglang.

export MODEL_PATH="${MODEL_PATH:-glm/models/chatglm3-6b-32k}"
export DOCKER_IMAGE="${DOCKER_IMAGE:-chatglm3_image}"
export TP="${TP:-8}"
export PORT="${PORT:-30000}"
export EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:---trust-remote-code}"

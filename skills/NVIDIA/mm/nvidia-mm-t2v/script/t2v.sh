#!/usr/bin/env bash
set -euo pipefail

# Open-Sora v2 text-to-video inference, 1 GPU. Resolution/prompt/offload can be controlled by env.

MM_T2V_PROJECT_ROOT="${MM_T2V_PROJECT_ROOT:-/workspace/code}"
MM_T2V_OPENSORA_DIR="${MM_T2V_OPENSORA_DIR:-${MM_T2V_PROJECT_ROOT}/Open-Sora}"
MM_T2V_WEIGHT_DIR="${MM_T2V_WEIGHT_DIR:-/workspace/weight}"
MM_T2V_WEIGHT_PATH="${MM_T2V_WEIGHT_PATH:-${MM_T2V_WEIGHT_DIR}/Open-Sora-v2}"
MM_T2V_LOGS_DIR="${MM_T2V_LOGS_DIR:-${MM_T2V_LOG_DIR:-/workspace/logs}}"
MM_T2V_FRAME_COUNT="${MM_T2V_FRAME_COUNT:-129}"
MM_T2V_PROMPT="${MM_T2V_PROMPT:-raining, sea}"
MM_T2V_OFFLOAD="${MM_T2V_OFFLOAD:-True}"
MM_T2V_NGPU="1"
PYTHON="${PYTHON:-/usr/bin/python3}"

if [ "$#" -gt 0 ]; then
    if [ "$1" = "--resolution" ]; then
        MM_T2V_RESOLUTION="${2:-${MM_T2V_RESOLUTION:-256px}}"
    else
        MM_T2V_RESOLUTION="$1"
    fi
fi

if [ -n "${MM_T2V_RESOLUTIONS:-}" ]; then
    RESOLUTION_LIST="$MM_T2V_RESOLUTIONS"
else
    RESOLUTION_LIST="${MM_T2V_RESOLUTION:-256px}"
fi

normalize_resolution() {
    case "$1" in
        256) echo "256px" ;;
        768) echo "768px" ;;
        256px|768px) echo "$1" ;;
        *) echo "Unsupported resolution: $1. Use 256px or 768px." >&2; return 1 ;;
    esac
}

mkdir -p "$MM_T2V_LOGS_DIR"
test -d "$MM_T2V_OPENSORA_DIR"
test -d "$MM_T2V_WEIGHT_PATH"
cd "$MM_T2V_OPENSORA_DIR"

IFS=',' read -r -a RESOLUTIONS <<< "$RESOLUTION_LIST"
for RESOLUTION_RAW in "${RESOLUTIONS[@]}"; do
    RESOLUTION_RAW="$(echo "$RESOLUTION_RAW" | xargs)"
    RESOLUTION="$(normalize_resolution "$RESOLUTION_RAW")"
    LOG_FILE="${MM_T2V_LOGS_DIR}/opensora_${RESOLUTION}_gpus1.log"

    torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py "configs/diffusion/inference/${RESOLUTION}.py" \
        --prompt "$MM_T2V_PROMPT" \
        --offload "$MM_T2V_OFFLOAD" > "$LOG_FILE" 2>&1

done

export MM_T2V_PROJECT_ROOT MM_T2V_OPENSORA_DIR MM_T2V_WEIGHT_DIR MM_T2V_WEIGHT_PATH MM_T2V_LOGS_DIR MM_T2V_FRAME_COUNT MM_T2V_PROMPT MM_T2V_OFFLOAD MM_T2V_NGPU RESOLUTION_LIST
"$PYTHON" - <<'MM_T2V_PARSE'
import json
import os
import pathlib
import re

frame_count = int(os.environ.get("MM_T2V_FRAME_COUNT", "129"))
log_dir = pathlib.Path(os.environ.get("MM_T2V_LOGS_DIR", "/workspace/logs"))
opensora_dir = pathlib.Path(os.environ.get("MM_T2V_OPENSORA_DIR", "/workspace/code/Open-Sora"))
prompt = os.environ.get("MM_T2V_PROMPT", "raining, sea")
offload = os.environ.get("MM_T2V_OFFLOAD", "True")
resolutions = [r.strip() for r in os.environ.get("RESOLUTION_LIST", "256px").split(",") if r.strip()]

def normalize(value):
    return {"256": "256px", "768": "768px"}.get(value, value)

results = {}
for resolution in [normalize(r) for r in resolutions]:
    log = log_dir / f"opensora_{resolution}_gpus1.log"
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    s_iter_rows = re.findall(r"([0-9]+(?:\.[0-9]+)?)s/it", text)
    video_rows = re.findall(r"Saved to\s+([^\r\n]+)", text)
    allocated_rows = re.findall(r"CUDA max memory.*?allocated at inference:\s*([0-9]+(?:\.[0-9]+)?)\s*GB", text)
    reserved_rows = re.findall(r"CUDA max memory.*?reserved at inference:\s*([0-9]+(?:\.[0-9]+)?)\s*GB", text)

    seconds_text = s_iter_rows[-1] if s_iter_rows else None
    seconds = float(seconds_text) if seconds_text else None
    fps = frame_count / seconds if seconds else None
    output_video = video_rows[-1].strip() if video_rows else None
    output_video_abs = None
    if output_video:
        p = pathlib.Path(output_video)
        output_video_abs = str(p if p.is_absolute() else opensora_dir / p)

    key = f"opensora_{resolution}_gpus1"
    results[key] = {
        "status": "success" if "Inference finished." in text and seconds is not None else "partial",
        "model": "opensora_v2",
        "gpu_count": 1,
        "resolution": resolution,
        "prompt": prompt,
        "offload": offload,
        "frame_count": frame_count,
        "seconds_per_iter": seconds,
        "frame_count_over_seconds_per_iter": f"{frame_count}/{seconds_text}s/it" if seconds_text else None,
        "frames_per_second": round(fps, 6) if fps is not None else None,
        "log": str(log),
        "output_video": output_video,
        "output_video_abs": output_video_abs,
        "cuda_memory_allocated_gb": float(allocated_rows[-1]) if allocated_rows else None,
        "cuda_memory_reserved_gb": float(reserved_rows[-1]) if reserved_rows else None,
    }

first = next(iter(results.values()), {}) if results else {}
aggregate = {
    "status": "success" if results and all(v.get("status") == "success" for v in results.values()) else "partial",
    "task": "mm_t2v",
    "model": "opensora_v2",
    "gpu_count": 1,
    "resolutions": [v.get("resolution") for v in results.values()],
    "frame_count": frame_count,
    "metric": {
        "name": "frames_per_second",
        "formula": "frame_count / seconds_per_iter",
        "expression": first.get("frame_count_over_seconds_per_iter"),
        "value": first.get("frames_per_second"),
        "unit": "frames/s",
    },
    "results": results,
}
path = log_dir / "eval_result.json"
path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"eval result json written: {path}")
MM_T2V_PARSE

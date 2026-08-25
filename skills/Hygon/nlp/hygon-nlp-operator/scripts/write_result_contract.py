#!/usr/bin/env python3
"""Convert collect_results.sh artifacts into the AIBenchAgent result-contract v1.2."""
import argparse
import json
import math
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/workspace/results")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--workload-fingerprint", required=True)
    parser.add_argument("--duration-seconds", required=True, type=float)
    parser.add_argument("--target", choices=("longtail", "gemm", "gemm-conv", "all"), required=True)
    args = parser.parse_args()
    if not finite(args.duration_seconds) or args.duration_seconds <= 0:
        raise SystemExit("--duration-seconds must be a positive finite number")

    results_dir = Path(args.results_dir)
    raw_path = results_dir / "result.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    artifacts = raw.get("artifacts", {})
    prefixes = {
        "longtail": ("longtail_fp16", "longtail_fp32"),
        "gemm": ("gemm_fp16", "gemm_fp32"),
        "gemm-conv": ("gemm_fp16", "gemm_fp32", "conv_fp16", "conv_fp32"),
        "all": ("gemm_fp16", "gemm_fp32", "conv_fp16", "conv_fp32", "longtail_fp16", "longtail_fp32", "transformer_block"),
    }[args.target]

    metrics, count = {}, 0
    for name in prefixes:
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict) or artifact.get("valid") is not True:
            raise SystemExit(f"invalid or missing artifact: {name}")
        if name == "transformer_block":
            values = artifact.get("seconds_per_iteration", [])
            if len(values) != 2 or not all(finite(v) and v > 0 for v in values):
                raise SystemExit("invalid transformer_block metrics")
            metrics["transformer_encoder_seconds"] = values[0]
            metrics["transformer_decoder_seconds"] = values[1]
            count += 2
        else:
            value = artifact.get("mean_baseline")
            rows = artifact.get("rows")
            if not finite(value) or value <= 0 or not isinstance(rows, int) or rows <= 0:
                raise SystemExit(f"invalid numeric artifact: {name}")
            metrics[f"{name}_baseline_mean"] = value
            count += rows

    result = {
        "schema_version": "1.2",
        "task_id": args.task_id,
        "status": "success",
        "metrics": metrics,
        "metadata": {
            "workload_fingerprint": args.workload_fingerprint,
            "measurement_count": count,
            "duration_seconds": args.duration_seconds,
            "source": "Validated operator artifacts from collect_results.sh",
        },
    }
    tmp = results_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(raw_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Convert collect_results.sh artifacts into the AIBenchAgent result-contract v1.2."""
import argparse
import csv
import json
import math
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def percentile(values, fraction):
    """Linear-interpolated percentile for the benchmark p50/p95 contract."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def baseline_values(artifact, name):
    """Load the validated baseline samples instead of reducing them to a mean."""
    path = Path(artifact.get("path", ""))
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        values = [float(row["baseline"]) for row in rows]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"cannot read baseline samples for {name}: {exc}") from exc
    if len(values) != artifact.get("rows") or not values or not all(
        finite(value) and value > 0 for value in values
    ):
        raise SystemExit(f"invalid baseline samples for {name}")
    return values


def add_baseline_metrics(metrics, prefix, values, *, aggregate=False):
    total = len(values)
    metrics[f"{prefix}_total_cases"] = total
    success_key = f"{prefix}_total_success_cases" if aggregate else f"{prefix}_success_cases"
    metrics[success_key] = total
    metrics[f"{prefix}_baseline_avg"] = round(sum(values) / total, 6)
    metrics[f"{prefix}_baseline_p50"] = round(percentile(values, 0.50), 6)
    metrics[f"{prefix}_baseline_p95"] = round(percentile(values, 0.95), 6)


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

    metrics, count, grouped_values = {}, 0, {}
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
            rows = artifact.get("rows")
            if not isinstance(rows, int) or rows <= 0:
                raise SystemExit(f"invalid numeric artifact: {name}")
            values = baseline_values(artifact, name)
            add_baseline_metrics(metrics, name, values)
            family = name.rsplit("_", 1)[0]
            grouped_values.setdefault(family, []).extend(values)
            count += rows

    for family, values in grouped_values.items():
        add_baseline_metrics(metrics, family, values, aggregate=True)

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
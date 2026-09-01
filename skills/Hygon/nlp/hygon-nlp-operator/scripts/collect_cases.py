#!/usr/bin/env python3
"""把海光 DCU 算子测量 CSV 确定性转换为 Result Contract 2.0。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"missing required environment variable: {name}")
    return value


def _integer(
    row: Dict[str, str],
    name: str,
    source: Path,
    line: int,
    minimum: int = 1,
) -> int:
    try:
        value = int(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}:{line}: {name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{source}:{line}: {name} must be >= {minimum}")
    return value


def _boolean(row: Dict[str, str], name: str, source: Path, line: int) -> bool:
    value = row.get(name)
    if value not in {"0", "1"}:
        raise ValueError(f"{source}:{line}: {name} must be 0 or 1")
    return value == "1"


def _non_negative_number(
    row: Dict[str, str],
    name: str,
    source: Path,
    line: int,
) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}:{line}: {name} must be a number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"{source}:{line}: {name} must be a non-negative finite number"
        )
    return value


def _latency(row: Dict[str, str], source: Path, line: int) -> float:
    return _non_negative_number(row, "baseline", source, line)


def _read_gemm_cases(source: Path, dtype: str) -> List[Dict[str, object]]:
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"M", "N", "K", "transA", "transB", "baseline"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing columns: {', '.join(sorted(missing))}")
        cases = []
        for line, row in enumerate(reader, start=2):
            cases.append(
                {
                    "dimensions": {
                        "dtype": dtype,
                        "m": _integer(row, "M", source, line),
                        "n": _integer(row, "N", source, line),
                        "k": _integer(row, "K", source, line),
                        "trans_a": _boolean(row, "transA", source, line),
                        "trans_b": _boolean(row, "transB", source, line),
                    },
                    "metrics": {"latency_ms": _latency(row, source, line)},
                }
            )
    if not cases:
        raise ValueError(f"{source}: no benchmark cases")
    return cases


def _read_conv_cases(source: Path, dtype: str) -> List[Dict[str, object]]:
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {
            "W",
            "H",
            "C",
            "N",
            "OutC",
            "kw",
            "kh",
            "pw",
            "ph",
            "sh",
            "sv",
            "baseline",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing columns: {', '.join(sorted(missing))}")
        cases = []
        for line, row in enumerate(reader, start=2):
            cases.append(
                {
                    "dimensions": {
                        "dtype": dtype,
                        "batch_size": _integer(row, "N", source, line),
                        "input_channels": _integer(row, "C", source, line),
                        "input_height": _integer(row, "H", source, line),
                        "input_width": _integer(row, "W", source, line),
                        "output_channels": _integer(row, "OutC", source, line),
                        "kernel_height": _integer(row, "kh", source, line),
                        "kernel_width": _integer(row, "kw", source, line),
                        "padding_height": _integer(row, "ph", source, line, minimum=0),
                        "padding_width": _integer(row, "pw", source, line, minimum=0),
                        "stride_horizontal": _integer(row, "sh", source, line),
                        "stride_vertical": _integer(row, "sv", source, line),
                    },
                    "metrics": {"latency_ms": _latency(row, source, line)},
                }
            )
    if not cases:
        raise ValueError(f"{source}: no benchmark cases")
    return cases


def _read_longtail_cases(
    source: Path,
    dtype: str,
    expected_run_token: str,
) -> List[Dict[str, object]]:
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"NO", "op", "baseline", "time", "score", "aibench_run_token"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing columns: {', '.join(sorted(missing))}")

        cases = []
        seen_numbers = set()
        seen_operators = set()
        for line, row in enumerate(reader, start=2):
            case_number = _integer(row, "NO", source, line, minimum=0)
            operator = (row.get("op") or "").strip()
            if not operator:
                raise ValueError(f"{source}:{line}: op must be a non-empty string")
            if row.get("aibench_run_token") != expected_run_token:
                raise ValueError(
                    f"{source}:{line}: result does not belong to the run-scoped case manifest"
                )
            if case_number in seen_numbers:
                raise ValueError(f"{source}:{line}: duplicate NO: {case_number}")
            if operator in seen_operators:
                raise ValueError(f"{source}:{line}: duplicate op: {operator}")
            seen_numbers.add(case_number)
            seen_operators.add(operator)
            cases.append(
                {
                    "dimensions": {"operator": operator, "dtype": dtype},
                    "metrics": {"latency_ms": _latency(row, source, line)},
                }
            )
    if not cases:
        raise ValueError(f"{source}: no benchmark cases")
    return cases


def _read_longtail_manifest(source: Path) -> Tuple[List[str], str]:
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"NO", "op", "baseline", "time", "score", "aibench_run_token"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing columns: {', '.join(sorted(missing))}")
        operators = []
        run_token = None
        seen_numbers = set()
        seen_operators = set()
        for line, row in enumerate(reader, start=2):
            case_number = _integer(row, "NO", source, line, minimum=0)
            operator = (row.get("op") or "").strip()
            if not operator:
                raise ValueError(f"{source}:{line}: op must be a non-empty string")
            row_run_token = (row.get("aibench_run_token") or "").strip()
            if not row_run_token:
                raise ValueError(
                    f"{source}:{line}: aibench_run_token must be a non-empty string"
                )
            if run_token is None:
                run_token = row_run_token
            elif row_run_token != run_token:
                raise ValueError(
                    f"{source}:{line}: aibench_run_token must be consistent"
                )
            if case_number in seen_numbers:
                raise ValueError(f"{source}:{line}: duplicate NO: {case_number}")
            if operator in seen_operators:
                raise ValueError(f"{source}:{line}: duplicate op: {operator}")
            seen_numbers.add(case_number)
            seen_operators.add(operator)
            operators.append(operator)
    if not operators:
        raise ValueError(f"{source}: no benchmark cases")
    assert run_token is not None
    return operators, run_token


def _read_transformer_block_cases(
    source: Path,
    expected_workload_fingerprint: str,
) -> List[Dict[str, object]]:
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {
            "block_type",
            "dtype",
            "execution_mode",
            "d_model",
            "num_heads",
            "ffn_hidden_size",
            "batch_size",
            "query_sequence_length",
            "key_value_sequence_length",
            "warmup_iterations",
            "measurement_iterations",
            "latency_ms",
            "aibench_workload_fingerprint",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing columns: {', '.join(sorted(missing))}")

        cases = []
        seen_block_types = set()
        for line, row in enumerate(reader, start=2):
            block_type = (row.get("block_type") or "").strip()
            if block_type not in {"encoder", "decoder"}:
                raise ValueError(
                    f"{source}:{line}: block_type must be encoder or decoder"
                )
            if block_type in seen_block_types:
                raise ValueError(f"{source}:{line}: duplicate block_type: {block_type}")
            if row.get("dtype") != "f32":
                raise ValueError(f"{source}:{line}: dtype must be f32")
            if row.get("execution_mode") != "inference":
                raise ValueError(f"{source}:{line}: execution_mode must be inference")
            if row.get("aibench_workload_fingerprint") != expected_workload_fingerprint:
                raise ValueError(
                    f"{source}:{line}: result does not belong to the active workload"
                )
            seen_block_types.add(block_type)
            cases.append(
                {
                    "dimensions": {
                        "block_type": block_type,
                        "dtype": "f32",
                        "execution_mode": "inference",
                        "d_model": _integer(row, "d_model", source, line),
                        "num_heads": _integer(row, "num_heads", source, line),
                        "ffn_hidden_size": _integer(
                            row, "ffn_hidden_size", source, line
                        ),
                        "batch_size": _integer(row, "batch_size", source, line),
                        "query_sequence_length": _integer(
                            row, "query_sequence_length", source, line
                        ),
                        "key_value_sequence_length": _integer(
                            row, "key_value_sequence_length", source, line
                        ),
                        "warmup_iterations": _integer(
                            row, "warmup_iterations", source, line, minimum=0
                        ),
                        "measurement_iterations": _integer(
                            row, "measurement_iterations", source, line
                        ),
                    },
                    "metrics": {
                        "latency_ms": _non_negative_number(
                            row, "latency_ms", source, line
                        )
                    },
                }
            )
    if seen_block_types != {"encoder", "decoder"}:
        raise ValueError(f"{source}: expected exactly one encoder and one decoder case")
    return cases


def _percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def build_result(
    input_dir: Path,
    duration_seconds: float,
    benchmark: str = "gemm",
) -> Dict[str, object]:
    if benchmark == "gemm":
        filenames = (("gemm_f16.csv", "f16"), ("gemm_f32.csv", "f32"))
        case_reader = _read_gemm_cases
        metric_prefix = "gemm"
    elif benchmark == "conv":
        filenames = (("conv_f16.csv", "f16"), ("conv_f32.csv", "f32"))
        case_reader = _read_conv_cases
        metric_prefix = "conv"
    elif benchmark == "longtail":
        filenames = (
            ("longtail_perf_gpu.csv", "f32"),
            ("longtail_perf_gpu_fp16.csv", "f16"),
        )
        case_reader = _read_longtail_cases
        metric_prefix = "longtail"
        manifest_source = input_dir / "longtail_cases_input.csv"
        if not manifest_source.is_file():
            raise ValueError(f"missing longtail case manifest: {manifest_source}")
        expected_longtail_identities, expected_longtail_run_token = (
            _read_longtail_manifest(manifest_source)
        )
    elif benchmark == "transformer_block":
        filenames = ()
        metric_prefix = "transformer_block"
    else:
        raise ValueError(f"unsupported benchmark: {benchmark}")

    cases = []
    sources = []
    longtail_identities = []
    if benchmark == "transformer_block":
        filename = "transformer_block_cases.csv"
        source = input_dir / filename
        if not source.is_file():
            raise ValueError(f"missing transformer_block CSV: {source}")
        sources.append(filename)
        cases.extend(
            _read_transformer_block_cases(
                source,
                _required_env("AIBENCH_WORKLOAD_FINGERPRINT"),
            )
        )
    else:
        for filename, dtype in filenames:
            source = input_dir / filename
            if not source.is_file():
                raise ValueError(f"missing {benchmark} CSV: {source}")
            sources.append(filename)
            if benchmark == "longtail":
                source_cases = case_reader(source, dtype, expected_longtail_run_token)
            else:
                source_cases = case_reader(source, dtype)
            cases.extend(source_cases)
            if benchmark == "longtail":
                longtail_identities.append(
                    [item["dimensions"]["operator"] for item in source_cases]
                )

    if benchmark == "longtail" and any(
        identities != expected_longtail_identities for identities in longtail_identities
    ):
        raise ValueError(
            "longtail f32 and f16 CSVs must match the run-scoped case manifest"
        )

    latencies = [float(item["metrics"]["latency_ms"]) for item in cases]
    total = len(cases)
    latency_label = "latency" if benchmark == "transformer_block" else "baseline"
    metrics = {
        f"{metric_prefix}_total_cases": total,
        f"{metric_prefix}_success_cases": total,
        f"{metric_prefix}_failed_cases": 0,
        f"{metric_prefix}_{latency_label}_avg_ms": sum(latencies) / total,
        f"{metric_prefix}_{latency_label}_p50_ms": _percentile(latencies, 0.50),
        f"{metric_prefix}_{latency_label}_p95_ms": _percentile(latencies, 0.95),
        f"{metric_prefix}_{latency_label}_min_ms": min(latencies),
        f"{metric_prefix}_{latency_label}_max_ms": max(latencies),
    }
    source_description = " and ".join(sources)
    if benchmark == "transformer_block":
        source_description += " measured latency_ms"
    else:
        source_description += " baseline values"
    return {
        "schema_version": "2.0",
        "task_id": _required_env("AIBENCH_TASK_ID"),
        "status": "success",
        "benchmark": {
            "spec_id": _required_env("AIBENCH_BENCHMARK_SPEC_ID"),
            "spec_version": _required_env("AIBENCH_BENCHMARK_SPEC_VERSION"),
            "case_schema_version": _required_env(
                "AIBENCH_BENCHMARK_CASE_SCHEMA_VERSION"
            ),
            "spec_sha256": _required_env("AIBENCH_BENCHMARK_SPEC_SHA256"),
        },
        "metrics": metrics,
        "cases": cases,
        "metadata": {
            "workload_fingerprint": _required_env("AIBENCH_WORKLOAD_FINGERPRINT"),
            "measurement_count": total,
            "duration_seconds": duration_seconds,
            "source": source_description,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=("gemm", "conv", "longtail", "transformer_block"),
        default="gemm",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("/workspace/operators"))
    parser.add_argument(
        "--output", type=Path, default=Path("/workspace/results/result.json")
    )
    parser.add_argument("--duration-seconds", type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
        parser.error("--duration-seconds must be a positive finite number")

    result = build_result(
        args.input_dir, args.duration_seconds, benchmark=args.benchmark
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    encoded = json.dumps(
        result, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    temporary.write_text(encoded, encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, args.output)
    print(f"result.json: {encoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

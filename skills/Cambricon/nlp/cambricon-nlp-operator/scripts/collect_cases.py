#!/usr/bin/env python3
"""Convert Cambricon MLU operator artifacts to Result Contract 2.0."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple


Case = Dict[str, object]


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


def _number(
    row: Dict[str, str],
    name: str,
    source: Path,
    line: int,
) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}:{line}: {name} must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{source}:{line}: {name} must be a positive finite number")
    return value


def _require_columns(reader: csv.DictReader, source: Path, required: set[str]) -> None:
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{source}: missing columns: {', '.join(sorted(missing))}")


def _geometry(
    row: Dict[str, str],
    source: Path,
    line: int,
    *,
    nvidia_stride_names: bool = False,
) -> Dict[str, object]:
    dimensions = {
        "batch_size": _integer(row, "N", source, line),
        "input_channels": _integer(row, "C", source, line),
        "input_height": _integer(row, "H", source, line),
        "input_width": _integer(row, "W", source, line),
        "output_channels": _integer(row, "OutC", source, line),
        "kernel_height": _integer(row, "kh", source, line),
        "kernel_width": _integer(row, "kw", source, line),
        "padding_height": _integer(row, "ph", source, line, minimum=0),
        "padding_width": _integer(row, "pw", source, line, minimum=0),
    }
    if nvidia_stride_names:
        dimensions["stride_horizontal"] = _integer(row, "sh", source, line)
        dimensions["stride_vertical"] = _integer(row, "sw", source, line)
    else:
        dimensions["stride_height"] = _integer(row, "sh", source, line)
        dimensions["stride_width"] = _integer(row, "sw", source, line)
    return dimensions


def _read_gemm_cases(source: Path, dtype: str) -> List[Case]:
    expected_dtype_code = {"f16": 0, "f32": 1}[dtype]
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _require_columns(
            reader,
            source,
            {"M", "N", "K", "transA", "transB", "i_d", "o_d", "baseline"},
        )
        cases: List[Case] = []
        for line, row in enumerate(reader, start=2):
            for column in ("i_d", "o_d"):
                if _integer(row, column, source, line, minimum=0) != expected_dtype_code:
                    raise ValueError(f"{source}:{line}: {column} does not match {dtype}")
            latency = _number(row, "baseline", source, line)
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
                    "metrics": {"latency_ms": latency},
                }
            )
    if not cases:
        raise ValueError(f"{source}: no GEMM cases")
    return cases


def _read_conv_cases(source: Path, dtype: str) -> List[Case]:
    required = {
        "W", "H", "C", "N", "OutC", "kw", "kh", "pw", "ph", "sh", "sw",
        "baseline",
    }
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _require_columns(reader, source, required)
        cases: List[Case] = []
        for line, row in enumerate(reader, start=2):
            latency = _number(row, "baseline", source, line)
            cases.append(
                {
                    "dimensions": {
                        "dtype": dtype,
                        **_geometry(
                            row, source, line, nvidia_stride_names=True
                        ),
                    },
                    "metrics": {"latency_ms": latency},
                }
            )
    if not cases:
        raise ValueError(f"{source}: no Conv2d cases")
    return cases


def _read_conv_component_cases(source: Path, dtype: str, phase: str) -> List[Case]:
    required = {
        "W", "H", "C", "N", "OutC", "kw", "kh", "pw", "ph", "sh", "sw",
        "baseline",
    }
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _require_columns(reader, source, required)
        cases: List[Case] = []
        for line, row in enumerate(reader, start=2):
            latency = _number(row, "baseline", source, line)
            cases.append(
                {
                    "dimensions": {
                        "phase": phase,
                        "dtype": dtype,
                        **_geometry(row, source, line),
                    },
                    "metrics": {"latency_ms": latency},
                }
            )
    if not cases:
        raise ValueError(f"{source}: no Conv2d component cases")
    return cases


def _read_longtail_manifest(source: Path) -> Tuple[List[Tuple[int, str]], str]:
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _require_columns(
            reader,
            source,
            {
                "NO", "op", "baseline", "time", "score", "inputshapes",
                "aibench_run_token",
            },
        )
        identities: List[Tuple[int, str]] = []
        seen_numbers = set()
        seen_operators = set()
        run_token = ""
        for line, row in enumerate(reader, start=2):
            number = _integer(row, "NO", source, line, minimum=0)
            operator = (row.get("op") or "").strip()
            if not operator:
                raise ValueError(f"{source}:{line}: op must be non-empty")
            if number in seen_numbers or operator in seen_operators:
                raise ValueError(f"{source}:{line}: duplicate LongTail case")
            if any((row.get(column) or "").strip() for column in ("baseline", "time", "score")):
                raise ValueError(
                    f"{source}:{line}: manifest measurement fields must be empty"
                )
            row_token = (row.get("aibench_run_token") or "").strip()
            if not row_token:
                raise ValueError(
                    f"{source}:{line}: aibench_run_token must be non-empty"
                )
            if not run_token:
                run_token = row_token
            elif row_token != run_token:
                raise ValueError(
                    f"{source}:{line}: aibench_run_token must be consistent"
                )
            seen_numbers.add(number)
            seen_operators.add(operator)
            identities.append((number, operator))
    if not identities:
        raise ValueError(f"{source}: no LongTail cases")
    return identities, run_token


def _read_longtail_cases(
    source: Path,
    expected_identities: List[Tuple[int, str]],
    expected_run_token: str,
) -> List[Case]:
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _require_columns(
            reader,
            source,
            {
                "NO", "op", "baseline", "time", "score", "inputshapes",
                "aibench_run_token",
            },
        )
        cases: List[Case] = []
        identities: List[Tuple[int, str]] = []
        seen_numbers = set()
        seen_operators = set()
        for line, row in enumerate(reader, start=2):
            number = _integer(row, "NO", source, line, minimum=0)
            operator = (row.get("op") or "").strip()
            if not operator:
                raise ValueError(f"{source}:{line}: op must be non-empty")
            if number in seen_numbers or operator in seen_operators:
                raise ValueError(f"{source}:{line}: duplicate LongTail case")
            if row.get("aibench_run_token") != expected_run_token:
                raise ValueError(f"{source}:{line}: stale aibench_run_token")
            seen_numbers.add(number)
            seen_operators.add(operator)
            identities.append((number, operator))
            latency = _number(row, "baseline", source, line)
            cases.append(
                {
                    "dimensions": {"operator": operator, "dtype": "f32"},
                    "metrics": {"latency_ms": latency},
                }
            )
    if not cases:
        raise ValueError(f"{source}: no LongTail cases")
    if identities != expected_identities:
        raise ValueError(f"{source}: case order or coverage differs from the manifest")
    return cases


def _read_accuracy_cases(source: Path) -> List[Case]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{source}: accuracy result must be a non-empty object")
    cases: List[Case] = []
    for operator in sorted(payload):
        result = payload[operator]
        if not isinstance(operator, str) or not operator:
            raise ValueError(f"{source}: operator names must be non-empty strings")
        if not isinstance(result, dict) or type(result.get("passed_fp32")) is not bool:
            raise ValueError(f"{source}: {operator}.passed_fp32 must be boolean")
        cases.append(
            {
                "dimensions": {"operator": operator, "dtype": "f32"},
                "metrics": {"passed": 1.0 if result["passed_fp32"] else 0.0},
            }
        )
    return cases


def _read_transformer_cases(source: Path, fingerprint: str) -> List[Case]:
    required = {
        "block_type", "dtype", "execution_mode", "d_model", "num_heads",
        "ffn_hidden_size", "batch_size", "query_sequence_length",
        "key_value_sequence_length", "warmup_iterations", "measurement_iterations",
        "latency_ms", "aibench_workload_fingerprint",
    }
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _require_columns(reader, source, required)
        cases: List[Case] = []
        seen = set()
        for line, row in enumerate(reader, start=2):
            block_type = (row.get("block_type") or "").strip()
            if block_type not in {"encoder", "decoder"}:
                raise ValueError(f"{source}:{line}: invalid block_type")
            if block_type in seen:
                raise ValueError(f"{source}:{line}: duplicate block_type")
            if row.get("dtype") != "f32" or row.get("execution_mode") != "inference":
                raise ValueError(f"{source}:{line}: transformer must be f32 inference")
            if row.get("aibench_workload_fingerprint") != fingerprint:
                raise ValueError(f"{source}:{line}: stale workload fingerprint")
            seen.add(block_type)
            cases.append(
                {
                    "dimensions": {
                        "block_type": block_type,
                        "dtype": "f32",
                        "execution_mode": "inference",
                        "d_model": _integer(row, "d_model", source, line),
                        "num_heads": _integer(row, "num_heads", source, line),
                        "ffn_hidden_size": _integer(row, "ffn_hidden_size", source, line),
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
                        "latency_ms": _number(row, "latency_ms", source, line)
                    },
                }
            )
    if seen != {"encoder", "decoder"}:
        raise ValueError(f"{source}: expected exactly one encoder and one decoder")
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


def _performance_metrics(
    prefix: str,
    cases: List[Case],
    summary_label: str = "latency",
) -> Dict[str, float]:
    latencies = [float(case["metrics"]["latency_ms"]) for case in cases]  # type: ignore[index]
    total = len(latencies)
    return {
        f"{prefix}_total_cases": total,
        f"{prefix}_success_cases": total,
        f"{prefix}_failed_cases": 0,
        f"{prefix}_{summary_label}_avg_ms": sum(latencies) / total,
        f"{prefix}_{summary_label}_p50_ms": _percentile(latencies, 0.50),
        f"{prefix}_{summary_label}_p95_ms": _percentile(latencies, 0.95),
        f"{prefix}_{summary_label}_min_ms": min(latencies),
        f"{prefix}_{summary_label}_max_ms": max(latencies),
    }


def _load_cases(input_dir: Path, benchmark: str) -> Tuple[List[Case], List[str], str]:
    cases: List[Case] = []
    sources: List[str] = []
    if benchmark == "accuracy":
        relative = Path("accuracy/mlu_val_result.json")
        source = input_dir / relative
        if not source.is_file():
            raise ValueError(f"missing accuracy JSON: {source}")
        return _read_accuracy_cases(source), [str(relative)], "accuracy"
    if benchmark == "gemm":
        for label, dtype in (("FP16", "f16"), ("FP32", "f32")):
            relative = Path(f"gemm/gemm_{label}_result.csv")
            source = input_dir / relative
            if not source.is_file():
                raise ValueError(f"missing GEMM CSV: {source}")
            cases.extend(_read_gemm_cases(source, dtype))
            sources.append(str(relative))
        return cases, sources, "gemm"
    if benchmark == "conv":
        for label, dtype in (("FP16", "f16"), ("FP32", "f32")):
            relative = Path(f"conv/conv_total_{label}_result.csv")
            source = input_dir / relative
            if not source.is_file():
                raise ValueError(f"missing Conv2d CSV: {source}")
            cases.extend(_read_conv_cases(source, dtype))
            sources.append(str(relative))
        return cases, sources, "conv"
    if benchmark in {"convbackdata", "convbackfilter"}:
        stem, phase = (
            ("convbk_data", "backward_data")
            if benchmark == "convbackdata"
            else ("convbk_filter", "backward_filter")
        )
        for label, dtype in (("FP16", "f16"), ("FP32", "f32")):
            relative = Path(f"conv/{stem}_{label}_result.csv")
            source = input_dir / relative
            if not source.is_file():
                raise ValueError(f"missing Conv2d component CSV: {source}")
            cases.extend(_read_conv_component_cases(source, dtype, phase))
            sources.append(str(relative))
        return cases, sources, "conv_component"
    if benchmark == "longtail":
        manifest_relative = Path("longtail/longtail_cases_input.csv")
        manifest_source = input_dir / manifest_relative
        if not manifest_source.is_file():
            raise ValueError(f"missing LongTail manifest: {manifest_source}")
        identities, run_token = _read_longtail_manifest(manifest_source)
        relative = Path("longtail/longtail_result.csv")
        source = input_dir / relative
        if not source.is_file():
            raise ValueError(f"missing LongTail CSV: {source}")
        return (
            _read_longtail_cases(source, identities, run_token),
            [str(manifest_relative), str(relative)],
            "longtail",
        )
    if benchmark == "transformer_block":
        relative = Path("transformer/transformer_block_cases.csv")
        source = input_dir / relative
        if not source.is_file():
            raise ValueError(f"missing Transformer Block CSV: {source}")
        return (
            _read_transformer_cases(
                source, _required_env("AIBENCH_WORKLOAD_FINGERPRINT")
            ),
            [str(relative)],
            "transformer_block",
        )
    raise ValueError(f"unsupported benchmark: {benchmark}")


def build_result(input_dir: Path, duration_seconds: float, benchmark: str) -> Dict[str, object]:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be a positive finite number")
    cases, sources, prefix = _load_cases(input_dir, benchmark)
    if benchmark == "accuracy":
        passed = sum(float(case["metrics"]["passed"]) for case in cases)  # type: ignore[index]
        total = len(cases)
        metrics: Dict[str, float] = {
            "accuracy_total_cases": total,
            "accuracy_passed_cases": passed,
            "accuracy_failed_cases": total - passed,
            "accuracy_pass_rate": passed / total,
        }
    else:
        # 与 NVIDIA operator Skill 的 summary contract 对齐。case 级数据统一
        # 使用 latency_ms；GEMM/Conv/LongTail 的值来自本轮生成的 baseline 列。
        summary_label = (
            "baseline" if benchmark in {"gemm", "conv", "longtail"} else "latency"
        )
        metrics = _performance_metrics(prefix, cases, summary_label)

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
            "measurement_count": len(cases),
            "duration_seconds": duration_seconds,
            "source": "measured artifacts: " + ", ".join(sources),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=(
            "accuracy",
            "gemm",
            "conv",
            "convbackdata",
            "convbackfilter",
            "longtail",
            "transformer_block",
        ),
    )
    parser.add_argument("--input-dir", type=Path, default=Path("/workspace/results"))
    parser.add_argument(
        "--output", type=Path, default=Path("/workspace/results/result.json")
    )
    parser.add_argument("--duration-seconds", type=float, required=True)
    args = parser.parse_args()

    result = build_result(args.input_dir, args.duration_seconds, args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    encoded = json.dumps(
        result, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    try:
        temporary.write_text(encoded, encoding="utf-8")
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"result.json: {encoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

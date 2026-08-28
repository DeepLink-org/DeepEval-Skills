#!/usr/bin/env python3
"""Convert Ascend operator artifacts to AIBench Result Contract 2.0."""

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


def _require_columns(reader: csv.DictReader, source: Path, required: set[str]) -> None:
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{source}: missing columns: {', '.join(sorted(missing))}")


def _integer(
    row: Dict[str, str], name: str, source: Path, line: int, minimum: int = 1
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


def _positive(row: Dict[str, str], name: str, source: Path, line: int) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}:{line}: {name} must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{source}:{line}: {name} must be positive and finite")
    return value


def _require_empty(row: Dict[str, str], name: str, source: Path, line: int) -> None:
    raw = (row.get(name) or "").strip()
    if raw.lower() not in {"", "nan", "none"}:
        raise ValueError(f"{source}:{line}: {name} must be empty")


def _check_unique(identity: tuple, seen: set[tuple], source: Path, line: int) -> None:
    if identity in seen:
        raise ValueError(f"{source}:{line}: duplicate case identity: {identity}")
    seen.add(identity)


def _read_gemm(source: Path, dtype: str) -> List[Case]:
    required = {
        "NO", "M", "N", "K", "transA", "transB", "baseline", "time", "score"
    }
    cases: List[Case] = []
    seen: set[tuple] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _require_columns(reader, source, required)
        for line, row in enumerate(reader, start=2):
            if _integer(row, "NO", source, line, minimum=0) != len(cases):
                raise ValueError(f"{source}:{line}: NO must be contiguous from zero")
            dimensions = {
                "dtype": dtype,
                "m": _integer(row, "M", source, line),
                "n": _integer(row, "N", source, line),
                "k": _integer(row, "K", source, line),
                "trans_a": _boolean(row, "transA", source, line),
                "trans_b": _boolean(row, "transB", source, line),
            }
            identity = tuple(dimensions.values())
            _check_unique(identity, seen, source, line)
            _require_empty(row, "time", source, line)
            _require_empty(row, "score", source, line)
            cases.append(
                {
                    "dimensions": dimensions,
                    "metrics": {
                        "latency_ms": _positive(row, "baseline", source, line)
                    },
                }
            )
    if not cases:
        raise ValueError(f"{source}: no GEMM cases")
    return cases


def _read_conv(source: Path, dtype: str) -> List[Case]:
    required = {
        "W", "H", "C", "N", "OutC", "kw", "kh", "pw", "ph", "sh", "sv",
        "NO", "baseline", "time", "score", "forward_ms",
        "backward_weight_ms", "backward_data_ms",
    }
    cases: List[Case] = []
    seen: set[tuple] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _require_columns(reader, source, required)
        for line, row in enumerate(reader, start=2):
            if _integer(row, "NO", source, line, minimum=0) != len(cases):
                raise ValueError(f"{source}:{line}: NO must be contiguous from zero")
            dimensions = {
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
            }
            identity = tuple(dimensions.values())
            _check_unique(identity, seen, source, line)
            total = _positive(row, "baseline", source, line)
            _require_empty(row, "time", source, line)
            _require_empty(row, "score", source, line)
            component_sum = sum(
                _positive(row, name, source, line)
                for name in (
                    "forward_ms", "backward_weight_ms", "backward_data_ms"
                )
            )
            if not math.isclose(total, component_sum, rel_tol=1e-5, abs_tol=5e-6):
                raise ValueError(
                    f"{source}:{line}: baseline must equal forward + backward_weight + backward_data"
                )
            cases.append(
                {"dimensions": dimensions, "metrics": {"latency_ms": total}}
            )
    if not cases:
        raise ValueError(f"{source}: no Conv2d cases")
    return cases


def _read_longtail(source: Path, dtype: str, run_token: str) -> Tuple[List[Case], int]:
    required = {
        "NO", "op", "baseline", "time", "score", "aibench_run_token"
    }
    cases: List[Case] = []
    attempted = 0
    seen_numbers: set[int] = set()
    seen_operators: set[str] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _require_columns(reader, source, required)
        for line, row in enumerate(reader, start=2):
            attempted += 1
            number = _integer(row, "NO", source, line, minimum=0)
            operator = (row.get("op") or "").strip()
            if not operator:
                raise ValueError(f"{source}:{line}: op must be non-empty")
            if number in seen_numbers or operator in seen_operators:
                raise ValueError(f"{source}:{line}: duplicate LongTail case")
            if row.get("aibench_run_token") != run_token:
                raise ValueError(f"{source}:{line}: stale aibench_run_token")
            seen_numbers.add(number)
            seen_operators.add(operator)
            _require_empty(row, "time", source, line)
            _require_empty(row, "score", source, line)
            cases.append(
                {
                    "dimensions": {"operator": operator, "dtype": dtype},
                    "metrics": {
                        "latency_ms": _positive(row, "baseline", source, line)
                    },
                }
            )
    if attempted == 0:
        raise ValueError(f"{source}: no LongTail cases")
    return cases, attempted


def _read_accuracy(source: Path) -> List[Case]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{source}: accuracy result must be a non-empty object")
    cases: List[Case] = []
    for operator in sorted(payload):
        result = payload[operator]
        if not isinstance(operator, str) or not operator:
            raise ValueError(f"{source}: invalid operator name")
        if not isinstance(result, dict):
            raise ValueError(f"{source}: {operator} result must be an object")
        for json_dtype, case_dtype in (("fp32", "f32"), ("fp16", "f16")):
            value = result.get(json_dtype)
            if value is None:
                continue
            if type(value) is not bool:
                raise ValueError(f"{source}: {operator}.{json_dtype} must be boolean or null")
            cases.append(
                {
                    "dimensions": {"operator": operator, "dtype": case_dtype},
                    "metrics": {"passed": 1.0 if value else 0.0},
                }
            )
    if not cases:
        raise ValueError(f"{source}: no tested accuracy cases")
    return cases


def _read_transformer(source: Path, fingerprint: str) -> List[Case]:
    required = {
        "block_type", "dtype", "execution_mode", "d_model", "num_heads",
        "ffn_hidden_size", "batch_size", "query_sequence_length",
        "key_value_sequence_length", "warmup_iterations", "measurement_iterations",
        "latency_ms", "aibench_workload_fingerprint",
    }
    cases: List[Case] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _require_columns(reader, source, required)
        for line, row in enumerate(reader, start=2):
            block_type = (row.get("block_type") or "").strip()
            if block_type not in {"encoder", "decoder"} or block_type in seen:
                raise ValueError(f"{source}:{line}: invalid or duplicate block_type")
            if row.get("dtype") != "f32" or row.get("execution_mode") != "inference":
                raise ValueError(f"{source}:{line}: expected f32 inference")
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
                        "latency_ms": _positive(row, "latency_ms", source, line)
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
    prefix: str, cases: List[Case], attempted: int, label: str
) -> Dict[str, float]:
    if attempted != len(cases):
        raise ValueError(
            f"{prefix}: incomplete result: measured {len(cases)} of {attempted} cases"
        )
    latencies = [float(case["metrics"]["latency_ms"]) for case in cases]  # type: ignore[index]
    if not latencies:
        raise ValueError(f"{prefix}: no successful measured cases")
    return {
        f"{prefix}_total_cases": attempted,
        f"{prefix}_success_cases": len(latencies),
        f"{prefix}_failed_cases": attempted - len(latencies),
        f"{prefix}_{label}_avg_ms": sum(latencies) / len(latencies),
        f"{prefix}_{label}_p50_ms": _percentile(latencies, 0.50),
        f"{prefix}_{label}_p95_ms": _percentile(latencies, 0.95),
        f"{prefix}_{label}_min_ms": min(latencies),
        f"{prefix}_{label}_max_ms": max(latencies),
    }


def _verify_fingerprint(source: Path) -> None:
    if not source.is_file():
        raise ValueError(f"missing workload fingerprint: {source}")
    actual = source.read_text(encoding="utf-8").strip()
    if actual != _required_env("AIBENCH_WORKLOAD_FINGERPRINT"):
        raise ValueError(f"{source}: result belongs to a stale workload")


def _identities_without_dtype(cases: List[Case]) -> List[tuple]:
    identities = []
    for case in cases:
        dimensions = case["dimensions"]
        assert isinstance(dimensions, dict)
        identities.append(tuple(value for key, value in dimensions.items() if key != "dtype"))
    return identities


def _load_cases(
    input_dir: Path, benchmark: str
) -> Tuple[List[Case], int, List[str], str, str]:
    if benchmark == "accuracy":
        relative = Path("accuracy/npu_val_result.json")
        source = input_dir / relative
        if not source.is_file():
            raise ValueError(f"missing accuracy JSON: {source}")
        fingerprint_file = input_dir / "accuracy/aibench_workload_fingerprint.txt"
        if fingerprint_file.read_text(encoding="utf-8").strip() != _required_env(
            "AIBENCH_WORKLOAD_FINGERPRINT"
        ):
            raise ValueError("accuracy result belongs to a stale workload")
        cases = _read_accuracy(source)
        return cases, len(cases), [str(relative)], "accuracy", "accuracy"

    if benchmark in {"gemm", "conv"}:
        _verify_fingerprint(
            input_dir / benchmark / "aibench_workload_fingerprint.txt"
        )
        reader = _read_gemm if benchmark == "gemm" else _read_conv
        cases: List[Case] = []
        sources: List[str] = []
        expected_per_dtype = 224 if benchmark == "gemm" else 63
        reference_identities: List[tuple] | None = None
        for dtype in ("f16", "f32"):
            relative = Path(f"{benchmark}/{benchmark}_{dtype}.csv")
            source = input_dir / relative
            if not source.is_file():
                raise ValueError(f"missing {benchmark} CSV: {source}")
            dtype_cases = reader(source, dtype)
            if len(dtype_cases) != expected_per_dtype:
                raise ValueError(
                    f"{source}: expected {expected_per_dtype} cases, found {len(dtype_cases)}"
                )
            identities = _identities_without_dtype(dtype_cases)
            if reference_identities is None:
                reference_identities = identities
            elif identities != reference_identities:
                raise ValueError(f"{benchmark} f16/f32 case identities differ")
            cases.extend(dtype_cases)
            sources.append(str(relative))
        return cases, len(cases), sources, benchmark, "baseline"

    if benchmark == "longtail":
        token_file = input_dir / "longtail/aibench_run_token.txt"
        if not token_file.is_file():
            raise ValueError(f"missing LongTail run token: {token_file}")
        run_token = token_file.read_text(encoding="utf-8").strip()
        if not run_token:
            raise ValueError("LongTail run token is empty")
        fingerprint = _required_env("AIBENCH_WORKLOAD_FINGERPRINT")
        if not run_token.startswith(fingerprint + ":"):
            raise ValueError("LongTail result belongs to a stale workload")
        cases = []
        attempted = 0
        sources = []
        reference_identities: List[tuple] | None = None
        for filename, dtype in (("longtail_fp32.csv", "f32"), ("longtail_fp16.csv", "f16")):
            relative = Path("longtail") / filename
            source = input_dir / relative
            if not source.is_file():
                raise ValueError(f"missing LongTail CSV: {source}")
            file_cases, file_attempted = _read_longtail(source, dtype, run_token)
            if file_attempted != 40:
                raise ValueError(
                    f"{source}: expected 40 cases, found {file_attempted}"
                )
            identities = _identities_without_dtype(file_cases)
            if reference_identities is None:
                reference_identities = identities
            elif identities != reference_identities:
                raise ValueError("LongTail f16/f32 case identities differ")
            cases.extend(file_cases)
            attempted += file_attempted
            sources.append(str(relative))
        return cases, attempted, sources, "longtail", "baseline"

    if benchmark == "transformer_block":
        relative = Path("transformer/transformer_block_cases.csv")
        source = input_dir / relative
        if not source.is_file():
            raise ValueError(f"missing Transformer CSV: {source}")
        cases = _read_transformer(
            source, _required_env("AIBENCH_WORKLOAD_FINGERPRINT")
        )
        return cases, len(cases), [str(relative)], "transformer_block", "latency"

    raise ValueError(f"unsupported benchmark: {benchmark}")


def build_result(input_dir: Path, duration_seconds: float, benchmark: str) -> Dict[str, object]:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    cases, attempted, sources, prefix, label = _load_cases(input_dir, benchmark)
    if benchmark == "accuracy":
        passed = sum(float(case["metrics"]["passed"]) for case in cases)  # type: ignore[index]
        metrics: Dict[str, float] = {
            "accuracy_total_cases": len(cases),
            "accuracy_passed_cases": passed,
            "accuracy_failed_cases": len(cases) - passed,
            "accuracy_pass_rate": passed / len(cases),
        }
    else:
        metrics = _performance_metrics(prefix, cases, attempted, label)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=("accuracy", "gemm", "conv", "longtail", "transformer_block"),
    )
    parser.add_argument("--input-dir", type=Path, default=Path("/workspace/results"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/results/result.json"))
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

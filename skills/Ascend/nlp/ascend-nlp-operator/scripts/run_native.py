#!/usr/bin/env python3
"""Build and run canonical Ascend GEMM or Conv benchmarks."""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


MEASUREMENT_COLUMNS = ("baseline", "time", "score")
CONFIGURATIONS = {
    "gemm": {
        "columns": (
            "NO", "M", "N", "K", "transA", "transB",
            "baseline", "time", "score",
        ),
        "identity": ("M", "N", "K", "transA", "transB"),
        "positive": ("M", "N", "K"),
        "non_negative": (),
        "boolean": ("transA", "transB"),
        "expected_rows": 224,
        "driver": "test_gemm_native.py",
    },
    "conv": {
        "columns": (
            "NO", "W", "H", "C", "N", "OutC", "kw", "kh", "pw", "ph",
            "sh", "sv", "baseline", "time", "score",
        ),
        "identity": (
            "W", "H", "C", "N", "OutC", "kw", "kh", "pw", "ph", "sh", "sv",
        ),
        "positive": ("W", "H", "C", "N", "OutC", "kw", "kh", "sh", "sv"),
        "non_negative": ("pw", "ph"),
        "boolean": (),
        "expected_rows": 63,
        "driver": "test_conv_native.py",
    },
}

Identity = Tuple[str, ...]


def _empty(value: object) -> bool:
    return str(value or "").strip().lower() in {"", "nan", "none"}


def _integer(row: Dict[str, str], column: str, source: Path, line: int) -> int:
    raw = (row.get(column) or "").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{source}:{line}: {column} must be an integer") from exc
    return value


def read_template(source: Path, benchmark: str) -> Tuple[List[Dict[str, str]], List[Identity]]:
    configuration = CONFIGURATIONS[benchmark]
    expected_columns = tuple(configuration["columns"])
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != expected_columns:
            raise ValueError(
                f"{source}: columns must be exactly: {','.join(expected_columns)}"
            )
        rows: List[Dict[str, str]] = []
        identities: List[Identity] = []
        seen: set[Identity] = set()
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{source}:{line}: row contains unexpected fields")
            number = _integer(row, "NO", source, line)
            if number != len(rows):
                raise ValueError(f"{source}:{line}: NO must be contiguous from zero")
            for column in configuration["positive"]:
                if _integer(row, str(column), source, line) <= 0:
                    raise ValueError(f"{source}:{line}: {column} must be positive")
            for column in configuration["non_negative"]:
                if _integer(row, str(column), source, line) < 0:
                    raise ValueError(f"{source}:{line}: {column} must be non-negative")
            for column in configuration["boolean"]:
                if _integer(row, str(column), source, line) not in {0, 1}:
                    raise ValueError(f"{source}:{line}: {column} must be 0 or 1")
            for column in MEASUREMENT_COLUMNS:
                if not _empty(row.get(column)):
                    raise ValueError(f"{source}:{line}: {column} must be empty")
            identity = tuple((row.get(str(column)) or "").strip() for column in configuration["identity"])
            if identity in seen:
                raise ValueError(f"{source}:{line}: duplicate case identity: {identity}")
            seen.add(identity)
            identities.append(identity)
            rows.append(dict(row))
    expected_rows = int(configuration["expected_rows"])
    if len(rows) != expected_rows:
        raise ValueError(f"{source}: expected {expected_rows} rows, found {len(rows)}")
    return rows, identities


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_input(source: Path, destination: Path, benchmark: str) -> List[Identity]:
    rows, identities = read_template(source, benchmark)
    _write_csv(destination, CONFIGURATIONS[benchmark]["columns"], rows)
    return identities


def _positive(row: Dict[str, str], column: str, source: Path, line: int) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}:{line}: {column} must be numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{source}:{line}: {column} must be positive and finite")
    return value


def validate_output(source: Path, benchmark: str, expected: Sequence[Identity]) -> None:
    configuration = CONFIGURATIONS[benchmark]
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        fields = tuple(reader.fieldnames or ())
        required = set(configuration["columns"])
        if benchmark == "conv":
            required.update(("forward_ms", "backward_weight_ms", "backward_data_ms"))
        missing = required - set(fields)
        if missing:
            raise ValueError(f"{source}: missing columns: {', '.join(sorted(missing))}")
        if {"status", "error"} & set(fields):
            raise ValueError(f"{source}: status/error columns are not part of the result contract")
        actual: List[Identity] = []
        for line, row in enumerate(reader, start=2):
            if _integer(row, "NO", source, line) != len(actual):
                raise ValueError(f"{source}:{line}: NO must be contiguous from zero")
            identity = tuple((row.get(str(column)) or "").strip() for column in configuration["identity"])
            actual.append(identity)
            total = _positive(row, "baseline", source, line)
            for column in ("time", "score"):
                if not _empty(row.get(column)):
                    raise ValueError(f"{source}:{line}: {column} must remain empty")
            if benchmark == "conv":
                component_sum = sum(
                    _positive(row, column, source, line)
                    for column in ("forward_ms", "backward_weight_ms", "backward_data_ms")
                )
                if not math.isclose(total, component_sum, rel_tol=1e-5, abs_tol=5e-6):
                    raise ValueError(
                        f"{source}:{line}: baseline must equal forward + backward_weight + backward_data"
                    )
        if actual != list(expected):
            raise ValueError(f"{source}: case order or coverage differs from the input manifest")


def _run_and_log(command: Sequence[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"command exited {returncode}; see {log_path}")


def _check_binary(binary: Path) -> None:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"native benchmark binary is missing: {binary}")
    completed = subprocess.run(
        ("ldd", str(binary)), capture_output=True, text=True, check=False
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or "not found" in output:
        raise RuntimeError(f"native benchmark has unresolved libraries:\n{output}")


def _write_fingerprint(output_dir: Path, fingerprint: str) -> None:
    target = output_dir / "aibench_workload_fingerprint.txt"
    temporary = target.with_name(target.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(fingerprint + "\n", encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def run_benchmark(
    benchmark: str,
    operators_root: Path,
    output_dir: Path,
    log_dir: Path,
    device: int,
    warmup: int,
    iterations: int,
    build_dir: Path | None = None,
) -> None:
    fingerprint = os.environ.get("AIBENCH_WORKLOAD_FINGERPRINT", "").strip()
    if not fingerprint:
        raise ValueError("AIBENCH_WORKLOAD_FINGERPRINT is required")
    speed_root = (operators_root / "speed_test").resolve()
    configuration = CONFIGURATIONS[benchmark]
    build_dir = (build_dir or speed_root / "npu_ops" / "build-dev").resolve()
    build_script = speed_root / "npu_ops" / "build.sh"
    driver = speed_root / str(configuration["driver"])
    if not build_script.is_file() or not driver.is_file():
        raise ValueError(f"native {benchmark} project files are missing under {speed_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_file = output_dir / "aibench_workload_fingerprint.txt"
    fingerprint_file.unlink(missing_ok=True)
    for dtype in ("f16", "f32"):
        for suffix in ("_input.csv", ".csv"):
            (output_dir / f"{benchmark}_{dtype}{suffix}").unlink(missing_ok=True)

    _run_and_log(
        ("bash", str(build_script), str(build_dir), benchmark),
        speed_root,
        log_dir / "build.log",
    )
    binary = build_dir / benchmark
    _check_binary(binary)

    reference_identities: List[Identity] | None = None
    for bits in (16, 32):
        dtype = f"f{bits}"
        source = speed_root / f"{benchmark}_{dtype}.csv"
        manifest = output_dir / f"{benchmark}_{dtype}_input.csv"
        result = output_dir / f"{benchmark}_{dtype}.csv"
        identities = prepare_input(source, manifest, benchmark)
        if reference_identities is None:
            reference_identities = identities
        elif identities != reference_identities:
            raise ValueError(f"{benchmark} f16/f32 case identities differ")
        rows, _ = read_template(manifest, benchmark)
        _write_csv(result, configuration["columns"], rows)
        _run_and_log(
            (
                sys.executable,
                str(driver),
                str(result),
                str(bits),
                "0",
                "--device",
                str(device),
                "--binary",
                str(binary),
                "--warmup",
                str(warmup),
                "--iterations",
                str(iterations),
            ),
            speed_root,
            log_dir / f"{benchmark}_{dtype}.log",
        )
        validate_output(result, benchmark, identities)
    _write_fingerprint(output_dir, fingerprint)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=tuple(CONFIGURATIONS))
    parser.add_argument("--operators-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    if args.device < 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("device/warmup must be non-negative and iterations positive")
    run_benchmark(
        args.benchmark,
        args.operators_root,
        args.output_dir,
        args.log_dir,
        args.device,
        args.warmup,
        args.iterations,
        args.build_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

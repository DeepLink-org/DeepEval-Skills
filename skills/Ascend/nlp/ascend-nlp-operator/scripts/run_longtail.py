#!/usr/bin/env python3
"""Run canonical FP32 and FP16 Ascend LongTail benchmarks."""

from __future__ import annotations

import argparse
import csv
import math
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Identity = Tuple[int, str]
TEMPLATE_COLUMNS = ("NO", "op", "baseline", "time", "score")
MANIFEST_COLUMNS = TEMPLATE_COLUMNS + ("aibench_run_token",)


def _empty(value: object) -> bool:
    return str(value or "").strip().lower() in {"", "nan", "none"}


def prepare_input(source: Path, destination: Path, run_token: str) -> List[Identity]:
    if not run_token.strip() or len(run_token) > 128:
        raise ValueError("run_token must contain between 1 and 128 characters")
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        columns = tuple(reader.fieldnames or ())
        if columns != TEMPLATE_COLUMNS:
            raise ValueError(
                f"{source}: columns must be exactly: {','.join(TEMPLATE_COLUMNS)}"
            )
        rows = []
        identities: List[Identity] = []
        seen_numbers = set()
        seen_operators = set()
        for line, row in enumerate(reader, start=2):
            try:
                number = int(row["NO"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line}: NO must be an integer") from exc
            operator = (row.get("op") or "").strip()
            if number != len(rows) or not operator or len(operator) > 128:
                raise ValueError(f"{source}:{line}: invalid NO or op")
            if number in seen_numbers or operator in seen_operators:
                raise ValueError(f"{source}:{line}: duplicate LongTail case")
            for column in ("baseline", "time", "score"):
                if not _empty(row.get(column)):
                    raise ValueError(f"{source}:{line}: {column} must be empty")
            seen_numbers.add(number)
            seen_operators.add(operator)
            identities.append((number, operator))
            rows.append(
                {
                    "NO": number,
                    "op": operator,
                    "baseline": "",
                    "time": "",
                    "score": "",
                    "aibench_run_token": run_token,
                }
            )
    if not rows:
        raise ValueError(f"{source}: no LongTail cases")
    if len(rows) != 40:
        raise ValueError(f"{source}: expected 40 rows, found {len(rows)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(
                file_obj, fieldnames=MANIFEST_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return identities


def validate_output(source: Path, expected: Sequence[Identity], run_token: str) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = set(MANIFEST_COLUMNS)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing columns: {', '.join(sorted(missing))}")
        if {"status", "error"} & set(reader.fieldnames or []):
            raise ValueError(f"{source}: status/error columns are not part of the result contract")
        actual: List[Identity] = []
        for line, row in enumerate(reader, start=2):
            try:
                number = int(row["NO"])
                latency = float(row["baseline"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source}:{line}: NO and baseline must be numeric"
                ) from exc
            operator = (row.get("op") or "").strip()
            actual.append((number, operator))
            if row.get("aibench_run_token") != run_token:
                raise ValueError(f"{source}:{line}: stale aibench_run_token")
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"{source}:{line}: invalid measured baseline")
            for column in ("time", "score"):
                if not _empty(row.get(column)):
                    raise ValueError(f"{source}:{line}: {column} must remain empty")
    if actual != list(expected):
        raise ValueError(f"{source}: case order or coverage changed")


def _run_and_log(
    command: Sequence[str], cwd: Path, environment: Dict[str, str], log_file: Path
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
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
            log.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"LongTail exited {returncode}; see {log_file}")


def run_one(
    project: Path,
    source: Path,
    input_csv: Path,
    output_csv: Path,
    log_file: Path,
    run_token: str,
    device: int,
    warmup: int,
    iterations: int,
) -> List[Identity]:
    api = project / "long_tail_bench" / "api" / "api.py"
    if not api.is_file() or not source.is_file():
        raise ValueError(f"LongTail project or source CSV is missing: {project}, {source}")
    expected = prepare_input(source, input_csv, run_token)
    output_csv.unlink(missing_ok=True)
    temporary = output_csv.with_name(output_csv.name + ".tmp")
    temporary.unlink(missing_ok=True)
    result_json = project / "results" / "torch.json"
    result_json.unlink(missing_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project) + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    command = [
        sys.executable,
        str(api),
        "-f",
        str(input_csv),
        "--outcsv",
        str(temporary),
        "--device",
        str(device),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]
    try:
        _run_and_log(command, project, environment, log_file)
        if not result_json.is_file() or result_json.stat().st_size == 0:
            raise RuntimeError(f"LongTail did not create raw results: {result_json}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"LongTail did not create {temporary}")
        validate_output(temporary, expected, run_token)
        os.replace(temporary, output_csv)
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.device < 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("device/warmup must be non-negative and iterations positive")

    fingerprint = os.environ.get("AIBENCH_WORKLOAD_FINGERPRINT", "").strip()
    if not fingerprint:
        raise SystemExit("AIBENCH_WORKLOAD_FINGERPRINT is required")
    run_token = f"{fingerprint}:{secrets.token_hex(16)}"
    speed_root = args.operators_root / "speed_test"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    token_file = args.output_dir / "aibench_run_token.txt"
    token_file.unlink(missing_ok=True)

    configurations = (
        (
            "fp32",
            speed_root / "LongTail-Bench",
            speed_root / "longtail_f32.csv",
        ),
        (
            "fp16",
            speed_root / "LongTail-Bench-fp16",
            speed_root / "longtail_f16.csv",
        ),
    )
    reference_identities: List[Identity] | None = None
    for dtype, project, source in configurations:
        identities = run_one(
            project,
            source,
            args.output_dir / f"longtail_{dtype}_input.csv",
            args.output_dir / f"longtail_{dtype}.csv",
            args.log_dir / f"longtail_{dtype}.log",
            run_token,
            args.device,
            args.warmup,
            args.iterations,
        )
        if reference_identities is None:
            reference_identities = identities
        elif identities != reference_identities:
            raise ValueError("LongTail f16/f32 case identities differ")

    temporary = token_file.with_name(token_file.name + ".tmp")
    try:
        temporary.write_text(run_token + "\n", encoding="utf-8")
        os.replace(temporary, token_file)
    finally:
        temporary.unlink(missing_ok=True)
    for dtype, _, _ in configurations:
        print(f"LongTail {dtype}: measured=40, failed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

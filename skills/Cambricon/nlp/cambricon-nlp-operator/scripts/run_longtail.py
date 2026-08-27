#!/usr/bin/env python3
"""Generate MLU LongTail baselines from one run-scoped case manifest."""

from __future__ import annotations

import argparse
import csv
import math
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


STATIC_COLUMNS = ("NO", "op", "baseline")
MANIFEST_COLUMNS = (
    "NO",
    "op",
    "baseline",
    "time",
    "score",
    "inputshapes",
    "aibench_run_token",
)
CaseIdentity = Tuple[int, str]


def _required_file(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{description} does not exist: {resolved}")
    return resolved


def _required_nonempty_file(path: Path, description: str) -> Path:
    resolved = _required_file(path, description)
    if resolved.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {resolved}")
    return resolved


def _read_static_cases(source: Path) -> List[CaseIdentity]:
    with source.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        columns = tuple(reader.fieldnames or ())
        if columns != STATIC_COLUMNS:
            raise ValueError(
                f"{source}: columns must be exactly: {','.join(STATIC_COLUMNS)}"
            )

        cases: List[CaseIdentity] = []
        seen_numbers = set()
        seen_operators = set()
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{source}:{line}: row contains unexpected fields")
            try:
                number = int(row["NO"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source}:{line}: NO must be a non-negative integer"
                ) from exc
            operator = (row.get("op") or "").strip()
            if number < 0:
                raise ValueError(f"{source}:{line}: NO must be non-negative")
            if not operator or len(operator) > 128:
                raise ValueError(
                    f"{source}:{line}: op must contain between 1 and 128 characters"
                )
            if (row.get("baseline") or "").strip():
                raise ValueError(f"{source}:{line}: baseline must be empty")
            if number in seen_numbers:
                raise ValueError(f"{source}:{line}: duplicate NO: {number}")
            if operator in seen_operators:
                raise ValueError(f"{source}:{line}: duplicate op: {operator}")
            seen_numbers.add(number)
            seen_operators.add(operator)
            cases.append((number, operator))

    if not cases:
        raise ValueError(f"{source}: no LongTail cases")
    return cases


def _write_manifest(
    manifest: Path,
    cases: Sequence[CaseIdentity],
    run_token: str,
) -> None:
    if not run_token.strip() or len(run_token) > 128:
        raise ValueError("run_token must contain between 1 and 128 characters")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(manifest.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=MANIFEST_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            for number, operator in cases:
                writer.writerow(
                    {
                        "NO": number,
                        "op": operator,
                        "baseline": "",
                        "time": "",
                        "score": "",
                        "inputshapes": "",
                        "aibench_run_token": run_token,
                    }
                )
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_measured_output(
    output: Path,
    expected_cases: Sequence[CaseIdentity],
    run_token: str,
) -> None:
    with output.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{output}: missing columns: {', '.join(sorted(missing))}"
            )

        measured_cases: List[CaseIdentity] = []
        seen_numbers = set()
        seen_operators = set()
        for line, row in enumerate(reader, start=2):
            try:
                number = int(row["NO"])
                latency = float(row["baseline"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{output}:{line}: NO and baseline must be numeric"
                ) from exc
            operator = (row.get("op") or "").strip()
            if number < 0 or not operator:
                raise ValueError(f"{output}:{line}: invalid LongTail case identity")
            if number in seen_numbers or operator in seen_operators:
                raise ValueError(f"{output}:{line}: duplicate LongTail case")
            if row.get("aibench_run_token") != run_token:
                raise ValueError(f"{output}:{line}: stale aibench_run_token")
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(
                    f"{output}:{line}: baseline must be a positive finite number"
                )
            seen_numbers.add(number)
            seen_operators.add(operator)
            measured_cases.append((number, operator))

    if measured_cases != list(expected_cases):
        raise ValueError(f"{output}: case order or coverage differs from the manifest")


def _run_and_log(
    command: Sequence[str], cwd: Path, environment: Dict[str, str], log: Path
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as log_file:
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
            log_file.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"LongTail benchmark failed with exit code {returncode}")


def run_benchmark(
    project_root: Path,
    source: Path,
    output: Path,
    log: Path,
    manifest: Path,
    run_token: Optional[str] = None,
) -> None:
    project_root = project_root.resolve()
    _required_file(project_root / "long_tail_bench" / "api" / "api.py", "LongTail API")
    source = _required_nonempty_file(source, "LongTail case template")
    output = output.resolve()
    manifest = manifest.resolve()
    temporary = output.with_name(output.name + ".tmp")
    raw_result = project_root / "results" / "torch.json"

    if source in {output, manifest} or output == manifest:
        raise ValueError("source, output, and manifest must use different paths")
    cases = _read_static_cases(source)
    run_token = (run_token or secrets.token_hex(16)).strip()

    output.parent.mkdir(parents=True, exist_ok=True)
    raw_result.parent.mkdir(parents=True, exist_ok=True)
    for path in (output, temporary, manifest, manifest.with_name(manifest.name + ".tmp"), raw_result):
        path.unlink(missing_ok=True)
    _write_manifest(manifest, cases, run_token)

    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(project_root) + (
        f"{os.pathsep}{current_pythonpath}" if current_pythonpath else ""
    )

    try:
        _run_and_log(
            (
                sys.executable,
                "-m",
                "long_tail_bench.api.api",
                "-f",
                str(manifest),
                "--outcsv",
                str(temporary),
                "--store_input_shape",
            ),
            project_root,
            environment,
            log,
        )
        _required_nonempty_file(raw_result, "LongTail raw result")
        _required_nonempty_file(temporary, "LongTail measured CSV")
        _validate_measured_output(temporary, cases, run_token)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/workspace/operators/speed_test/LongTail-Bench_mlu"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/workspace/operators/speed_test/longtail_perf.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/results/longtail/longtail_result.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/workspace/results/longtail/longtail_cases_input.csv"),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("/workspace/logs/longtail/longtail.log"),
    )
    args = parser.parse_args()
    run_benchmark(
        args.project_root,
        args.source,
        args.output,
        args.log,
        args.manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a run-scoped LongTail case manifest with empty measurement fields."""

from __future__ import annotations

import argparse
import csv
import os
import secrets
from pathlib import Path
from typing import Optional


def prepare_input(source: Path, output: Path, run_token: Optional[str] = None) -> int:
    run_token = run_token or secrets.token_hex(16)
    if len(run_token) > 128:
        raise ValueError("run_token must contain at most 128 characters")
    with source.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        required = {"NO", "op", "baseline", "time", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing columns: {', '.join(sorted(missing))}")

        rows = []
        seen_numbers = set()
        seen_operators = set()
        for line, row in enumerate(reader, start=2):
            try:
                case_number = int(row["NO"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line}: NO must be a non-negative integer") from exc
            operator = (row.get("op") or "").strip()
            if case_number < 0:
                raise ValueError(f"{source}:{line}: NO must be a non-negative integer")
            if not operator:
                raise ValueError(f"{source}:{line}: op must be a non-empty string")
            if case_number in seen_numbers:
                raise ValueError(f"{source}:{line}: duplicate NO: {case_number}")
            if operator in seen_operators:
                raise ValueError(f"{source}:{line}: duplicate op: {operator}")
            seen_numbers.add(case_number)
            seen_operators.add(operator)
            rows.append(
                {
                    "NO": case_number,
                    "op": operator,
                    "baseline": "",
                    "time": "",
                    "score": "",
                    "aibench_run_token": run_token,
                }
            )

    if not rows:
        raise ValueError(f"{source}: no benchmark cases")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=(
                    "NO",
                    "op",
                    "baseline",
                    "time",
                    "score",
                    "aibench_run_token",
                ),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_input(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

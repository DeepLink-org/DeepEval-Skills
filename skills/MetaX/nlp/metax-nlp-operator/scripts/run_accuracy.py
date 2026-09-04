#!/usr/bin/env python3
"""Run MetaX accuracy validation while preserving valid failed measurements."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def regenerate_cpu_reference(project_root: Path, reference_dir: Path) -> None:
    """Generate a complete CPU reference before replacing the previous one."""
    if reference_dir.is_symlink():
        raise SystemExit(f"refusing to replace symlink reference: {reference_dir}")
    project_parent = project_root.resolve().parent
    reference_dir = reference_dir.resolve()
    if (
        reference_dir.parent != project_parent
        or reference_dir.name != "accuracy_reference_cpu"
    ):
        raise SystemExit(
            "regenerated CPU reference must be the accuracy_reference_cpu sibling "
            "of accuracy_test"
        )

    generator = project_root / "cpu_ground_truth_gen.py"
    if not generator.is_file():
        raise SystemExit(f"CPU reference generator not found: {generator}")

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".accuracy_reference_cpu.", dir=project_parent)
    )
    try:
        completed = subprocess.run(
            [sys.executable, str(generator), str(temporary_dir)],
            cwd=project_root,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"CPU reference generator exited {completed.returncode}"
            )
        if not (temporary_dir / "info.json").is_file():
            raise SystemExit("CPU reference generator did not create info.json")

        if reference_dir.exists():
            if not reference_dir.is_dir():
                raise SystemExit(f"reference path is not a directory: {reference_dir}")
            shutil.rmtree(reference_dir)
        os.replace(temporary_dir, reference_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--regenerate-cpu-reference",
        action="store_true",
        help="force a fresh CPU reference and replace the previous directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", help="optional operator name or CONV")
    args = parser.parse_args()

    validator = args.project_root / "cuda_op_validate.py"
    if not validator.is_file():
        raise SystemExit(f"validator not found: {validator}")
    if args.regenerate_cpu_reference:
        regenerate_cpu_reference(args.project_root, args.reference_dir)
    if not args.reference_dir.is_dir():
        raise SystemExit(f"reference directory not found: {args.reference_dir}")
    fingerprint = os.environ.get("AIBENCH_WORKLOAD_FINGERPRINT", "").strip()
    if not fingerprint:
        raise SystemExit("AIBENCH_WORKLOAD_FINGERPRINT is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_json = args.output_dir / "mx_val_result.json"
    result_csv = args.output_dir / "mx_val_result.csv"
    fingerprint_file = args.output_dir / "aibench_workload_fingerprint.txt"
    for path in (result_json, result_csv, fingerprint_file):
        path.unlink(missing_ok=True)

    command = [
        sys.executable,
        str(validator),
        str(args.reference_dir),
        str(args.output_dir),
    ]
    if args.case:
        command.append(args.case)
    command.extend(("--device", str(args.device)))
    completed = subprocess.run(command, cwd=args.project_root, check=False)
    if completed.returncode not in {0, 2}:
        raise SystemExit(f"accuracy validator exited {completed.returncode}")
    if not result_json.is_file() or not result_csv.is_file():
        raise SystemExit("accuracy validator did not create both JSON and CSV outputs")
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise SystemExit("accuracy JSON is empty or invalid")
    temporary = fingerprint_file.with_name(fingerprint_file.name + ".tmp")
    try:
        temporary.write_text(fingerprint + "\n", encoding="utf-8")
        os.replace(temporary, fingerprint_file)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"accuracy artifacts: {result_json}, {result_csv}; "
        f"validator_exit={completed.returncode}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

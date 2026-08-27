#!/usr/bin/env python3
"""Generate CPU ground truth and run MLU operator accuracy validation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


_BOOTSTRAP = r"""
import runpy
import sys
import torch
import torch_mlu  # noqa: F401
import passop_config

device_name = sys.argv[1]
script = sys.argv[2]
passop_config.device = torch.device(device_name)
sys.argv = [script, *sys.argv[3:]]
runpy.run_path(script, run_name="__main__")
"""


def _run_and_log(command: Sequence[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
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
        raise RuntimeError(f"accuracy command failed with exit code {returncode}")


def _validate_result(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("MLU accuracy result must be a non-empty object")
    for operator, result in payload.items():
        if not isinstance(operator, str) or not operator:
            raise ValueError("accuracy result contains an empty operator name")
        if not isinstance(result, dict) or type(result.get("passed_fp32")) is not bool:
            raise ValueError(f"{operator}: passed_fp32 must be boolean")


def run_accuracy(
    project_root: Path,
    result_dir: Path,
    log_dir: Path,
    ground_truth: Path | None = None,
) -> None:
    project_root = project_root.resolve()
    generator = project_root / "cpu_ground_truth_gen.py"
    validator = project_root / "mlu_op_validate.py"
    for path in (generator, validator, project_root / "passop_config.py"):
        if not path.is_file():
            raise ValueError(f"accuracy project is incomplete: {path}")

    result_dir = result_dir.resolve()
    log_dir = log_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if ground_truth is None:
        ground_truth = result_dir / "ground_truth"
        shutil.rmtree(ground_truth, ignore_errors=True)
        _run_and_log(
            (sys.executable, "-c", _BOOTSTRAP, "cpu", str(generator), str(ground_truth)),
            project_root,
            log_dir / "cpu_ground_truth.log",
        )
    else:
        ground_truth = ground_truth.resolve()
    if not (ground_truth / "info.json").is_file():
        raise ValueError(f"CPU ground truth is incomplete: {ground_truth}")

    result_json = result_dir / "mlu_val_result.json"
    result_csv = result_dir / "mlu_val_result.csv"
    result_json.unlink(missing_ok=True)
    result_csv.unlink(missing_ok=True)
    _run_and_log(
        (
            sys.executable,
            "-c",
            _BOOTSTRAP,
            "mlu",
            str(validator),
            str(ground_truth),
            str(result_dir),
        ),
        project_root,
        log_dir / "mlu_validate.log",
    )
    if not result_json.is_file() or not result_csv.is_file():
        raise ValueError("MLU accuracy validation did not produce both JSON and CSV")
    _validate_result(result_json)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/workspace/operators/accuracy_test"),
    )
    parser.add_argument(
        "--result-dir", type=Path, default=Path("/workspace/results/accuracy")
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("/workspace/logs/accuracy")
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Use an existing trusted CPU ground-truth directory instead of regenerating it.",
    )
    args = parser.parse_args()
    run_accuracy(args.project_root, args.result_dir, args.log_dir, args.ground_truth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

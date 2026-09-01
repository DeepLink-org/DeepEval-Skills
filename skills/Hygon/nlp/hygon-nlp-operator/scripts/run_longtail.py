#!/usr/bin/env python3
"""确定性运行 LongTail-Bench f32/f16，并原子发布本轮 CSV。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence


def _required_file(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{description} does not exist or is not a file: {resolved}")
    return resolved


def _required_nonempty_file(path: Path, description: str) -> Path:
    resolved = _required_file(path, description)
    if resolved.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {resolved}")
    return resolved


def _required_project(path: Path) -> Path:
    resolved = path.resolve()
    _required_file(resolved / "long_tail_bench" / "api" / "api.py", "LongTail API")
    return resolved


def _run_and_log(command: Sequence[str], cwd: Path, env: Dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
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
        raise RuntimeError(f"LongTail benchmark failed with exit code {returncode}: {cwd}")


def run_variant(
    project_root: Path,
    manifest: Path,
    output: Path,
    log: Path,
) -> None:
    project_root = _required_project(project_root)
    manifest = _required_nonempty_file(manifest, "LongTail manifest")
    result_dir = project_root / "results"
    raw_result = result_dir / "torch.json"
    temporary = output.with_name(output.name + ".tmp")

    result_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_result.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)

    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(project_root) + (
        f"{os.pathsep}{current_pythonpath}" if current_pythonpath else ""
    )

    try:
        _run_and_log(
            (
                sys.executable,
                "./long_tail_bench/api/api.py",
                "-f",
                str(manifest),
                "--outcsv",
                str(temporary),
            ),
            project_root,
            environment,
            log,
        )
        _required_nonempty_file(raw_result, "LongTail raw result")
        _required_nonempty_file(temporary, "LongTail output CSV")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--f32-project-root", type=Path, required=True)
    parser.add_argument("--f16-project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()

    run_variant(
        args.f32_project_root,
        args.manifest,
        args.output_dir / "longtail_perf_gpu.csv",
        args.log_dir / "longtail_gpu_baseline.log",
    )
    run_variant(
        args.f16_project_root,
        args.manifest,
        args.output_dir / "longtail_perf_gpu_fp16.csv",
        args.log_dir / "longtail_gpu_baseline_fp16.log",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

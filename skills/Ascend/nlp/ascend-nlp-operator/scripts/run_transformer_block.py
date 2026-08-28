#!/usr/bin/env python3
"""Run the Ascend Transformer Block benchmark and normalize its CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


FIELDNAMES = [
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
]


def normalize_results(payload: object, fingerprint: str) -> List[Dict[str, object]]:
    if not isinstance(payload, list):
        raise ValueError("transformer output must be a list")
    rows: List[Dict[str, object]] = []
    seen = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("transformer result items must be objects")
        block = item.get("block")
        if block not in {"encoder", "decoder"} or block in seen:
            raise ValueError("expected unique encoder and decoder results")
        if item.get("dtype") != "fp32" or item.get("mode") != "forward":
            raise ValueError("Transformer contract requires fp32 forward")
        if item.get("training") is not False:
            raise ValueError("Transformer contract requires eval/inference mode")
        latency = float(item["latency_ms"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("latency_ms must be positive and finite")
        query_length = int(item["sequence_length"])
        memory_length = item.get("memory_length")
        key_value_length = query_length if memory_length is None else int(memory_length)
        row = {
            "block_type": block,
            "dtype": "f32",
            "execution_mode": "inference",
            "d_model": int(item["d_model"]),
            "num_heads": int(item["heads"]),
            "ffn_hidden_size": int(item["ffn_hidden"]),
            "batch_size": int(item["batch_size"]),
            "query_sequence_length": query_length,
            "key_value_sequence_length": key_value_length,
            "warmup_iterations": int(item["warmup"]),
            "measurement_iterations": int(item["iterations"]),
            "latency_ms": latency,
            "aibench_workload_fingerprint": fingerprint,
        }
        for name in (
            "d_model",
            "num_heads",
            "ffn_hidden_size",
            "batch_size",
            "query_sequence_length",
            "key_value_sequence_length",
            "measurement_iterations",
        ):
            if row[name] <= 0:
                raise ValueError(f"{name} must be positive")
        if row["warmup_iterations"] < 0:
            raise ValueError("warmup_iterations must be non-negative")
        seen.add(block)
        rows.append(row)
    if seen != {"encoder", "decoder"}:
        raise ValueError("expected exactly one encoder and one decoder")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-hidden-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measurement-iterations", type=int, default=1000)
    args = parser.parse_args()
    benchmark = args.project_root / "test.py"
    if not benchmark.is_file():
        raise SystemExit(f"Transformer benchmark not found: {benchmark}")
    fingerprint = os.environ.get("AIBENCH_WORKLOAD_FINGERPRINT", "").strip()
    if not fingerprint:
        raise SystemExit("AIBENCH_WORKLOAD_FINGERPRINT is required")
    if args.d_model % args.num_heads:
        parser.error("--d-model must be divisible by --num-heads")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.output.with_name(args.output.stem + ".raw.json")
    for path in (args.output, raw_output):
        path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(benchmark),
        "--kind",
        "both",
        "--dtype",
        "fp32",
        "--device",
        str(args.device),
        "--d-model",
        str(args.d_model),
        "--heads",
        str(args.num_heads),
        "--ffn-hidden",
        str(args.ffn_hidden_size),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.sequence_length),
        "--memory-len",
        str(args.sequence_length),
        "--warmup",
        str(args.warmup_iterations),
        "--iterations",
        str(args.measurement_iterations),
        "--output",
        str(raw_output),
    ]
    completed = subprocess.run(command, cwd=args.project_root, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"Transformer benchmark exited {completed.returncode}")
    if not raw_output.is_file():
        raise SystemExit("Transformer benchmark did not create raw JSON")
    rows = normalize_results(
        json.loads(raw_output.read_text(encoding="utf-8")), fingerprint
    )

    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Transformer cases written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

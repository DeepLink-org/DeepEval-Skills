#!/usr/bin/env python3
"""确定性运行天数智芯 GPU Transformer encoder/decoder block inference benchmark。"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"missing required environment variable: {name}")
    return value


def _positive(name: str, value: int, minimum: int = 1) -> int:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _measure_latency_ms(
    operation: Callable[[], object],
    warmup_iterations: int,
    measurement_iterations: int,
    device: object,
) -> float:
    import torch

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            operation()
        torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        for _ in range(measurement_iterations):
            operation()
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started_at) * 1000.0 / measurement_iterations
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise RuntimeError("measured latency must be a non-negative finite number")
    return latency_ms


def run_benchmark(
    project_root: Path,
    *,
    d_model: int,
    num_heads: int,
    ffn_hidden_size: int,
    batch_size: int,
    sequence_length: int,
    warmup_iterations: int,
    measurement_iterations: int,
) -> List[Dict[str, object]]:
    import torch

    project_root = project_root.resolve()
    if not (project_root / "blocks" / "encoder_layer.py").is_file():
        raise ValueError(f"Transformer Block project is incomplete: {project_root}")
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Transformer Block benchmark")

    sys.path.insert(0, str(project_root))
    try:
        from blocks.decoder_layer import DecoderLayer
        from blocks.encoder_layer import EncoderLayer
    finally:
        sys.path.pop(0)

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    encoder = EncoderLayer(d_model, ffn_hidden_size, num_heads, drop_prob=0.1).to(
        device
    )
    encoder.eval()
    encoder_input = torch.rand(
        batch_size,
        sequence_length,
        d_model,
        device=device,
        dtype=torch.float32,
    )
    encoder_latency = _measure_latency_ms(
        lambda: encoder(encoder_input, None),
        warmup_iterations,
        measurement_iterations,
        device,
    )
    encoder = None
    encoder_input = None
    torch.cuda.empty_cache()

    decoder = DecoderLayer(d_model, ffn_hidden_size, num_heads, drop_prob=0.1).to(
        device
    )
    decoder.eval()
    target = torch.rand(
        batch_size,
        sequence_length,
        d_model,
        device=device,
        dtype=torch.float32,
    )
    memory = torch.rand(
        batch_size,
        sequence_length,
        d_model,
        device=device,
        dtype=torch.float32,
    )
    decoder_latency = _measure_latency_ms(
        lambda: decoder(target, memory, None, None),
        warmup_iterations,
        measurement_iterations,
        device,
    )

    common: Dict[str, object] = {
        "dtype": "f32",
        "execution_mode": "inference",
        "d_model": d_model,
        "num_heads": num_heads,
        "ffn_hidden_size": ffn_hidden_size,
        "batch_size": batch_size,
        "query_sequence_length": sequence_length,
        "key_value_sequence_length": sequence_length,
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "aibench_workload_fingerprint": _required_env("AIBENCH_WORKLOAD_FINGERPRINT"),
    }
    return [
        {"block_type": "encoder", **common, "latency_ms": encoder_latency},
        {"block_type": "decoder", **common, "latency_ms": decoder_latency},
    ]


def write_results(output: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
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
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/workspace/operators/transformer_block"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/results/transformer_block_cases.csv"),
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-hidden-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measurement-iterations", type=int, default=1000)
    args = parser.parse_args()

    for name in (
        "d_model",
        "num_heads",
        "ffn_hidden_size",
        "batch_size",
        "sequence_length",
        "measurement_iterations",
    ):
        _positive(name, getattr(args, name))
    _positive("warmup_iterations", args.warmup_iterations, minimum=0)

    rows = run_benchmark(
        args.project_root,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )
    write_results(args.output, rows)
    for row in rows:
        print(f"{row['block_type']} latency_ms={float(row['latency_ms']):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

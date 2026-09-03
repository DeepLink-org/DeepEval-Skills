#!/usr/bin/env python3
"""Deterministically benchmark MetaX Transformer encoder/decoder blocks."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List


FIELDS = ["block_type", "dtype", "execution_mode", "d_model", "num_heads",
          "ffn_hidden_size", "batch_size", "query_sequence_length",
          "key_value_sequence_length", "warmup_iterations",
          "measurement_iterations", "latency_ms",
          "aibench_workload_fingerprint"]
FIELDNAMES = FIELDS


def normalize_results(payload: object, fingerprint: str) -> List[Dict[str, object]]:
    """Normalize an already structured payload for compatibility with callers/tests."""
    if not isinstance(payload, list):
        raise ValueError("transformer output must be a list")
    rows = []
    for item in payload:
        if not isinstance(item, dict) or item.get("block") not in {"encoder", "decoder"}:
            raise ValueError("invalid transformer result item")
        block = item["block"]
        rows.append({"block_type": block, "dtype": "f32", "execution_mode": "inference",
                     "d_model": int(item["d_model"]), "num_heads": int(item["heads"]),
                     "ffn_hidden_size": int(item["ffn_hidden"]), "batch_size": int(item["batch_size"]),
                     "query_sequence_length": int(item["sequence_length"]),
                     "key_value_sequence_length": int(item.get("memory_length") or item["sequence_length"]),
                     "warmup_iterations": int(item["warmup"]), "measurement_iterations": int(item["iterations"]),
                     "latency_ms": float(item["latency_ms"]),
                     "aibench_workload_fingerprint": fingerprint})
    if {row["block_type"] for row in rows} != {"encoder", "decoder"}:
        raise ValueError("expected exactly one encoder and one decoder")
    return rows


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"missing required environment variable: {name}")
    return value


def _measure(operation: Callable[[], object], warmup: int, iterations: int, device: object) -> float:
    import torch
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize(device)
    value = (time.perf_counter() - start) * 1000.0 / iterations
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("measured latency must be positive and finite")
    return value


def run(project: Path, args: argparse.Namespace) -> List[Dict[str, object]]:
    import torch
    project = project.resolve()
    if not (project / "blocks" / "encoder_layer.py").is_file():
        raise ValueError(f"Transformer Block project is incomplete: {project}")
    if args.d_model % args.num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Transformer Block benchmark")
    sys.path.insert(0, str(project))
    try:
        from blocks.encoder_layer import EncoderLayer
        from blocks.decoder_layer import DecoderLayer
    finally:
        sys.path.pop(0)
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    encoder = EncoderLayer(args.d_model, args.ffn_hidden_size, args.num_heads, drop_prob=0.1).to(device)
    encoder.eval()
    enc_input = torch.rand(args.batch_size, args.sequence_length, args.d_model, device=device)
    enc_ms = _measure(lambda: encoder(enc_input, None), args.warmup_iterations, args.measurement_iterations, device)
    encoder = enc_input = None
    torch.cuda.empty_cache()
    decoder = DecoderLayer(args.d_model, args.ffn_hidden_size, args.num_heads, drop_prob=0.1).to(device)
    decoder.eval()
    target = torch.rand(args.batch_size, args.sequence_length, args.d_model, device=device)
    memory = torch.rand(args.batch_size, args.sequence_length, args.d_model, device=device)
    dec_ms = _measure(lambda: decoder(target, memory, None, None), args.warmup_iterations, args.measurement_iterations, device)
    common = {"dtype": "f32", "execution_mode": "inference", "d_model": args.d_model,
              "num_heads": args.num_heads, "ffn_hidden_size": args.ffn_hidden_size,
              "batch_size": args.batch_size, "query_sequence_length": args.sequence_length,
              "key_value_sequence_length": args.sequence_length,
              "warmup_iterations": args.warmup_iterations,
              "measurement_iterations": args.measurement_iterations,
              "aibench_workload_fingerprint": _env("AIBENCH_WORKLOAD_FINGERPRINT")}
    return [dict(common, block_type="encoder", latency_ms=enc_ms), dict(common, block_type="decoder", latency_ms=dec_ms)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/operators/speed_test/transformer_block"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/results/transformer/transformer_block_cases.csv"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-hidden-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measurement-iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.device < 0 or args.warmup_iterations < 0 or args.measurement_iterations <= 0:
        parser.error("invalid device, warmup, or measurement iterations")
    rows = run(args.project_root, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    for row in rows:
        print(f"{row['block_type']} latency_ms={float(row['latency_ms']):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

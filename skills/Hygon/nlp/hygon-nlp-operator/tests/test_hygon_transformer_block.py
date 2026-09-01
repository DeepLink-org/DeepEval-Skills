#!/usr/bin/env python3
"""Transformer Block BenchmarkSpec、runner 输出与 collector 回归测试。"""

import csv
import importlib.util
from pathlib import Path

import pytest
import yaml

SKILL_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = SKILL_ROOT / "benchmark_specs" / "transformer_block.yaml"
COLLECTOR_PATH = SKILL_ROOT / "scripts" / "collect_cases.py"
RUNNER_PATH = SKILL_ROOT / "scripts" / "run_transformer_block.py"
SKILL_PATH = SKILL_ROOT / "SKILL.md"
FINGERPRINT = "0123456789abcdef0123456789abcdef"
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


def _load_module(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


def _set_environment(monkeypatch, spec_sha256="a" * 64):
    values = {
        "AIBENCH_TASK_ID": "Hygon_nlp_operator",
        "AIBENCH_WORKLOAD_FINGERPRINT": FINGERPRINT,
        "AIBENCH_BENCHMARK_SPEC_ID": "operator.transformer_block",
        "AIBENCH_BENCHMARK_SPEC_VERSION": "1.0.0",
        "AIBENCH_BENCHMARK_CASE_SCHEMA_VERSION": "1",
        "AIBENCH_BENCHMARK_SPEC_SHA256": spec_sha256,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _row(block_type, latency_ms, fingerprint=FINGERPRINT):
    return {
        "block_type": block_type,
        "dtype": "f32",
        "execution_mode": "inference",
        "d_model": 512,
        "num_heads": 8,
        "ffn_hidden_size": 2048,
        "batch_size": 32,
        "query_sequence_length": 512,
        "key_value_sequence_length": 512,
        "warmup_iterations": 20,
        "measurement_iterations": 1000,
        "latency_ms": latency_ms,
        "aibench_workload_fingerprint": fingerprint,
    }


def _write_cases(directory, rows):
    output = directory / "transformer_block_cases.csv"
    with output.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output


def test_skill_binds_transformer_block_spec_and_deterministic_scripts():
    skill = SKILL_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))

    assert "transformer_block: benchmark_specs/transformer_block.yaml" in skill
    assert "/workspace/scripts/run_transformer_block.py" in skill
    assert "--benchmark transformer_block" in skill
    assert "生成 2 条 cases" in skill
    assert "torch.inference_mode()" in runner
    assert "torch.cuda.synchronize(device)" in runner
    assert spec["benchmark"] == {
        "spec_id": "operator.transformer_block",
        "spec_version": "1.0.0",
        "case_schema_version": "1",
        "display_name": "Transformer Block",
    }


def test_collector_builds_two_complete_cases(tmp_path, monkeypatch):
    _write_cases(tmp_path, [_row("encoder", 1.0), _row("decoder", 3.0)])
    _set_environment(monkeypatch)

    payload = _load_module("transformer_collector", COLLECTOR_PATH).build_result(
        tmp_path,
        2.0,
        benchmark="transformer_block",
    )

    assert payload["schema_version"] == "2.0"
    assert payload["status"] == "success"
    assert [case["dimensions"]["block_type"] for case in payload["cases"]] == [
        "encoder",
        "decoder",
    ]
    assert payload["metrics"] == {
        "transformer_block_total_cases": 2,
        "transformer_block_success_cases": 2,
        "transformer_block_failed_cases": 0,
        "transformer_block_latency_avg_ms": 2.0,
        "transformer_block_latency_p50_ms": 2.0,
        "transformer_block_latency_p95_ms": pytest.approx(2.9),
        "transformer_block_latency_min_ms": 1.0,
        "transformer_block_latency_max_ms": 3.0,
    }
    assert payload["metadata"] == {
        "workload_fingerprint": FINGERPRINT,
        "measurement_count": 2,
        "duration_seconds": 2.0,
        "source": "transformer_block_cases.csv measured latency_ms",
    }


def test_collector_output_passes_aibench_result_contract(tmp_path, monkeypatch):
    benchmark_spec = pytest.importorskip("agent.benchmark_spec")
    result_contract = pytest.importorskip("agent.result_contract")
    spec = benchmark_spec.load_benchmark_spec(SPEC_PATH)
    _set_environment(monkeypatch, spec.spec_sha256)
    _write_cases(tmp_path, [_row("encoder", 1.0), _row("decoder", 3.0)])

    payload = _load_module(
        "transformer_contract_collector", COLLECTOR_PATH
    ).build_result(
        tmp_path,
        2.0,
        benchmark="transformer_block",
    )
    result = result_contract.validate_result_payload(
        payload,
        "Hygon_nlp_operator",
        expected_workload_fingerprint=FINGERPRINT,
        expected_benchmark_spec=spec,
    )

    assert len(result.cases) == 2
    assert len({case.case_key for case in result.cases}) == 2
    assert result.benchmark.spec_id == "operator.transformer_block"


def test_existing_aibench_registry_discovers_transformer_block_capability():
    skill_registry = pytest.importorskip("skills.skill_registry")
    registry = skill_registry.SkillRegistry()

    skill = registry.get_benchmark_skill(
        "Hygon",
        "nlp",
        "operator",
        "transformer_block",
    )
    capabilities = registry.get_benchmark_capabilities()

    assert skill.benchmark_spec.spec_id == "operator.transformer_block"
    assert any(
        item["chip_type"] == "Hygon"
        and item["application_scenario"] == "nlp"
        and item["task_type"] == "operator"
        and item["test_case"] == "transformer_block"
        and item["benchmark_spec"]["spec_id"] == "operator.transformer_block"
        for item in capabilities
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_row("encoder", 1.0)], "exactly one encoder and one decoder"),
        (
            [_row("encoder", 1.0), _row("encoder", 2.0)],
            "duplicate block_type",
        ),
        (
            [_row("encoder", 1.0, "old"), _row("decoder", 2.0, "old")],
            "does not belong to the active workload",
        ),
        ([_row("encoder", "nan"), _row("decoder", 2.0)], "non-negative finite"),
        ([_row("encoder", -1.0), _row("decoder", 2.0)], "non-negative finite"),
    ],
)
def test_collector_rejects_incomplete_stale_or_invalid_results(
    tmp_path,
    monkeypatch,
    rows,
    message,
):
    _write_cases(tmp_path, rows)
    _set_environment(monkeypatch)

    with pytest.raises(ValueError, match=message):
        _load_module("invalid_transformer_collector", COLLECTOR_PATH).build_result(
            tmp_path,
            1.0,
            benchmark="transformer_block",
        )


def test_runner_writes_atomic_structured_csv(tmp_path):
    runner = _load_module("transformer_runner", RUNNER_PATH)
    output = tmp_path / "transformer_block_cases.csv"
    rows = [_row("encoder", 1.0), _row("decoder", 3.0)]

    runner.write_results(output, rows)

    with output.open("r", encoding="utf-8", newline="") as file_obj:
        written = list(csv.DictReader(file_obj))
    assert [row["block_type"] for row in written] == ["encoder", "decoder"]
    assert not output.with_name(output.name + ".tmp").exists()

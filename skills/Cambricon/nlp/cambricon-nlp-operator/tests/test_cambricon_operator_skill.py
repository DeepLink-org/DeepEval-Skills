#!/usr/bin/env python3
"""Cambricon operator Skill structure, collector, and contract regression tests."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = SKILL_ROOT / "benchmark_specs"
SCRIPT_ROOT = SKILL_ROOT / "scripts"
FINGERPRINT = "0123456789abcdef0123456789abcdef"


def _load_module(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


COLLECTOR = _load_module("cambricon_collect_cases", SCRIPT_ROOT / "collect_cases.py")
TRANSFORMER_RUNNER = _load_module(
    "cambricon_transformer_runner", SCRIPT_ROOT / "run_transformer_block.py"
)
LONGTAIL_RUNNER = _load_module(
    "cambricon_longtail_runner", SCRIPT_ROOT / "run_longtail.py"
)


def _frontmatter() -> dict:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    return yaml.safe_load(text.split("---", 2)[1])


def _set_environment(monkeypatch, spec) -> None:
    values = {
        "AIBENCH_TASK_ID": "Cambricon_nlp_operator",
        "AIBENCH_WORKLOAD_FINGERPRINT": FINGERPRINT,
        "AIBENCH_BENCHMARK_SPEC_ID": spec.spec_id,
        "AIBENCH_BENCHMARK_SPEC_VERSION": spec.spec_version,
        "AIBENCH_BENCHMARK_CASE_SCHEMA_VERSION": spec.case_schema_version,
        "AIBENCH_BENCHMARK_SPEC_SHA256": spec.spec_sha256,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_spec(name: str):
    benchmark_spec = pytest.importorskip("agent.benchmark_spec")
    return benchmark_spec.load_benchmark_spec(SPEC_ROOT / f"{name}.yaml")


def _validate_contract(payload: dict, spec):
    result_contract = pytest.importorskip("agent.result_contract")
    return result_contract.validate_result_payload(
        payload,
        "Cambricon_nlp_operator",
        expected_workload_fingerprint=FINGERPRINT,
        expected_benchmark_spec=spec,
    )


def test_skill_maps_each_supported_test_case_to_a_spec():
    benchmark_spec = pytest.importorskip("agent.benchmark_spec")
    frontmatter = _frontmatter()
    expected = {
        "accuracy": "operator.accuracy_validation",
        "gemm": "operator.gemm",
        "conv": "operator.conv2d",
        "convbackdata": "operator.conv2d_backward_component",
        "convbackfilter": "operator.conv2d_backward_component",
        "longtail": "operator.longtail",
        "transformer": "operator.transformer_block",
        "transformer_block": "operator.transformer_block",
    }
    for test_case, spec_id in expected.items():
        path = benchmark_spec.resolve_benchmark_spec_path(
            SKILL_ROOT, frontmatter, test_case
        )
        assert path is not None
        assert benchmark_spec.load_benchmark_spec(path).spec_id == spec_id
    assert (
        benchmark_spec.resolve_benchmark_spec_path(SKILL_ROOT, frontmatter, "all")
        is None
    )


def test_transformer_spec_matches_nvidia_case_contract():
    nvidia_spec = (
        SKILL_ROOT.parents[2]
        / "NVIDIA/nlp/nvidia-nlp-operator/benchmark_specs/transformer_block.yaml"
    )
    assert yaml.safe_load((SPEC_ROOT / "transformer_block.yaml").read_text()) == yaml.safe_load(
        nvidia_spec.read_text()
    )


def test_conv_spec_matches_nvidia_case_contract():
    nvidia_spec = (
        SKILL_ROOT.parents[2]
        / "NVIDIA/nlp/nvidia-nlp-operator/benchmark_specs/conv.yaml"
    )
    cambricon = yaml.safe_load((SPEC_ROOT / "conv.yaml").read_text())
    nvidia = yaml.safe_load(nvidia_spec.read_text())

    assert cambricon["benchmark"]["spec_id"] == nvidia["benchmark"]["spec_id"]
    assert cambricon["benchmark"]["case_schema_version"] == nvidia["benchmark"]["case_schema_version"]
    assert cambricon["case"] == nvidia["case"]


def test_longtail_spec_matches_nvidia_contract():
    nvidia_spec = (
        SKILL_ROOT.parents[2]
        / "NVIDIA/nlp/nvidia-nlp-operator/benchmark_specs/longtail.yaml"
    )
    assert yaml.safe_load((SPEC_ROOT / "longtail.yaml").read_text()) == yaml.safe_load(
        nvidia_spec.read_text()
    )


@pytest.mark.parametrize("benchmark", ["gemm", "conv", "longtail"])
def test_shared_performance_summary_metric_names_match_nvidia(benchmark):
    nvidia_spec = (
        SKILL_ROOT.parents[2]
        / f"NVIDIA/nlp/nvidia-nlp-operator/benchmark_specs/{benchmark}.yaml"
    )
    cambricon = yaml.safe_load((SPEC_ROOT / f"{benchmark}.yaml").read_text())
    nvidia = yaml.safe_load(nvidia_spec.read_text())

    assert [item["name"] for item in cambricon["summary_metrics"]] == [
        item["name"] for item in nvidia["summary_metrics"]
    ]


def test_gemm_collector_uses_generated_baseline_and_passes_contract(
    tmp_path, monkeypatch
):
    fields = ["M", "N", "K", "transA", "transB", "i_d", "o_d", "baseline"]
    _write_csv(
        tmp_path / "gemm/gemm_FP16_result.csv",
        fields,
        [{
            "M": 8, "N": 16, "K": 32, "transA": 0, "transB": 0,
            "i_d": 0, "o_d": 0, "baseline": 0.005,
        }],
    )
    _write_csv(
        tmp_path / "gemm/gemm_FP32_result.csv",
        fields,
        [{
            "M": 8, "N": 16, "K": 32, "transA": 0, "transB": 0,
            "i_d": 1, "o_d": 1, "baseline": 0.01,
        }],
    )
    spec = _load_spec("gemm")
    _set_environment(monkeypatch, spec)

    payload = COLLECTOR.build_result(tmp_path, 2.0, "gemm")
    result = _validate_contract(payload, spec)

    assert [case.metrics["latency_ms"] for case in result.cases] == [0.005, 0.01]
    assert payload["metrics"]["gemm_baseline_avg_ms"] == pytest.approx(0.0075)
    assert "gemm_latency_avg_ms" not in payload["metrics"]
    assert "metadata" not in payload["cases"][0]


def _conv_row(dtype: str) -> dict:
    return {
        "W": 224,
        "H": 224,
        "C": 3,
        "N": 8,
        "OutC": 64,
        "kw": 3,
        "kh": 3,
        "pw": 1,
        "ph": 1,
        "sh": 1,
        "sw": 1,
        "baseline": 0.6,
    }


def test_conv_collector_uses_generated_baseline_and_passes_contract(
    tmp_path, monkeypatch
):
    fields = list(_conv_row("FP16"))
    for dtype in ("FP16", "FP32"):
        _write_csv(
            tmp_path / f"conv/conv_total_{dtype}_result.csv",
            fields,
            [_conv_row(dtype)],
        )
    spec = _load_spec("conv")
    _set_environment(monkeypatch, spec)

    payload = COLLECTOR.build_result(tmp_path, 3.0, "conv")
    result = _validate_contract(payload, spec)

    assert len(result.cases) == 2
    assert result.cases[0].metrics == {"latency_ms": 0.6}
    assert result.cases[0].dimensions["stride_horizontal"] == 1
    assert result.cases[0].dimensions["stride_vertical"] == 1
    assert payload["metrics"]["conv_baseline_avg_ms"] == pytest.approx(0.6)
    assert "conv_latency_avg_ms" not in payload["metrics"]
    assert "metadata" not in payload["cases"][0]


def test_conv_collector_rejects_empty_generated_baseline(tmp_path, monkeypatch):
    fields = list(_conv_row("FP16"))
    row = _conv_row("FP16")
    row["baseline"] = ""
    _write_csv(tmp_path / "conv/conv_total_FP16_result.csv", fields, [row])
    _write_csv(
        tmp_path / "conv/conv_total_FP32_result.csv",
        fields,
        [_conv_row("FP32")],
    )
    spec = _load_spec("conv")
    _set_environment(monkeypatch, spec)

    with pytest.raises(ValueError, match="baseline must be a number"):
        COLLECTOR.build_result(tmp_path, 1.0, "conv")


def test_conv_component_collector_tracks_phase(tmp_path, monkeypatch):
    fields = [
        "W", "H", "C", "N", "OutC", "kw", "kh", "pw", "ph", "sh", "sw",
        "baseline",
    ]
    base = {
        "W": 224, "H": 224, "C": 3, "N": 8, "OutC": 64, "kw": 3, "kh": 3,
        "pw": 1, "ph": 1, "sh": 1, "sw": 1, "baseline": 0.4,
    }
    _write_csv(
        tmp_path / "conv/convbk_data_FP16_result.csv", fields, [base]
    )
    _write_csv(
        tmp_path / "conv/convbk_data_FP32_result.csv", fields, [base]
    )
    spec = _load_spec("conv_component")
    _set_environment(monkeypatch, spec)

    payload = COLLECTOR.build_result(tmp_path, 1.0, "convbackdata")
    result = _validate_contract(payload, spec)
    assert {case.dimensions["phase"] for case in result.cases} == {"backward_data"}
    assert {case.metrics["latency_ms"] for case in result.cases} == {0.4}
    assert "metadata" not in payload["cases"][0]


def test_accuracy_collector_only_accepts_real_fp32_ground_truth(tmp_path, monkeypatch):
    path = tmp_path / "accuracy/mlu_val_result.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "conv2d": {"passed_fp32": False, "passed_fp16": True},
                "relu": {"passed_fp32": True, "passed_fp16": True},
            }
        ),
        encoding="utf-8",
    )
    spec = _load_spec("accuracy")
    _set_environment(monkeypatch, spec)

    payload = COLLECTOR.build_result(tmp_path, 1.0, "accuracy")
    result = _validate_contract(payload, spec)

    assert len(result.cases) == 2
    assert {case.dimensions["dtype"] for case in result.cases} == {"f32"}
    assert payload["metrics"] == {
        "accuracy_total_cases": 2,
        "accuracy_passed_cases": 1.0,
        "accuracy_failed_cases": 1.0,
        "accuracy_pass_rate": 0.5,
    }


def _longtail_runtime_row(token: str, baseline: object) -> dict:
    return {
        "NO": 0,
        "op": "bbox2delta",
        "baseline": baseline,
        "time": "",
        "score": "",
        "inputshapes": "[[128, 4], [128, 4]]",
        "aibench_run_token": token,
    }


def test_longtail_runner_generates_baseline_from_three_column_template(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "LongTail-Bench_mlu"
    api = project_root / "long_tail_bench/api/api.py"
    api.parent.mkdir(parents=True)
    api.write_text("# synthetic API\n", encoding="utf-8")
    source = tmp_path / "longtail_perf.csv"
    _write_csv(source, ["NO", "op", "baseline"], [{
        "NO": 0, "op": "bbox2delta", "baseline": "",
    }])
    output = tmp_path / "results/longtail_result.csv"
    manifest = tmp_path / "results/longtail_cases_input.csv"
    log = tmp_path / "logs/longtail.log"

    def fake_run(command, cwd, environment, log_path):
        assert "--validate" not in command
        assert "--store_input_shape" in command
        assert cwd == project_root.resolve()
        assert log_path == log
        with manifest.open("r", encoding="utf-8", newline="") as file_obj:
            rows = list(csv.DictReader(file_obj))
        assert rows[0]["baseline"] == ""
        assert rows[0]["aibench_run_token"] == "run-token"
        measured = {**rows[0], "baseline": 0.25, "inputshapes": "[[128, 4]]"}
        temporary = Path(command[command.index("--outcsv") + 1])
        _write_csv(temporary, list(measured), [measured])
        raw_result = project_root / "results/torch.json"
        raw_result.parent.mkdir(parents=True, exist_ok=True)
        raw_result.write_text('{"bbox2delta":{"time":0.25}}', encoding="utf-8")

    monkeypatch.setattr(LONGTAIL_RUNNER, "_run_and_log", fake_run)
    LONGTAIL_RUNNER.run_benchmark(
        project_root,
        source,
        output,
        log,
        manifest,
        run_token="run-token",
    )

    with output.open("r", encoding="utf-8", newline="") as file_obj:
        result_rows = list(csv.DictReader(file_obj))
    assert result_rows[0]["baseline"] == "0.25"
    assert manifest.is_file()
    assert not output.with_name(output.name + ".tmp").exists()


def test_longtail_runner_rejects_nonempty_template_baseline(tmp_path):
    source = tmp_path / "longtail_perf.csv"
    _write_csv(source, ["NO", "op", "baseline"], [{
        "NO": 0, "op": "bbox2delta", "baseline": 0.25,
    }])
    with pytest.raises(ValueError, match="baseline must be empty"):
        LONGTAIL_RUNNER._read_static_cases(source)


def test_longtail_collector_uses_generated_baseline(tmp_path, monkeypatch):
    token = "run-token"
    manifest_row = _longtail_runtime_row(token, "")
    manifest_row["inputshapes"] = ""
    _write_csv(
        tmp_path / "longtail/longtail_cases_input.csv",
        list(manifest_row),
        [manifest_row],
    )
    _write_csv(
        tmp_path / "longtail/longtail_result.csv",
        list(manifest_row),
        [_longtail_runtime_row(token, 0.5)],
    )
    spec = _load_spec("longtail")
    _set_environment(monkeypatch, spec)

    payload = COLLECTOR.build_result(tmp_path, 1.0, "longtail")
    result = _validate_contract(payload, spec)
    assert result.cases[0].metrics["latency_ms"] == 0.5
    assert payload["metrics"]["longtail_baseline_avg_ms"] == pytest.approx(0.5)
    assert "longtail_latency_avg_ms" not in payload["metrics"]
    assert "metadata" not in payload["cases"][0]


def _transformer_row(block_type: str, latency: float, fingerprint=FINGERPRINT) -> dict:
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
        "latency_ms": latency,
        "aibench_workload_fingerprint": fingerprint,
    }


def test_transformer_runner_output_and_collector_pass_contract(tmp_path, monkeypatch):
    output = tmp_path / "transformer/transformer_block_cases.csv"
    rows = [_transformer_row("encoder", 1.0), _transformer_row("decoder", 3.0)]
    TRANSFORMER_RUNNER.write_results(output, rows)
    spec = _load_spec("transformer_block")
    _set_environment(monkeypatch, spec)

    payload = COLLECTOR.build_result(tmp_path, 2.0, "transformer_block")
    result = _validate_contract(payload, spec)

    assert len(result.cases) == 2
    assert payload["metrics"]["transformer_block_latency_p50_ms"] == 2.0
    assert not output.with_name(output.name + ".tmp").exists()


def test_deterministic_scripts_encode_mlu_specific_invariants():
    transformer = (SCRIPT_ROOT / "run_transformer_block.py").read_text(encoding="utf-8")
    accuracy = (SCRIPT_ROOT / "run_accuracy.py").read_text(encoding="utf-8")
    longtail = (SCRIPT_ROOT / "run_longtail.py").read_text(encoding="utf-8")
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "torch.inference_mode()" in transformer
    assert "torch.mlu.synchronize()" in transformer
    assert ".eval()" in transformer
    assert 'passop_config.device = torch.device(device_name)' in accuracy
    assert '"--validate"' not in longtail
    assert '"--store_input_shape"' in longtail
    assert "aibench_run_token" in longtail
    assert list(SCRIPT_ROOT.glob("*.sh")) == []
    assert {path.name for path in SCRIPT_ROOT.glob("*.py")} == {
        "collect_cases.py",
        "run_accuracy.py",
        "run_longtail.py",
        "run_transformer_block.py",
    }
    assert "make -B -C /workspace/operators/speed_test/mlu_ops/gemm_sample" in skill
    assert 'source_csv="conv_${dtype}.csv"' in skill
    assert "mlu_test_conv_total.py" in skill
    assert '--component "$CONV_COMPONENT"' in skill
    assert "python3 /workspace/scripts/collect_cases.py" in skill

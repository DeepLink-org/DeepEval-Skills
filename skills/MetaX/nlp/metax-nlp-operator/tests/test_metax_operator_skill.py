from __future__ import annotations

import csv
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SKILL_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SKILL_ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


collector = load_module("metax_collect_cases", "collect_cases.py")
accuracy_runner = load_module("metax_run_accuracy", "run_accuracy.py")
longtail_runner = load_module("metax_run_longtail", "run_longtail.py")
transformer_runner = load_module("metax_run_transformer", "run_transformer_block.py")


ENVIRONMENT = {
    "AIBENCH_TASK_ID": "task-1",
    "AIBENCH_WORKLOAD_FINGERPRINT": "fingerprint-1",
    "AIBENCH_BENCHMARK_SPEC_ID": "operator.test",
    "AIBENCH_BENCHMARK_SPEC_VERSION": "1.0.0",
    "AIBENCH_BENCHMARK_CASE_SCHEMA_VERSION": "1",
    "AIBENCH_BENCHMARK_SPEC_SHA256": "0" * 64,
}


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_fingerprint(root: Path, benchmark: str) -> None:
    directory = root / benchmark
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "aibench_workload_fingerprint.txt").write_text(
        ENVIRONMENT["AIBENCH_WORKLOAD_FINGERPRINT"] + "\n", encoding="utf-8"
    )


def gemm_rows(latency: float):
    return [
        {
            "NO": number,
            "M": number + 1,
            "N": 16,
            "K": 32,
            "transA": 0,
            "transB": 1,
            "baseline": latency,
            "time": "",
            "score": "",
        }
        for number in range(224)
    ]


def conv_rows(total: float = 0.6):
    return [
        {
            "NO": number,
            "W": number + 1,
            "H": 7,
            "C": 16,
            "N": 8,
            "OutC": 32,
            "kw": 3,
            "kh": 3,
            "pw": 1,
            "ph": 1,
            "sh": 1,
            "sv": 1,
            "baseline": total,
            "time": "",
            "score": "",
            "forward_ms": 0.1,
            "backward_weight_ms": 0.2,
            "backward_data_ms": 0.3,
        }
        for number in range(63)
    ]


def longtail_rows(token: str, latency: float = 0.2):
    return [
        {
            "NO": number,
            "op": f"operator_{number}",
            "baseline": latency,
            "time": "",
            "score": "",
            "aibench_run_token": token,
        }
        for number in range(40)
    ]


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environment = mock.patch.dict(os.environ, ENVIRONMENT, clear=False)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_gemm_reads_current_metax_baseline(self):
        fields = [
            "NO", "M", "N", "K", "transA", "transB", "baseline", "time", "score"
        ]
        write_fingerprint(self.root, "gemm")
        for dtype, latency in (("f16", "2.5"), ("f32", "3.5")):
            write_csv(
                self.root / "gemm" / f"gemm_{dtype}.csv",
                fields,
                gemm_rows(float(latency)),
            )
        result = collector.build_result(self.root, 1.25, "gemm")
        self.assertEqual(
            [
                result["cases"][0]["metrics"]["latency_ms"],
                result["cases"][224]["metrics"]["latency_ms"],
            ],
            [2.5, 3.5],
        )
        self.assertEqual(result["metrics"]["gemm_total_cases"], 448)
        self.assertEqual(result["metrics"]["gemm_failed_cases"], 0)
        self.assertEqual(result["metadata"]["duration_seconds"], 1.25)

    def test_conv_reads_latency_baseline(self):
        fields = [
            "NO", "W", "H", "C", "N", "OutC", "kw", "kh", "pw", "ph",
            "sh", "sv", "baseline", "time", "score", "forward_ms",
            "backward_weight_ms", "backward_data_ms",
        ]
        write_fingerprint(self.root, "conv")
        for dtype in ("f16", "f32"):
            write_csv(
                self.root / "conv" / f"conv_{dtype}.csv", fields, conv_rows()
            )
        result = collector.build_result(self.root, 2.0, "conv")
        self.assertEqual(result["metrics"]["conv_success_cases"], 126)
        self.assertEqual(result["metrics"]["conv_baseline_avg_ms"], 0.6)

    def test_longtail_requires_complete_current_baselines(self):
        token = ENVIRONMENT["AIBENCH_WORKLOAD_FINGERPRINT"] + ":abc"
        directory = self.root / "longtail"
        directory.mkdir()
        (directory / "aibench_run_token.txt").write_text(token + "\n", encoding="utf-8")
        fields = ["NO", "op", "baseline", "time", "score", "aibench_run_token"]
        rows = longtail_rows(token)
        write_csv(directory / "longtail_fp32.csv", fields, rows)
        write_csv(directory / "longtail_fp16.csv", fields, rows)
        result = collector.build_result(self.root, 3.0, "longtail")
        self.assertEqual(result["metrics"]["longtail_total_cases"], 80)
        self.assertEqual(result["metrics"]["longtail_success_cases"], 80)
        self.assertEqual(result["metrics"]["longtail_failed_cases"], 0)
        rows[0]["baseline"] = ""
        write_csv(directory / "longtail_fp16.csv", fields, rows)
        with self.assertRaisesRegex(ValueError, "baseline"):
            collector.build_result(self.root, 3.0, "longtail")

    def test_accuracy_collects_each_available_dtype(self):
        directory = self.root / "accuracy"
        directory.mkdir()
        (directory / "aibench_workload_fingerprint.txt").write_text(
            ENVIRONMENT["AIBENCH_WORKLOAD_FINGERPRINT"] + "\n", encoding="utf-8"
        )
        payload = {
            "conv2d": {"fp32": True, "fp16": False, "passed": False, "errors": {}},
            "ctc_loss": {"fp32": None, "fp16": None, "passed": False, "errors": {}},
        }
        (directory / "mx_val_result.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        result = collector.build_result(self.root, 1.0, "accuracy")
        self.assertEqual(result["metrics"]["accuracy_total_cases"], 2)
        self.assertEqual(result["metrics"]["accuracy_passed_cases"], 1.0)
        self.assertEqual(result["metrics"]["accuracy_pass_rate"], 0.5)

    def test_transformer_requires_fingerprint_and_both_blocks(self):
        fields = transformer_runner.FIELDNAMES
        common = {
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
            "aibench_workload_fingerprint": ENVIRONMENT[
                "AIBENCH_WORKLOAD_FINGERPRINT"
            ],
        }
        rows = [dict(common, block_type="encoder", latency_ms=1.0),
                dict(common, block_type="decoder", latency_ms=2.0)]
        write_csv(
            self.root / "transformer" / "transformer_block_cases.csv", fields, rows
        )
        result = collector.build_result(self.root, 4.0, "transformer_block")
        self.assertEqual(result["metrics"]["transformer_block_total_cases"], 2)


class RunnerHelperTests(unittest.TestCase):
    def test_accuracy_force_replaces_cpu_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            operators = Path(temporary) / "operators"
            project = operators / "accuracy_test"
            reference = operators / "accuracy_reference_cpu"
            project.mkdir(parents=True)
            reference.mkdir()
            (project / "cpu_ground_truth_gen.py").touch()
            (reference / "old_marker.txt").write_text("old", encoding="utf-8")

            def fake_run(command, **_kwargs):
                generated = Path(command[2])
                (generated / "info.json").write_text("{}", encoding="utf-8")
                (generated / "new_marker.txt").write_text("new", encoding="utf-8")
                return mock.Mock(returncode=0)

            with mock.patch.object(
                accuracy_runner.subprocess, "run", side_effect=fake_run
            ) as run:
                accuracy_runner.regenerate_cpu_reference(project, reference)

            self.assertEqual(run.call_count, 1)
            self.assertFalse((reference / "old_marker.txt").exists())
            self.assertEqual(
                (reference / "new_marker.txt").read_text(encoding="utf-8"), "new"
            )
            self.assertTrue((reference / "info.json").is_file())

    def test_longtail_input_is_run_scoped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            output = root / "input.csv"
            fields = ["NO", "op", "baseline", "time", "score"]
            rows = [
                {
                    "NO": number,
                    "op": f"operator_{number}",
                    "baseline": "",
                    "time": "",
                    "score": "",
                }
                for number in range(40)
            ]
            write_csv(
                source,
                fields,
                rows,
            )
            identities = longtail_runner.prepare_input(source, output, "token")
            self.assertEqual(len(identities), 40)
            with output.open("r", encoding="utf-8", newline="") as file_obj:
                row = next(csv.DictReader(file_obj))
            self.assertEqual(row["baseline"], "")
            self.assertEqual(row["time"], "")
            self.assertEqual(row["score"], "")
            self.assertEqual(row["aibench_run_token"], "token")

    def test_transformer_normalization(self):
        common = {
            "device": "cuda:0",
            "dtype": "fp32",
            "mode": "forward",
            "training": False,
            "samples_per_second": 1.0,
            "batch_size": 32,
            "sequence_length": 512,
            "d_model": 512,
            "heads": 8,
            "ffn_hidden": 2048,
            "warmup": 20,
            "iterations": 1000,
        }
        payload = [
            dict(common, block="encoder", latency_ms=1.0, memory_length=None),
            dict(common, block="decoder", latency_ms=2.0, memory_length=256),
        ]
        rows = transformer_runner.normalize_results(payload, "fingerprint")
        self.assertEqual(rows[0]["key_value_sequence_length"], 512)
        self.assertEqual(rows[1]["key_value_sequence_length"], 256)
        self.assertEqual(rows[1]["aibench_workload_fingerprint"], "fingerprint")


if __name__ == "__main__":
    unittest.main()

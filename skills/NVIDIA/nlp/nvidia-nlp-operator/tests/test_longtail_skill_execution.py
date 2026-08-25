#!/usr/bin/env python3
"""LongTail Skill 确定性 runner 的工作目录和结果隔离回归测试。"""

import importlib.util
from pathlib import Path

SKILL_PATH = Path(__file__).resolve().parents[1] / "SKILL.md"
RUNNER_PATH = SKILL_PATH.parent / "scripts" / "run_longtail.py"


def _load_runner():
    module_spec = importlib.util.spec_from_file_location("run_longtail", RUNNER_PATH)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


def _fake_project(root: Path) -> Path:
    api = root / "long_tail_bench" / "api" / "api.py"
    api.parent.mkdir(parents=True)
    api.write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('-f')\n"
        "parser.add_argument('--outcsv')\n"
        "args = parser.parse_args()\n"
        "project_root = Path(__file__).resolve().parents[2]\n"
        "assert Path.cwd() == project_root\n"
        "(project_root / 'results').mkdir(exist_ok=True)\n"
        "(project_root / 'results' / 'torch.json').write_text("
        "'{\"fresh\":true}', encoding='utf-8')\n"
        "Path(args.outcsv).write_text("
        "Path(args.f).read_text(encoding='utf-8'), encoding='utf-8')\n",
        encoding="utf-8",
    )
    return root


def test_longtail_skill_calls_deterministic_runner():
    skill = SKILL_PATH.read_text(encoding="utf-8")

    assert "python3 /workspace/scripts/run_longtail.py" in skill
    assert "--f32-project-root /workspace/operators/LongTail-Bench" in skill
    assert "--f16-project-root /workspace/operators/LongTail-Bench-fp16" in skill
    assert "python ./LongTail-Bench/long_tail_bench/api/api.py" not in skill
    assert "python ./LongTail-Bench-fp16/long_tail_bench/api/api.py" not in skill


def test_longtail_runner_uses_project_cwd_and_replaces_stale_results(tmp_path):
    skill = SKILL_PATH.read_text(encoding="utf-8")
    project = _fake_project(tmp_path / "project")
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "output.csv"
    log = tmp_path / "run.log"
    raw_result = project / "results" / "torch.json"
    manifest.write_text("NO,op\n0,bbox2delta\n", encoding="utf-8")
    raw_result.parent.mkdir(parents=True)
    raw_result.write_text('{"stale":true}', encoding="utf-8")
    output.write_text("stale", encoding="utf-8")

    runner = _load_runner()
    runner.run_variant(project, manifest, output, log)

    assert raw_result.read_text(encoding="utf-8") == '{"fresh":true}'
    assert output.read_text(encoding="utf-8") == manifest.read_text(encoding="utf-8")
    assert log.is_file()
    assert "必须直接调用上述 `run_longtail.py`" in skill
    assert "防止失败 case 复用历史结果" in skill


def test_longtail_runner_propagates_benchmark_failure(tmp_path):
    project = _fake_project(tmp_path / "project")
    api = project / "long_tail_bench" / "api" / "api.py"
    api.write_text("raise SystemExit(7)\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("NO,op\n0,bbox2delta\n", encoding="utf-8")

    runner = _load_runner()
    try:
        runner.run_variant(project, manifest, tmp_path / "output.csv", tmp_path / "run.log")
    except RuntimeError as exc:
        assert "exit code 7" in str(exc)
    else:
        raise AssertionError("run_variant() must propagate the benchmark failure")

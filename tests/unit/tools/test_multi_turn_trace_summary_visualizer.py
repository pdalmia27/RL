# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYZER_SCRIPT = REPO_ROOT / "tools" / "multi_turn_trace_analyzer.py"
VISUALIZER_SCRIPT = REPO_ROOT / "tools" / "multi_turn_trace_summary_visualizer.py"
_VISUALIZER_SPEC = importlib.util.spec_from_file_location("multi_turn_trace_summary_visualizer", VISUALIZER_SCRIPT)
assert _VISUALIZER_SPEC and _VISUALIZER_SPEC.loader
visualizer = importlib.util.module_from_spec(_VISUALIZER_SPEC)
_VISUALIZER_SPEC.loader.exec_module(visualizer)


def write_trace(path: Path, trace_events: list[dict]) -> None:
    path.write_text(json.dumps({"traceEvents": trace_events}), encoding="utf-8")


def build_synthetic_trace() -> list[dict]:
    return [
        {"ph": "M", "pid": 1, "tid": 0, "name": "process_name", "args": {"name": "toy_task (2 rollouts)"}},
        {"ph": "M", "pid": 1, "tid": 1, "name": "thread_name", "args": {"name": "R1 [PASS] gen=3s eval=0s llm=2s tool=1s"}},
        {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "R2 [FAIL] gen=2s eval=0s llm=1s tool=1s"}},
        {
            "ph": "X",
            "pid": 1,
            "tid": 1,
            "cat": "llm_generation",
            "name": "LLM Generation (GPU)",
            "ts": 0,
            "dur": 10,
            "args": {"turn": 0, "prompt_tokens": 8, "completion_tokens": 4, "response_id": "a"},
        },
        {
            "ph": "X",
            "pid": 1,
            "tid": 2,
            "cat": "tool_execution",
            "name": "Tool Execution (CPU)",
            "ts": 2,
            "dur": 12,
            "args": {"turn": 0, "observation_type": "CmdOutputObservation", "observation_id": "b", "message": "hello"},
        },
        {
            "ph": "X",
            "pid": 1,
            "tid": 1,
            "cat": "framework_overhead",
            "name": "Framework Overhead",
            "ts": 10,
            "dur": 3,
            "args": {"turn": 0},
        },
        {
            "ph": "X",
            "pid": 1,
            "tid": 2,
            "cat": "llm_generation",
            "name": "LLM Generation (GPU)",
            "ts": 15,
            "dur": 8,
            "args": {"turn": 1, "prompt_tokens": 9, "completion_tokens": 6, "response_id": "c"},
        },
    ]


def run_analyzer(trace_path: Path, outdir: Path) -> None:
    subprocess.run(
        [sys.executable, str(ANALYZER_SCRIPT), "--trace", str(trace_path), "--outdir", str(outdir)],
        check=True,
    )


def run_visualizer(input_dir: Path, output_html: Path) -> None:
    subprocess.run(
        [sys.executable, str(VISUALIZER_SCRIPT), "--input-dir", str(input_dir), "--output-html", str(output_html)],
        check=True,
    )


def test_parse_task_name_handles_trace_specific_patterns():
    assert visualizer.parse_task_name("instance_shivammathur__setup-php-4bb4f1812c099fe4c9941b3a0b9e9d854947371b (32 rollouts)") == "shivammathur/setup-php · 4bb4f181"
    assert visualizer.parse_task_name("python__mypy-15722 (32 rollouts)") == "mypy · 15722"
    assert visualizer.parse_task_name("pandas-dev__pandas-fc6b441ba527ca32b460ae4f4f5a6802335497f9 (32 rollouts)") == "pandas-dev/pandas · fc6b441b"
    assert visualizer.parse_task_name("toy_task (2 rollouts)") == "toy_task"


def test_select_focus_tasks_prefers_diverse_task_families():
    rows = [
        {"task": "pandas-dev__pandas-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa (32 rollouts)", "avg_e2e_s": 10.0},
        {"task": "pandas-dev__pandas-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb (32 rollouts)", "avg_e2e_s": 9.5},
        {"task": "instance_facebook__metro-cccccccccccccccccccccccccccccccccccccccc (32 rollouts)", "avg_e2e_s": 9.0},
        {"task": "python__mypy-15722 (32 rollouts)", "avg_e2e_s": 8.5},
        {"task": "aio-libs__aiohttp-dddddddddddddddddddddddddddddddddddddddd (32 rollouts)", "avg_e2e_s": 8.0},
        {"task": "numpy__numpy-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee (32 rollouts)", "avg_e2e_s": 7.5},
    ]

    selected = visualizer.select_focus_tasks(rows, count=5)

    assert selected == [
        "pandas-dev__pandas-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa (32 rollouts)",
        "instance_facebook__metro-cccccccccccccccccccccccccccccccccccccccc (32 rollouts)",
        "python__mypy-15722 (32 rollouts)",
        "aio-libs__aiohttp-dddddddddddddddddddddddddddddddddddddddd (32 rollouts)",
        "numpy__numpy-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee (32 rollouts)",
    ]


def test_visualizer_renders_expected_sections_from_analyzer_outputs(tmp_path):
    trace_path = tmp_path / "synthetic_trace.json"
    outdir = tmp_path / "analyzer_out"
    html_path = tmp_path / "visual_summary.html"
    write_trace(trace_path, build_synthetic_trace())

    run_analyzer(trace_path, outdir)
    run_visualizer(outdir, html_path)

    assert html_path.is_file()
    content = html_path.read_text(encoding="utf-8")
    for text in [
        "Per-task E2E across rollouts",
        "Rollout E2E box plots for the 5 highest-E2E tasks",
        "Tool latency by task",
        "Tool response length by task",
        "Prefill vs decode work by turn",
        "Trace-level concurrency",
        "GPU efficiency loss due to tool calling",
        "toy_task (2 rollouts)",
        "toy_task",
        "Show table",
    ]:
        assert text in content


def test_visualizer_handles_missing_concurrency_outputs_gracefully(tmp_path):
    trace_path = tmp_path / "synthetic_trace.json"
    outdir = tmp_path / "analyzer_out"
    html_path = tmp_path / "visual_summary.html"
    write_trace(trace_path, build_synthetic_trace())

    run_analyzer(trace_path, outdir)
    (outdir / "concurrency_intervals.csv").unlink()
    (outdir / "concurrency_summary.csv").unlink()

    run_visualizer(outdir, html_path)

    content = html_path.read_text(encoding="utf-8")
    assert "No concurrency outputs found. Re-run the analyzer with the current V2 version." in content

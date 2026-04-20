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

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "tools" / "multi_turn_trace_analyzer.py"
REAL_TRACE_PATH = Path(
    "/home/scratch.pdalmia2/dlsim3_posttraining_agent_e2e_20260313/swebench_results_1771545534478_e75dd1cf_trace_sync_iter3_reorder.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location("multi_turn_trace_analyzer", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def first_matching_row(path: Path, predicate) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if predicate(row):
                return row
    raise AssertionError(f"No matching row found in {path}")


def write_trace(path: Path, trace_events: list[dict]) -> None:
    path.write_text(json.dumps({"traceEvents": trace_events}), encoding="utf-8")


def build_synthetic_trace() -> list[dict]:
    return [
        {"ph": "M", "pid": 100, "tid": 0, "name": "process_name", "args": {"name": "task_alpha (2 rollouts)"}},
        {
            "ph": "M",
            "pid": 100,
            "tid": 1,
            "name": "thread_name",
            "args": {"name": "R1 [PASS] gen=3s eval=0s llm=2s tool=1s"},
        },
        {
            "ph": "M",
            "pid": 100,
            "tid": 2,
            "name": "thread_name",
            "args": {"name": "R2 [FAIL] gen=1s eval=0s llm=1s tool=0s"},
        },
        {"ph": "i", "pid": 100, "tid": 1, "name": "ignored_instant", "ts": 5},
        {"ph": "X", "pid": 100, "tid": 1, "cat": "queue_wait", "name": "Ray Queue Wait", "ts": 0, "dur": 10, "args": {"type": "startup"}},
        {
            "ph": "X",
            "pid": 100,
            "tid": 1,
            "cat": "llm_generation",
            "name": "LLM Generation (GPU)",
            "ts": 10,
            "dur": 20,
            "args": {"turn": 1, "prompt_tokens": 5, "completion_tokens": 7, "response_id": "resp-1"},
        },
        {
            "ph": "X",
            "pid": 100,
            "tid": 1,
            "cat": "tool_execution",
            "name": "Tool Execution (CPU)",
            "ts": 30,
            "dur": 30,
            "args": {
                "turn": 1,
                "observation_type": "CmdOutputObservation",
                "observation_id": "obs-1",
                "message": "hi\nthere",
            },
        },
        {
            "ph": "X",
            "pid": 100,
            "tid": 1,
            "cat": "framework_overhead",
            "name": "Framework Overhead",
            "ts": 60,
            "dur": 5,
            "args": {"turn": 1},
        },
        {
            "ph": "X",
            "pid": 100,
            "tid": 1,
            "cat": "llm_generation",
            "name": "LLM Generation (GPU)",
            "ts": 70,
            "dur": 10,
            "args": {"turn": 2, "prompt_tokens": 11, "completion_tokens": 3, "response_id": "resp-2"},
        },
        {
            "ph": "X",
            "pid": 100,
            "tid": 1,
            "cat": "evaluation",
            "name": "Evaluation (CPU)",
            "ts": 90,
            "dur": 4,
            "args": {"resolved": True, "timestamp_source": "wall_clock", "type": "judge"},
        },
        {
            "ph": "X",
            "pid": 100,
            "tid": 2,
            "cat": "llm_generation",
            "name": "LLM Generation (GPU)",
            "ts": 5,
            "dur": 15,
            "args": {"turn": 1, "prompt_tokens": 4, "completion_tokens": 6, "response_id": "resp-3"},
        },
        {
            "ph": "X",
            "pid": 100,
            "tid": 2,
            "cat": "tool_execution",
            "name": "Tool Execution (CPU)",
            "ts": 25,
            "dur": 20,
            "args": {"turn": 1, "observation_type": "FileReadObservation", "observation_id": "obs-2"},
        },
    ]


def test_load_trace_normalizes_metadata_and_optional_fields(tmp_path):
    module = load_module()
    trace_path = tmp_path / "synthetic_trace.json"
    write_trace(trace_path, build_synthetic_trace())

    events, process_names, thread_names, phase_counts = module.load_trace(trace_path)

    assert phase_counts["M"] == 3
    assert phase_counts["X"] == 8
    assert phase_counts["i"] == 1
    assert len(events) == 8
    assert process_names[100] == "task_alpha (2 rollouts)"
    assert thread_names[(100, 1)].startswith("R1 [PASS]")

    first_llm = next(event for event in events if event["cat"] == "llm_generation" and event["tid"] == 1)
    assert first_llm["task_name"] == "task_alpha (2 rollouts)"
    assert first_llm["rollout_id"] == 1
    assert first_llm["status"] == "PASS"
    assert first_llm["turn"] == 1
    assert first_llm["prompt_tokens"] == 5
    assert first_llm["completion_tokens"] == 7
    assert first_llm["response_id"] == "resp-1"

    tool_event = next(event for event in events if event["cat"] == "tool_execution" and event["tid"] == 1)
    assert tool_event["message"] == "hi\\nthere"
    assert tool_event["message_len_chars"] == 9
    assert tool_event["observation_type"] == "CmdOutputObservation"
    assert tool_event["observation_id"] == "obs-1"

    tool_without_message = next(event for event in events if event["cat"] == "tool_execution" and event["tid"] == 2)
    assert tool_without_message["message"] is None
    assert tool_without_message["message_len_chars"] is None


def test_aggregate_builds_expected_trajectory_task_and_turn_rows(tmp_path):
    module = load_module()
    trace_path = tmp_path / "synthetic_trace.json"
    write_trace(trace_path, build_synthetic_trace())

    events, process_names, thread_names, phase_counts = module.load_trace(trace_path)
    aggregates = module.aggregate(events)
    summary = module.build_summary_json(trace_path, events, process_names, thread_names, phase_counts, aggregates)

    trajectory_rows = {(row["pid"], row["tid"]): row for row in aggregates["trajectory_rows"]}
    traj1 = trajectory_rows[(100, 1)]
    traj2 = trajectory_rows[(100, 2)]

    assert traj1["e2e_wall_us"] == 94
    assert traj1["llm_generation_us"] == 30
    assert traj1["tool_execution_us"] == 30
    assert traj1["framework_overhead_us"] == 5
    assert traj1["queue_wait_us"] == 10
    assert traj1["evaluation_us"] == 4
    assert traj1["llm_event_count"] == 2
    assert traj1["tool_event_count"] == 1
    assert traj1["turn_count"] == 2
    assert traj1["total_prompt_tokens"] == 16
    assert traj1["total_completion_tokens"] == 10

    assert traj2["e2e_wall_us"] == 40
    assert traj2["status"] == "FAIL"
    assert traj2["tool_execution_us"] == 20
    assert traj2["turn_count"] == 1

    assert len(aggregates["task_rows"]) == 1
    task_row = aggregates["task_rows"][0]
    assert task_row["trajectory_count"] == 2
    assert task_row["pass_count"] == 1
    assert task_row["fail_count"] == 1
    assert task_row["e2e_wall_p50_us"] == 67.0
    assert task_row["completion_tokens_p50"] == 8.0
    assert task_row["completion_tokens_p90"] == 9.6

    turn_rows = {(row["pid"], row["tid"], row["turn"]): row for row in aggregates["turn_rows"]}
    assert set(turn_rows) == {(100, 1, 1), (100, 1, 2), (100, 2, 1)}
    assert turn_rows[(100, 1, 1)]["tool_message_chars_sum"] == 9
    assert turn_rows[(100, 1, 1)]["tool_message_count"] == 1
    assert turn_rows[(100, 1, 2)]["tool_event_count"] == 0
    assert turn_rows[(100, 2, 1)]["tool_message_chars_sum"] == 0

    concurrency_rows = aggregates["concurrency_rows"]
    assert concurrency_rows[0]["start_ts_us"] == 0
    assert concurrency_rows[0]["end_ts_us"] == 5
    assert concurrency_rows[0]["active_total"] == 1
    assert concurrency_rows[0]["active_queue_wait"] == 1
    assert concurrency_rows[1]["start_ts_us"] == 5
    assert concurrency_rows[1]["end_ts_us"] == 10
    assert concurrency_rows[1]["active_total"] == 2
    assert concurrency_rows[1]["active_llm_generation"] == 1
    assert concurrency_rows[1]["active_queue_wait"] == 1
    assert any(row["dur_us"] == 5 and row["active_total"] == 0 for row in concurrency_rows)

    concurrency_summary = {row["metric"]: row for row in aggregates["concurrency_summary_rows"]}
    assert concurrency_summary["active_total"]["max_active"] == 2
    assert concurrency_summary["active_llm_generation"]["max_active"] == 2
    assert concurrency_summary["active_tool_execution"]["max_active"] == 2

    report = module.render_report(
        summary=summary,
        trajectory_rows=aggregates["trajectory_rows"],
        task_rows=aggregates["task_rows"],
        turn_rows=aggregates["turn_rows"],
        tool_events=aggregates["tool_events"],
        llm_events=aggregates["llm_events"],
        concurrency_rows=aggregates["concurrency_rows"],
        concurrency_summary_rows=aggregates["concurrency_summary_rows"],
    )
    for heading in [
        "## Trace Overview",
        "## Across-Trajectory E2E Analysis",
        "## Trace-Level Concurrency",
        "## Tool Latency Distribution Per Task",
        "## Tool Response-Length vs Latency Per Task",
        "## GPU Response-Length Distribution Per Turn For Every Task",
    ]:
        assert heading in report


def test_cli_on_real_trace_emits_expected_artifacts_and_invariants(tmp_path):
    if not REAL_TRACE_PATH.is_file():
        pytest.skip(f"Missing real trace fixture at {REAL_TRACE_PATH}")

    outdir = tmp_path / "analyzer_out"
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--trace", str(REAL_TRACE_PATH), "--outdir", str(outdir)],
        check=True,
    )

    expected_files = [
        "events.csv",
        "tool_events.csv",
        "llm_events.csv",
        "trajectory_summary.csv",
        "task_summary.csv",
        "turn_summary.csv",
        "concurrency_intervals.csv",
        "concurrency_summary.csv",
        "summary.json",
        "report.md",
    ]
    for name in expected_files:
        assert (outdir / name).is_file(), f"Missing output artifact: {name}"

    with (outdir / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["event_count"] == 346_940
    assert summary["phase_counts"]["X"] == 346_940
    assert summary["phase_counts"]["M"] == 1_632
    assert summary["task_count"] == 48
    assert summary["trajectory_count"] == 1_536
    assert summary["category_counts"]["tool_execution"] == 116_582
    assert summary["category_counts"]["llm_generation"] == 117_831

    assert count_csv_rows(outdir / "events.csv") == 346_940
    assert count_csv_rows(outdir / "tool_events.csv") == 116_582
    assert count_csv_rows(outdir / "llm_events.csv") == 117_831
    assert count_csv_rows(outdir / "trajectory_summary.csv") == 1_536
    assert count_csv_rows(outdir / "task_summary.csv") == 48
    assert count_csv_rows(outdir / "turn_summary.csv") == 117_831
    assert count_csv_rows(outdir / "concurrency_intervals.csv") > 0
    assert count_csv_rows(outdir / "concurrency_summary.csv") == 8

    concurrency_summary_rows = read_csv_rows(outdir / "concurrency_summary.csv")
    by_metric = {row["metric"]: row for row in concurrency_summary_rows}
    assert int(float(by_metric["active_total"]["max_active"])) > 0
    assert int(float(by_metric["active_llm_generation"]["max_active"])) > 0
    assert int(float(by_metric["active_tool_execution"]["max_active"])) > 0

    sample_llm = first_matching_row(
        outdir / "llm_events.csv",
        lambda row: row["turn"] and row["prompt_tokens"] and row["completion_tokens"],
    )
    assert sample_llm["response_id"]

    sample_tool = first_matching_row(
        outdir / "tool_events.csv",
        lambda row: row["observation_type"],
    )
    assert sample_tool["observation_id"]
    if sample_tool["message"]:
        assert sample_tool["message_len_chars"]

    report = (outdir / "report.md").read_text(encoding="utf-8")
    for heading in [
        "## Trace Overview",
        "## Across-Trajectory E2E Analysis",
        "## Trace-Level Concurrency",
        "## Tool Latency Distribution Per Task",
        "## Tool Response-Length vs Latency Per Task",
        "## GPU Response-Length Distribution Per Turn For Every Task",
    ]:
        assert heading in report

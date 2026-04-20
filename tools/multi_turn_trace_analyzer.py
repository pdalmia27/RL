#!/usr/bin/env python3
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
"""Analyze a single Chrome/Perfetto multi-turn RL trace.

This utility parses Chrome trace JSON, normalizes complete-duration events
(`ph="X"`), and emits CSV/JSON/Markdown artifacts for quick multi-turn RL
analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TRACE_CATEGORIES = [
    "llm_generation",
    "tool_execution",
    "framework_overhead",
    "queue_wait",
    "container_startup",
    "evaluation",
    "generation_total",
]

TRAJECTORY_TIME_CATEGORIES = [
    "llm_generation",
    "tool_execution",
    "framework_overhead",
    "queue_wait",
    "container_startup",
    "evaluation",
]

ROLLOUT_LABEL_RE = re.compile(r"^R(?P<rollout_id>\d+)\s+\[(?P<status>[A-Z]+)\]")
TOOL_MESSAGE_BINS = [
    (0, 0, "0"),
    (1, 32, "1-32"),
    (33, 128, "33-128"),
    (129, 512, "129-512"),
    (513, 2048, "513-2048"),
    (2049, None, "2049+"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a single multi-turn RL Chrome/Perfetto trace."
    )
    parser.add_argument("--trace", required=True, help="Path to trace JSON")
    parser.add_argument("--outdir", required=True, help="Directory for outputs")
    return parser.parse_args()


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def normalize_message(value: Any) -> str | None:
    text = safe_str(value)
    if text is None:
        return None
    return text.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")


def parse_rollout_label(label: str | None) -> tuple[int | None, str | None]:
    if not label:
        return None, None
    match = ROLLOUT_LABEL_RE.match(label)
    if not match:
        return None, None
    return int(match.group("rollout_id")), match.group("status")


def percentile(values: list[int | float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def mean(values: list[int | float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def fmt_us_as_ms(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1000.0:.3f}"


def fmt_us_as_s(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1_000_000.0:.3f}"


def fmt_num(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def tool_message_bucket(length: int | None) -> str:
    if length is None:
        return "missing"
    for lower, upper, label in TOOL_MESSAGE_BINS:
        if upper is None and length >= lower:
            return label
        if upper is not None and lower <= length <= upper:
            return label
    return "missing"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def time_weighted_percentile(intervals: list[dict[str, Any]], field: str, pct: float) -> float | None:
    weighted = [(row[field], row["dur_us"]) for row in intervals if row["dur_us"] > 0]
    total_weight = sum(weight for _, weight in weighted)
    if not weighted or total_weight <= 0:
        return None
    threshold = total_weight * pct
    cumulative = 0
    for value, weight in sorted(weighted, key=lambda item: item[0]):
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(weighted[-1][0])


def load_trace(trace_path: Path) -> tuple[list[dict[str, Any]], dict[int, str], dict[tuple[int, int], str], Counter[str]]:
    with trace_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trace_events = data.get("traceEvents")
    if not isinstance(trace_events, list):
        raise ValueError("Trace JSON missing top-level traceEvents list")

    process_names: dict[int, str] = {}
    thread_names: dict[tuple[int, int], str] = {}
    phase_counts: Counter[str] = Counter()
    normalized_events: list[dict[str, Any]] = []

    for event in trace_events:
        if not isinstance(event, dict):
            continue

        phase = safe_str(event.get("ph"))
        if phase is None:
            continue
        phase_counts[phase] += 1

        if phase == "M":
            pid = safe_int(event.get("pid"))
            tid = safe_int(event.get("tid"))
            name = safe_str(event.get("name"))
            args = event.get("args", {})
            if not isinstance(args, dict):
                args = {}
            if name == "process_name" and pid is not None:
                process_names[pid] = safe_str(args.get("name")) or f"pid_{pid}"
            elif name == "thread_name" and pid is not None and tid is not None:
                thread_names[(pid, tid)] = safe_str(args.get("name")) or f"tid_{tid}"
            continue

        if phase != "X":
            continue

        pid = safe_int(event.get("pid"))
        tid = safe_int(event.get("tid"))
        ts_us = safe_int(event.get("ts"))
        dur_us = safe_int(event.get("dur"))
        if pid is None or tid is None or ts_us is None or dur_us is None:
            continue

        args = event.get("args", {})
        if not isinstance(args, dict):
            args = {}

        rollout_label = thread_names.get((pid, tid))
        rollout_id, status = parse_rollout_label(rollout_label)
        task_name = process_names.get(pid, f"pid_{pid}")
        message = normalize_message(args.get("message"))

        normalized_events.append(
            {
                "pid": pid,
                "tid": tid,
                "task_name": task_name,
                "rollout_label": rollout_label,
                "rollout_id": rollout_id,
                "status": status,
                "cat": safe_str(event.get("cat")),
                "name": safe_str(event.get("name")),
                "ts_us": ts_us,
                "dur_us": dur_us,
                "end_ts_us": ts_us + dur_us,
                "turn": safe_int(args.get("turn")),
                "prompt_tokens": safe_int(args.get("prompt_tokens")),
                "completion_tokens": safe_int(args.get("completion_tokens")),
                "response_id": safe_str(args.get("response_id")),
                "observation_type": safe_str(args.get("observation_type")),
                "observation_id": safe_str(args.get("observation_id")),
                "message": message,
                "message_len_chars": len(message) if message is not None else None,
                "resolved": args.get("resolved"),
                "timestamp_source": safe_str(args.get("timestamp_source")),
            }
        )

    return normalized_events, process_names, thread_names, phase_counts


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(events: list[dict[str, Any]]) -> dict[str, Any]:
    category_totals_us: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    trajectory_events: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    task_events: dict[int, list[dict[str, Any]]] = defaultdict(list)
    turn_events: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)

    tool_events = []
    llm_events = []

    for event in events:
        cat = event["cat"]
        if cat:
            category_totals_us[cat] += event["dur_us"]
            category_counts[cat] += 1

        key = (event["pid"], event["tid"])
        trajectory_events[key].append(event)
        task_events[event["pid"]].append(event)

        if event["turn"] is not None:
            turn_events[(event["pid"], event["tid"], event["turn"])].append(event)

        if cat == "tool_execution":
            tool_events.append(event)
        elif cat == "llm_generation":
            llm_events.append(event)

    trajectory_rows = []
    for (pid, tid), evs in sorted(trajectory_events.items()):
        cat_sums = Counter()
        llm_prompt = 0
        llm_completion = 0
        turns = set()
        llm_count = 0
        tool_count = 0
        task_name = evs[0]["task_name"]
        rollout_id = evs[0]["rollout_id"]
        status = evs[0]["status"]
        rollout_label = evs[0]["rollout_label"]
        for event in evs:
            cat = event["cat"]
            if cat in TRAJECTORY_TIME_CATEGORIES:
                cat_sums[cat] += event["dur_us"]
            if event["turn"] is not None:
                turns.add(event["turn"])
            if cat == "llm_generation":
                llm_count += 1
                llm_prompt += event["prompt_tokens"] or 0
                llm_completion += event["completion_tokens"] or 0
            elif cat == "tool_execution":
                tool_count += 1
        first_ts = min(e["ts_us"] for e in evs)
        last_end = max(e["end_ts_us"] for e in evs)
        trajectory_rows.append(
            {
                "task_name": task_name,
                "pid": pid,
                "tid": tid,
                "rollout_label": rollout_label,
                "rollout_id": rollout_id,
                "status": status,
                "first_ts_us": first_ts,
                "last_end_ts_us": last_end,
                "e2e_wall_us": last_end - first_ts,
                "llm_generation_us": cat_sums["llm_generation"],
                "tool_execution_us": cat_sums["tool_execution"],
                "framework_overhead_us": cat_sums["framework_overhead"],
                "queue_wait_us": cat_sums["queue_wait"],
                "container_startup_us": cat_sums["container_startup"],
                "evaluation_us": cat_sums["evaluation"],
                "llm_event_count": llm_count,
                "tool_event_count": tool_count,
                "turn_count": len(turns),
                "total_prompt_tokens": llm_prompt,
                "total_completion_tokens": llm_completion,
            }
        )

    task_rows = []
    for pid, evs in sorted(task_events.items()):
        traj_rows = [row for row in trajectory_rows if row["pid"] == pid]
        task_name = traj_rows[0]["task_name"] if traj_rows else f"pid_{pid}"
        statuses = Counter(row["status"] or "UNKNOWN" for row in traj_rows)
        task_rows.append(
            {
                "task_name": task_name,
                "pid": pid,
                "trajectory_count": len(traj_rows),
                "pass_count": statuses["PASS"],
                "fail_count": statuses["FAIL"],
                "e2e_wall_p50_us": percentile([row["e2e_wall_us"] for row in traj_rows], 0.50),
                "e2e_wall_p90_us": percentile([row["e2e_wall_us"] for row in traj_rows], 0.90),
                "e2e_wall_p99_us": percentile([row["e2e_wall_us"] for row in traj_rows], 0.99),
                "llm_time_p50_us": percentile([row["llm_generation_us"] for row in traj_rows], 0.50),
                "tool_time_p50_us": percentile([row["tool_execution_us"] for row in traj_rows], 0.50),
                "framework_time_p50_us": percentile([row["framework_overhead_us"] for row in traj_rows], 0.50),
                "completion_tokens_p50": percentile([row["total_completion_tokens"] for row in traj_rows], 0.50),
                "completion_tokens_p90": percentile([row["total_completion_tokens"] for row in traj_rows], 0.90),
            }
        )

    turn_rows = []
    for (pid, tid, turn), evs in sorted(turn_events.items()):
        cat_sums = Counter()
        llm_event_count = 0
        tool_event_count = 0
        prompt_tokens_sum = 0
        completion_tokens_sum = 0
        tool_message_chars_sum = 0
        tool_message_count = 0
        task_name = evs[0]["task_name"]
        rollout_id = evs[0]["rollout_id"]
        for event in evs:
            cat = event["cat"]
            if cat in ["llm_generation", "tool_execution", "framework_overhead", "queue_wait"]:
                cat_sums[cat] += event["dur_us"]
            if cat == "llm_generation":
                llm_event_count += 1
                prompt_tokens_sum += event["prompt_tokens"] or 0
                completion_tokens_sum += event["completion_tokens"] or 0
            elif cat == "tool_execution":
                tool_event_count += 1
                if event["message_len_chars"] is not None:
                    tool_message_chars_sum += event["message_len_chars"]
                    tool_message_count += 1
        turn_rows.append(
            {
                "task_name": task_name,
                "pid": pid,
                "tid": tid,
                "rollout_id": rollout_id,
                "turn": turn,
                "llm_generation_us": cat_sums["llm_generation"],
                "tool_execution_us": cat_sums["tool_execution"],
                "framework_overhead_us": cat_sums["framework_overhead"],
                "queue_wait_us": cat_sums["queue_wait"],
                "llm_event_count": llm_event_count,
                "tool_event_count": tool_event_count,
                "prompt_tokens_sum": prompt_tokens_sum,
                "completion_tokens_sum": completion_tokens_sum,
                "tool_message_chars_sum": tool_message_chars_sum,
                "tool_message_count": tool_message_count,
            }
        )

    concurrency_deltas: dict[int, Counter[str]] = defaultdict(Counter)
    for event in events:
        ts_us = event["ts_us"]
        end_ts_us = event["end_ts_us"]
        if end_ts_us <= ts_us:
            continue
        cat = event["cat"]
        concurrency_deltas[ts_us]["active_total"] += 1
        concurrency_deltas[end_ts_us]["active_total"] -= 1
        if cat in TRACE_CATEGORIES:
            field = f"active_{cat}"
            concurrency_deltas[ts_us][field] += 1
            concurrency_deltas[end_ts_us][field] -= 1

    concurrency_rows = []
    if concurrency_deltas:
        current = Counter()
        timestamps = sorted(concurrency_deltas)
        for idx, ts_us in enumerate(timestamps[:-1]):
            for field, delta in concurrency_deltas[ts_us].items():
                current[field] += delta
            next_ts_us = timestamps[idx + 1]
            row = {
                "start_ts_us": ts_us,
                "end_ts_us": next_ts_us,
                "dur_us": next_ts_us - ts_us,
                "active_total": current["active_total"],
            }
            for cat in TRACE_CATEGORIES:
                row[f"active_{cat}"] = current[f"active_{cat}"]
            concurrency_rows.append(row)

    concurrency_summary_rows = []
    concurrency_fields = ["active_total"] + [f"active_{cat}" for cat in TRACE_CATEGORIES]
    for field in concurrency_fields:
        values = [row[field] for row in concurrency_rows]
        concurrency_summary_rows.append(
            {
                "metric": field,
                "interval_count": len(concurrency_rows),
                "time_covered_us": sum(row["dur_us"] for row in concurrency_rows),
                "mean_active": mean(values),
                "max_active": max(values) if values else None,
                "time_weighted_p50_active": time_weighted_percentile(concurrency_rows, field, 0.50),
                "time_weighted_p90_active": time_weighted_percentile(concurrency_rows, field, 0.90),
                "time_weighted_p99_active": time_weighted_percentile(concurrency_rows, field, 0.99),
            }
        )

    return {
        "category_totals_us": category_totals_us,
        "category_counts": category_counts,
        "trajectory_rows": trajectory_rows,
        "task_rows": task_rows,
        "turn_rows": turn_rows,
        "tool_events": tool_events,
        "llm_events": llm_events,
        "concurrency_rows": concurrency_rows,
        "concurrency_summary_rows": concurrency_summary_rows,
    }


def build_summary_json(
    trace_path: Path,
    events: list[dict[str, Any]],
    process_names: dict[int, str],
    thread_names: dict[tuple[int, int], str],
    phase_counts: Counter[str],
    aggregates: dict[str, Any],
) -> dict[str, Any]:
    first_ts = min((event["ts_us"] for event in events), default=None)
    last_end = max((event["end_ts_us"] for event in events), default=None)
    return {
        "trace_path": str(trace_path),
        "event_count": len(events),
        "phase_counts": dict(phase_counts),
        "task_count": len({event["pid"] for event in events}),
        "trajectory_count": len({(event["pid"], event["tid"]) for event in events}),
        "trace_span_us": (last_end - first_ts) if first_ts is not None and last_end is not None else None,
        "first_ts_us": first_ts,
        "last_end_ts_us": last_end,
        "process_name_count": len(process_names),
        "thread_name_count": len(thread_names),
        "category_counts": dict(aggregates["category_counts"]),
        "category_totals_us": dict(aggregates["category_totals_us"]),
    }


def render_report(
    summary: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    turn_rows: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
    llm_events: list[dict[str, Any]],
    concurrency_rows: list[dict[str, Any]],
    concurrency_summary_rows: list[dict[str, Any]],
) -> str:
    lines = ["# Multi-Turn RL Trace Analysis Report", ""]

    lines.extend(
        [
            "## Trace Overview",
            "",
            markdown_table(
                ["Metric", "Value"],
                [
                    ["Trace span (s)", fmt_us_as_s(summary["trace_span_us"])],
                    ["Tasks", summary["task_count"]],
                    ["Trajectories", summary["trajectory_count"]],
                    ["Total normalized events", summary["event_count"]],
                    ["Metadata events (M)", summary["phase_counts"].get("M", 0)],
                    ["Complete events (X)", summary["phase_counts"].get("X", 0)],
                ],
            ),
            "",
            markdown_table(
                ["Category", "Count", "Total Duration (s)"],
                [
                    [cat, summary["category_counts"].get(cat, 0), fmt_us_as_s(summary["category_totals_us"].get(cat, 0))]
                    for cat in TRACE_CATEGORIES
                ],
            ),
            "",
        ]
    )

    e2e_values = [row["e2e_wall_us"] for row in trajectory_rows]
    slowest = sorted(trajectory_rows, key=lambda row: row["e2e_wall_us"], reverse=True)[:10]
    status_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        status_groups[row["status"] or "UNKNOWN"].append(row)

    lines.extend(
        [
            "## Across-Trajectory E2E Analysis",
            "",
            markdown_table(
                ["Metric", "Value"],
                [
                    ["E2E p50 (s)", fmt_us_as_s(percentile(e2e_values, 0.50))],
                    ["E2E p90 (s)", fmt_us_as_s(percentile(e2e_values, 0.90))],
                    ["E2E p99 (s)", fmt_us_as_s(percentile(e2e_values, 0.99))],
                    ["E2E max (s)", fmt_us_as_s(max(e2e_values) if e2e_values else None)],
                ],
            ),
            "",
            "### Top Slowest Trajectories",
            "",
            markdown_table(
                ["Task", "Rollout", "Status", "E2E (s)", "LLM (s)", "Tool (s)", "Turns"],
                [
                    [
                        row["task_name"],
                        row["rollout_id"],
                        row["status"] or "UNKNOWN",
                        fmt_us_as_s(row["e2e_wall_us"]),
                        fmt_us_as_s(row["llm_generation_us"]),
                        fmt_us_as_s(row["tool_execution_us"]),
                        row["turn_count"],
                    ]
                    for row in slowest
                ],
            ),
            "",
            "### Pass/Fail Split vs E2E Time",
            "",
            markdown_table(
                ["Status", "Count", "Avg E2E (s)", "p50 E2E (s)", "p90 E2E (s)"],
                [
                    [
                        status,
                        len(rows),
                        fmt_us_as_s(mean([row["e2e_wall_us"] for row in rows])),
                        fmt_us_as_s(percentile([row["e2e_wall_us"] for row in rows], 0.50)),
                        fmt_us_as_s(percentile([row["e2e_wall_us"] for row in rows], 0.90)),
                    ]
                    for status, rows in sorted(status_groups.items())
                ],
            ),
            "",
            "### Per-Task E2E Percentiles",
            "",
            markdown_table(
                ["Task", "Traj", "PASS", "FAIL", "p50 E2E (s)", "p90 E2E (s)", "p99 E2E (s)"],
                [
                    [
                        row["task_name"],
                        row["trajectory_count"],
                        row["pass_count"],
                        row["fail_count"],
                        fmt_us_as_s(row["e2e_wall_p50_us"]),
                        fmt_us_as_s(row["e2e_wall_p90_us"]),
                        fmt_us_as_s(row["e2e_wall_p99_us"]),
                    ]
                    for row in task_rows
                ],
            ),
            "",
        ]
    )

    busiest = sorted(concurrency_rows, key=lambda row: (row["active_total"], row["dur_us"]), reverse=True)[:10]
    lines.extend(["## Trace-Level Concurrency", ""])
    lines.extend(
        [
            markdown_table(
                ["Metric", "Mean Active", "Max Active", "Time-weighted p50", "Time-weighted p90", "Time-weighted p99"],
                [
                    [
                        row["metric"],
                        fmt_num(row["mean_active"]),
                        fmt_num(row["max_active"]),
                        fmt_num(row["time_weighted_p50_active"]),
                        fmt_num(row["time_weighted_p90_active"]),
                        fmt_num(row["time_weighted_p99_active"]),
                    ]
                    for row in concurrency_summary_rows
                    if row["metric"]
                    in [
                        "active_total",
                        "active_llm_generation",
                        "active_tool_execution",
                        "active_framework_overhead",
                    ]
                ],
            ),
            "",
            "### Top Busiest Intervals",
            "",
            markdown_table(
                [
                    "Start (s)",
                    "End (s)",
                    "Dur (ms)",
                    "Active Total",
                    "Active LLM",
                    "Active Tool",
                    "Active Framework",
                ],
                [
                    [
                        fmt_us_as_s(row["start_ts_us"]),
                        fmt_us_as_s(row["end_ts_us"]),
                        fmt_us_as_ms(row["dur_us"]),
                        row["active_total"],
                        row["active_llm_generation"],
                        row["active_tool_execution"],
                        row["active_framework_overhead"],
                    ]
                    for row in busiest
                ],
            ),
            "",
        ]
    )

    tool_by_task: dict[str, list[int]] = defaultdict(list)
    tool_by_task_obs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for event in tool_events:
        task = event["task_name"]
        tool_by_task[task].append(event["dur_us"])
        obs = event["observation_type"] or "UNKNOWN"
        tool_by_task_obs[(task, obs)].append(event["dur_us"])

    lines.extend(["## Tool Latency Distribution Per Task", ""])
    lines.extend(
        [
            markdown_table(
                ["Task", "Count", "p50 (ms)", "p90 (ms)", "p99 (ms)", "Max (ms)"],
                [
                    [
                        task,
                        len(values),
                        fmt_us_as_ms(percentile(values, 0.50)),
                        fmt_us_as_ms(percentile(values, 0.90)),
                        fmt_us_as_ms(percentile(values, 0.99)),
                        fmt_us_as_ms(max(values) if values else None),
                    ]
                    for task, values in sorted(tool_by_task.items())
                ],
            ),
            "",
            "### By Observation Type",
            "",
            markdown_table(
                ["Task", "Observation Type", "Count", "p50 (ms)", "p90 (ms)", "p99 (ms)", "Max (ms)"],
                [
                    [
                        task,
                        obs,
                        len(values),
                        fmt_us_as_ms(percentile(values, 0.50)),
                        fmt_us_as_ms(percentile(values, 0.90)),
                        fmt_us_as_ms(percentile(values, 0.99)),
                        fmt_us_as_ms(max(values) if values else None),
                    ]
                    for (task, obs), values in sorted(tool_by_task_obs.items())
                ],
            ),
            "",
        ]
    )

    tool_bucket_task: dict[tuple[str, str], list[int]] = defaultdict(list)
    tool_bucket_task_obs: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for event in tool_events:
        task = event["task_name"]
        obs = event["observation_type"] or "UNKNOWN"
        bucket = tool_message_bucket(event["message_len_chars"])
        tool_bucket_task[(task, bucket)].append(event["dur_us"])
        tool_bucket_task_obs[(task, obs, bucket)].append(event["dur_us"])

    lines.extend(["## Tool Response-Length vs Latency Per Task", ""])
    lines.extend(
        [
            markdown_table(
                ["Task", "Message Length Bin", "Count", "Avg (ms)", "p50 (ms)", "p90 (ms)"],
                [
                    [
                        task,
                        bucket,
                        len(values),
                        fmt_us_as_ms(mean(values)),
                        fmt_us_as_ms(percentile(values, 0.50)),
                        fmt_us_as_ms(percentile(values, 0.90)),
                    ]
                    for (task, bucket), values in sorted(tool_bucket_task.items())
                ],
            ),
            "",
            "### By Observation Type",
            "",
            markdown_table(
                ["Task", "Observation Type", "Message Length Bin", "Count", "Avg (ms)", "p50 (ms)", "p90 (ms)"],
                [
                    [
                        task,
                        obs,
                        bucket,
                        len(values),
                        fmt_us_as_ms(mean(values)),
                        fmt_us_as_ms(percentile(values, 0.50)),
                        fmt_us_as_ms(percentile(values, 0.90)),
                    ]
                    for (task, obs, bucket), values in sorted(tool_bucket_task_obs.items())
                ],
            ),
            "",
        ]
    )

    llm_by_task_turn: dict[tuple[str, int], list[int]] = defaultdict(list)
    llm_by_task_all: dict[str, list[int]] = defaultdict(list)
    for event in llm_events:
        task = event["task_name"]
        if event["completion_tokens"] is not None:
            llm_by_task_all[task].append(event["completion_tokens"])
            if event["turn"] is not None:
                llm_by_task_turn[(task, event["turn"])].append(event["completion_tokens"])

    lines.extend(["## GPU Response-Length Distribution Per Turn For Every Task", ""])
    lines.extend(
        [
            markdown_table(
                ["Task", "Turn", "Count", "Mean", "p50", "p90", "p99", "Max"],
                [
                    [
                        task,
                        turn,
                        len(values),
                        fmt_num(mean(values)),
                        fmt_num(percentile(values, 0.50)),
                        fmt_num(percentile(values, 0.90)),
                        fmt_num(percentile(values, 0.99)),
                        fmt_num(max(values) if values else None),
                    ]
                    for (task, turn), values in sorted(llm_by_task_turn.items())
                ],
            ),
            "",
            "### Overall Per-Task Completion Token Percentiles",
            "",
            markdown_table(
                ["Task", "Count", "p50", "p90", "p99", "Max"],
                [
                    [
                        task,
                        len(values),
                        fmt_num(percentile(values, 0.50)),
                        fmt_num(percentile(values, 0.90)),
                        fmt_num(percentile(values, 0.99)),
                        fmt_num(max(values) if values else None),
                    ]
                    for task, values in sorted(llm_by_task_all.items())
                ],
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    events, process_names, thread_names, phase_counts = load_trace(trace_path)
    aggregates = aggregate(events)
    summary = build_summary_json(trace_path, events, process_names, thread_names, phase_counts, aggregates)

    event_fields = [
        "pid",
        "tid",
        "task_name",
        "rollout_label",
        "rollout_id",
        "status",
        "cat",
        "name",
        "ts_us",
        "dur_us",
        "end_ts_us",
        "turn",
        "prompt_tokens",
        "completion_tokens",
        "response_id",
        "observation_type",
        "observation_id",
        "message",
        "message_len_chars",
        "resolved",
        "timestamp_source",
    ]
    write_csv(outdir / "events.csv", events, event_fields)
    write_csv(outdir / "tool_events.csv", aggregates["tool_events"], event_fields)
    write_csv(outdir / "llm_events.csv", aggregates["llm_events"], event_fields)

    trajectory_fields = [
        "task_name",
        "pid",
        "tid",
        "rollout_label",
        "rollout_id",
        "status",
        "first_ts_us",
        "last_end_ts_us",
        "e2e_wall_us",
        "llm_generation_us",
        "tool_execution_us",
        "framework_overhead_us",
        "queue_wait_us",
        "container_startup_us",
        "evaluation_us",
        "llm_event_count",
        "tool_event_count",
        "turn_count",
        "total_prompt_tokens",
        "total_completion_tokens",
    ]
    write_csv(outdir / "trajectory_summary.csv", aggregates["trajectory_rows"], trajectory_fields)

    task_fields = [
        "task_name",
        "pid",
        "trajectory_count",
        "pass_count",
        "fail_count",
        "e2e_wall_p50_us",
        "e2e_wall_p90_us",
        "e2e_wall_p99_us",
        "llm_time_p50_us",
        "tool_time_p50_us",
        "framework_time_p50_us",
        "completion_tokens_p50",
        "completion_tokens_p90",
    ]
    write_csv(outdir / "task_summary.csv", aggregates["task_rows"], task_fields)

    turn_fields = [
        "task_name",
        "pid",
        "tid",
        "rollout_id",
        "turn",
        "llm_generation_us",
        "tool_execution_us",
        "framework_overhead_us",
        "queue_wait_us",
        "llm_event_count",
        "tool_event_count",
        "prompt_tokens_sum",
        "completion_tokens_sum",
        "tool_message_chars_sum",
        "tool_message_count",
    ]
    write_csv(outdir / "turn_summary.csv", aggregates["turn_rows"], turn_fields)

    concurrency_fields = [
        "start_ts_us",
        "end_ts_us",
        "dur_us",
        "active_total",
        "active_llm_generation",
        "active_tool_execution",
        "active_framework_overhead",
        "active_queue_wait",
        "active_container_startup",
        "active_evaluation",
        "active_generation_total",
    ]
    write_csv(outdir / "concurrency_intervals.csv", aggregates["concurrency_rows"], concurrency_fields)

    concurrency_summary_fields = [
        "metric",
        "interval_count",
        "time_covered_us",
        "mean_active",
        "max_active",
        "time_weighted_p50_active",
        "time_weighted_p90_active",
        "time_weighted_p99_active",
    ]
    write_csv(outdir / "concurrency_summary.csv", aggregates["concurrency_summary_rows"], concurrency_summary_fields)

    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    report = render_report(
        summary=summary,
        trajectory_rows=aggregates["trajectory_rows"],
        task_rows=aggregates["task_rows"],
        turn_rows=aggregates["turn_rows"],
        tool_events=aggregates["tool_events"],
        llm_events=aggregates["llm_events"],
        concurrency_rows=aggregates["concurrency_rows"],
        concurrency_summary_rows=aggregates["concurrency_summary_rows"],
    )
    (outdir / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

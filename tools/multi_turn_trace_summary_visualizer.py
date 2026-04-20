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
"""Render a lightweight HTML visual summary from analyzer outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROLLOUT_SUFFIX_RE = re.compile(r"\s+\((\d+)\s+rollouts\)$")
HEXISH_RE = re.compile(r"^[0-9a-fA-F]{8,}$")
GENERIC_TASK_OWNERS = {"python"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an HTML summary for multi-turn trace analyzer outputs.")
    parser.add_argument("--input-dir", required=True, help="Analyzer output directory")
    parser.add_argument("--output-html", required=True, help="Path to write the HTML summary")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def fmt_us_as_s(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) / 1_000_000.0:.2f}s"


def fmt_us_as_ms(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) / 1000.0:.2f}ms"


def fmt_num(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value >= 100:
            return f"{value:.0f}"
        return f"{value:.2f}"
    return str(value)


def sample_points(points: list[tuple[float, ...]], max_points: int) -> list[tuple[float, ...]]:
    if len(points) <= max_points:
        return points
    sampled = []
    step = (len(points) - 1) / float(max_points - 1)
    for idx in range(max_points):
        sampled.append(points[round(idx * step)])
    return sampled


def html_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "<p class='muted'>No data.</p>"
    head = "".join(f"<th>{html.escape(str(cell))}</th>" for cell in headers)
    body_rows = []
    task_col = bool(headers and str(headers[0]).lower().startswith("task"))
    for row in rows:
        cells = []
        for idx, cell in enumerate(row):
            cell_html = html.escape(str(cell))
            if task_col and idx == 0:
                cells.append(f"<td class='task-cell'><span title='{cell_html}'>{cell_html}</span></td>")
            else:
                cells.append(f"<td>{cell_html}</td>")
        body_cells = "".join(cells)
        body_rows.append(f"<tr>{body_cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def collapsible_table(button_label: str, table_html: str) -> str:
    return (
        "<details class='table-details'>"
        f"<summary>{html.escape(button_label)}</summary>"
        f"{table_html}"
        "</details>"
    )


def card_grid(cards: list[tuple[str, str]]) -> str:
    blocks = [
        f"<div class='card'><div class='card-label'>{html.escape(label)}</div><div class='card-value'>{html.escape(value)}</div></div>"
        for label, value in cards
    ]
    return f"<div class='card-grid'>{''.join(blocks)}</div>"


def parse_task_name(value: str) -> str:
    base = ROLLOUT_SUFFIX_RE.sub("", value).removeprefix("instance_")
    if "__" not in base:
        return base

    owner, right = base.split("__", 1)
    repo = right
    identifier = None
    if "-" in right:
        repo, maybe_identifier = right.rsplit("-", 1)
        if maybe_identifier:
            identifier = maybe_identifier

    owner = owner.strip()
    repo = repo.strip()
    if not repo:
        repo = right.strip()

    if owner == repo or owner in GENERIC_TASK_OWNERS:
        stem = repo
    else:
        stem = f"{owner}/{repo}"

    if identifier:
        if HEXISH_RE.fullmatch(identifier):
            identifier = identifier[:8]
        return f"{stem} · {identifier}"
    return stem


def task_family_key(value: str) -> str:
    return parse_task_name(value).split(" · ", 1)[0]


def select_focus_tasks(task_rows: list[dict[str, Any]], count: int = 5) -> list[str]:
    selected: list[str] = []
    seen_families: set[str] = set()
    for row in task_rows:
        task = row["task"]
        family = task_family_key(task)
        if family in seen_families:
            continue
        selected.append(task)
        seen_families.add(family)
        if len(selected) >= count:
            return selected
    for row in task_rows:
        task = row["task"]
        if task in selected:
            continue
        selected.append(task)
        if len(selected) >= count:
            break
    return selected


def shorten_task_name(value: str, max_len: int = 46) -> str:
    value = parse_task_name(value)
    if len(value) <= max_len:
        return value
    head = value[: max_len - 12].rstrip("-_")
    tail = value[-10:]
    return f"{head}…{tail}"


def svg_bar_chart(title: str, labels: list[str], values: list[float], value_formatter) -> str:
    width = 900
    height = max(240, 36 * len(labels) + 80)
    margin_left = 320
    margin_right = 80
    margin_top = 45
    bar_height = 22
    gap = 10
    plot_width = width - margin_left - margin_right
    max_value = max(values) if values else 1.0
    if max_value <= 0:
        max_value = 1.0
    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='chart'>",
        f"<text x='20' y='28' class='chart-title'>{html.escape(title)}</text>",
    ]
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + idx * (bar_height + gap)
        bar_w = (value / max_value) * plot_width
        safe_label = shorten_task_name(label)
        parts.append(
            f"<text x='{margin_left - 12}' y='{y + 16}' text-anchor='end' class='axis-label'><title>{html.escape(label)}</title>{html.escape(safe_label)}</text>"
        )
        parts.append(f"<rect x='{margin_left}' y='{y}' width='{bar_w:.2f}' height='{bar_height}' rx='4' fill='#4a90e2' />")
        parts.append(
            f"<text x='{margin_left + bar_w + 8:.2f}' y='{y + 16}' class='value-label'>{html.escape(value_formatter(value))}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_stacked_horizontal_bar_chart(
    title: str,
    rows: list[tuple[str, list[tuple[str, float, str]]]],
    value_formatter,
    *,
    max_total: float | None = None,
    tick_values: list[float] | None = None,
) -> str:
    width = 980
    row_height = 44
    height = max(220, 140 + row_height * len(rows))
    margin_left = 320
    margin_right = 70
    margin_top = 80
    margin_bottom = 56
    computed_max_total = max((sum(value for _, value, _ in segments) for _, segments in rows), default=1.0)
    if computed_max_total <= 0:
        computed_max_total = 1.0
    if max_total is None:
        max_total = computed_max_total
    plot_width = width - margin_left - margin_right

    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='chart'>",
        f"<text x='20' y='28' class='chart-title'>{html.escape(title)}</text>",
    ]

    legend_x = margin_left
    legend_y = 36
    if rows:
        for seg_name, _, color in rows[0][1]:
            if legend_x > width - margin_right - 130:
                legend_x = margin_left
                legend_y += 22
            parts.append(f"<rect x='{legend_x}' y='{legend_y}' width='12' height='12' fill='{color}' />")
            parts.append(f"<text x='{legend_x + 18}' y='{legend_y + 10}' class='axis-label'>{html.escape(seg_name)}</text>")
            legend_x += 130

    for idx, (label, segments) in enumerate(rows):
        y = margin_top + idx * row_height
        safe_label = shorten_task_name(label, max_len=34)
        parts.append(
            f"<text x='{margin_left - 12}' y='{y + 16}' text-anchor='end' class='axis-label'><title>{html.escape(label)}</title>{html.escape(safe_label)}</text>"
        )
        cursor = margin_left
        total = sum(value for _, value, _ in segments)
        for _, value, color in segments:
            bar_w = (value / max_total) * plot_width
            parts.append(f"<rect x='{cursor:.2f}' y='{y}' width='{max(2.0, bar_w):.2f}' height='22' rx='4' fill='{color}' />")
            cursor += bar_w
        parts.append(f"<text x='{margin_left + (total / max_total) * plot_width + 8:.2f}' y='{y + 16}' class='value-label'>{html.escape(value_formatter(total))}</text>")

    parts.append(f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='#94a3b8' stroke-width='1' />")
    if tick_values:
        for tick in tick_values:
            if tick < 0 or tick > max_total:
                continue
            x = margin_left + (tick / max_total) * plot_width
            parts.append(f"<line x1='{x:.2f}' y1='{margin_top - 6}' x2='{x:.2f}' y2='{height - margin_bottom}' stroke='#e5e7eb' stroke-width='1' />")
            parts.append(f"<text x='{x:.2f}' y='{height - 12}' text-anchor='middle' class='axis-label'>{html.escape(value_formatter(tick))}</text>")
    else:
        parts.append(f"<text x='{margin_left}' y='{height - 12}' class='axis-label'>0</text>")
        parts.append(f"<text x='{width - margin_right}' y='{height - 12}' text-anchor='end' class='axis-label'>{html.escape(value_formatter(max_total))}</text>")
    parts.append("</svg>")
    return "".join(parts)


def svg_histogram(title: str, values: list[float], bins: int, value_formatter) -> str:
    width = 900
    height = 320
    margin_left = 60
    margin_right = 20
    margin_top = 40
    margin_bottom = 40
    if not values:
        return f"<div class='section'><h3>{html.escape(title)}</h3><p class='muted'>No data.</p></div>"
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        maximum = minimum + 1.0
    bucket_width = (maximum - minimum) / bins
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, int((value - minimum) / bucket_width))
        counts[idx] += 1
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_count = max(counts) if counts else 1
    bar_width = plot_width / bins
    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='chart'>",
        f"<text x='20' y='24' class='chart-title'>{html.escape(title)}</text>",
    ]
    for idx, count in enumerate(counts):
        x = margin_left + idx * bar_width
        bar_h = 0 if max_count == 0 else (count / max_count) * plot_height
        y = margin_top + plot_height - bar_h
        parts.append(f"<rect x='{x + 1:.2f}' y='{y:.2f}' width='{bar_width - 2:.2f}' height='{bar_h:.2f}' fill='#7ed321' />")
    parts.append(f"<text x='{margin_left}' y='{height - 12}' class='axis-label'>{html.escape(value_formatter(minimum))}</text>")
    parts.append(
        f"<text x='{width - margin_right}' y='{height - 12}' text-anchor='end' class='axis-label'>{html.escape(value_formatter(maximum))}</text>"
    )
    parts.append(f"<text x='{margin_left}' y='{margin_top - 8}' class='axis-label'>count max {max_count}</text>")
    parts.append("</svg>")
    return "".join(parts)


def stats_row_from_seconds(values: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    if not values:
        return None, None, None, None
    return mean(values), percentile(values, 0.50), percentile(values, 0.90), percentile(values, 0.99)


def svg_box_plot_comparison(
    title: str,
    series: list[tuple[str, list[float]]],
    value_formatter,
    *,
    min_x: float | None = None,
    max_x: float | None = None,
    tick_values: list[float] | None = None,
) -> str:
    width = 980
    row_height = 56
    height = max(220, 90 + row_height * len(series))
    margin_left = 320
    margin_right = 60
    margin_top = 48
    margin_bottom = 36
    all_values = [value for _, values in series for value in values]
    if not all_values:
        return f"<div class='section'><h3>{html.escape(title)}</h3><p class='muted'>No data.</p></div>"

    minimum = min(all_values) if min_x is None else min_x
    maximum = max(all_values) if max_x is None else max_x
    if maximum == minimum:
        maximum = minimum + 1.0
    plot_width = width - margin_left - margin_right

    def sx(value: float) -> float:
        return margin_left + ((value - minimum) / (maximum - minimum)) * plot_width

    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='chart'>",
        f"<text x='20' y='28' class='chart-title'>{html.escape(title)}</text>",
        f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='#94a3b8' stroke-width='1' />",
    ]
    if tick_values:
        for tick in tick_values:
            if tick < minimum or tick > maximum:
                continue
            x = sx(tick)
            parts.append(f"<line x1='{x:.2f}' y1='{margin_top - 6}' x2='{x:.2f}' y2='{height - margin_bottom}' stroke='#e5e7eb' stroke-width='1' />")
            parts.append(f"<text x='{x:.2f}' y='{height - 12}' text-anchor='middle' class='axis-label'>{html.escape(value_formatter(tick))}</text>")
    else:
        parts.append(f"<text x='{margin_left}' y='{height - 12}' class='axis-label'>{html.escape(value_formatter(minimum))}</text>")
        parts.append(f"<text x='{width - margin_right}' y='{height - 12}' text-anchor='end' class='axis-label'>{html.escape(value_formatter(maximum))}</text>")

    for idx, (label, values) in enumerate(series):
        ordered = sorted(values)
        q1 = percentile(ordered, 0.25) or 0.0
        p50 = percentile(ordered, 0.50) or 0.0
        q3 = percentile(ordered, 0.75) or 0.0
        low = ordered[0]
        high = ordered[-1]
        y = margin_top + idx * row_height + 18
        box_top = y - 11
        box_height = 22
        whisker_y = y
        safe_label = shorten_task_name(label, max_len=34)
        parts.append(
            f"<text x='{margin_left - 12}' y='{y + 4}' text-anchor='end' class='axis-label'><title>{html.escape(label)}</title>{html.escape(safe_label)}</text>"
        )
        parts.append(f"<line x1='{sx(low):.2f}' y1='{whisker_y:.2f}' x2='{sx(high):.2f}' y2='{whisker_y:.2f}' stroke='#64748b' stroke-width='2' />")
        parts.append(f"<line x1='{sx(low):.2f}' y1='{y - 8:.2f}' x2='{sx(low):.2f}' y2='{y + 8:.2f}' stroke='#64748b' stroke-width='2' />")
        parts.append(f"<line x1='{sx(high):.2f}' y1='{y - 8:.2f}' x2='{sx(high):.2f}' y2='{y + 8:.2f}' stroke='#64748b' stroke-width='2' />")
        parts.append(
            f"<rect x='{sx(q1):.2f}' y='{box_top:.2f}' width='{max(2.0, sx(q3)-sx(q1)):.2f}' height='{box_height}' rx='5' fill='#dbeafe' stroke='#2563eb' stroke-width='2' />"
        )
        parts.append(f"<line x1='{sx(p50):.2f}' y1='{box_top:.2f}' x2='{sx(p50):.2f}' y2='{box_top + box_height:.2f}' stroke='#1d4ed8' stroke-width='3' />")
        label_x = min(width - margin_right - 6, sx(high) + 8)
        parts.append(f"<text x='{label_x:.2f}' y='{y + 4:.2f}' class='value-label'>p50 {html.escape(value_formatter(p50))}</text>")

    parts.append("</svg>")
    return "".join(parts)


def svg_multi_line_chart(title: str, series: list[tuple[str, str, list[tuple[float, float]]]], x_formatter=None) -> str:
    width = 960
    height = 360
    margin_left = 60
    margin_right = 30
    margin_top = 40
    margin_bottom = 40
    all_points = [point for _, _, pts in series for point in pts]
    if not all_points:
        return f"<div class='section'><h3>{html.escape(title)}</h3><p class='muted'>No data.</p></div>"
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = 0.0
    max_y = max(point[1] for point in all_points)
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def sx(x: float) -> float:
        return margin_left + ((x - min_x) / (max_x - min_x)) * plot_width

    def sy(y: float) -> float:
        return margin_top + plot_height - ((y - min_y) / (max_y - min_y)) * plot_height

    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='chart'>",
        f"<text x='20' y='24' class='chart-title'>{html.escape(title)}</text>",
        f"<line x1='{margin_left}' y1='{margin_top + plot_height}' x2='{width - margin_right}' y2='{margin_top + plot_height}' stroke='#888' stroke-width='1' />",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' stroke='#888' stroke-width='1' />",
    ]

    legend_x = margin_left
    for name, color, points in series:
        sampled = sample_points(points, 300)
        path = " ".join(
            ("M" if idx == 0 else "L") + f" {sx(x):.2f} {sy(y):.2f}"
            for idx, (x, y) in enumerate(sampled)
        )
        parts.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2' />")
        safe_name = shorten_task_name(name, max_len=26)
        parts.append(f"<rect x='{legend_x}' y='{height - 20}' width='12' height='12' fill='{color}' />")
        parts.append(
            f"<text x='{legend_x + 18}' y='{height - 10}' class='axis-label'><title>{html.escape(name)}</title>{html.escape(safe_name)}</text>"
        )
        legend_x += 160

    if x_formatter is None:
        x_formatter = lambda x: f"{x:.1f}"
    parts.append(f"<text x='{margin_left}' y='{height - 28}' class='axis-label'>{html.escape(x_formatter(min_x))}</text>")
    parts.append(
        f"<text x='{width - margin_right}' y='{height - 28}' text-anchor='end' class='axis-label'>{html.escape(x_formatter(max_x))}</text>"
    )
    parts.append(f"<text x='{margin_left - 8}' y='{margin_top + 12}' text-anchor='end' class='axis-label'>{fmt_num(max_y)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def svg_multi_line_chart_annotated(
    title: str,
    series: list[tuple[str, str, list[tuple[float, float]]]],
    *,
    x_formatter,
    y_formatter,
    x_label: str,
    y_label: str,
    x_tick_values: list[float] | None = None,
    show_legend: bool = True,
) -> str:
    width = 960
    height = 360
    margin_left = 72
    margin_right = 28
    margin_top = 40
    margin_bottom = 56
    all_points = [point for _, _, pts in series for point in pts]
    if not all_points:
        return f"<div class='section'><h3>{html.escape(title)}</h3><p class='muted'>No data.</p></div>"
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = 0.0
    max_y = max(point[1] for point in all_points)
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def sx(x: float) -> float:
        return margin_left + ((x - min_x) / (max_x - min_x)) * plot_width

    def sy(y: float) -> float:
        return margin_top + plot_height - ((y - min_y) / (max_y - min_y)) * plot_height

    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='chart'>",
        f"<text x='20' y='24' class='chart-title'>{html.escape(title)}</text>",
        f"<line x1='{margin_left}' y1='{margin_top + plot_height}' x2='{width - margin_right}' y2='{margin_top + plot_height}' stroke='#94a3b8' stroke-width='1' />",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' stroke='#94a3b8' stroke-width='1' />",
        f"<text x='{margin_left + plot_width/2:.2f}' y='{height - 16}' text-anchor='middle' class='axis-label'>{html.escape(x_label)}</text>",
        f"<text x='18' y='{margin_top + plot_height/2:.2f}' text-anchor='middle' class='axis-label' transform='rotate(-90 18 {margin_top + plot_height/2:.2f})'>{html.escape(y_label)}</text>",
    ]

    tick_values = x_tick_values or [min_x, max_x]
    for tick in tick_values:
        if tick < min_x or tick > max_x:
            continue
        x = sx(tick)
        parts.append(f"<line x1='{x:.2f}' y1='{margin_top}' x2='{x:.2f}' y2='{margin_top + plot_height}' stroke='#e5e7eb' stroke-width='1' />")
        parts.append(f"<text x='{x:.2f}' y='{height - 30}' text-anchor='middle' class='axis-label'>{html.escape(x_formatter(tick))}</text>")

    y_ticks = [0.0, max_y / 2.0, max_y]
    for tick in y_ticks:
        y = sy(tick)
        parts.append(f"<line x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' stroke='#eef2f7' stroke-width='1' />")
        parts.append(f"<text x='{margin_left - 8}' y='{y + 4:.2f}' text-anchor='end' class='axis-label'>{html.escape(y_formatter(tick))}</text>")

    if show_legend:
        legend_x = margin_left
        for name, color, _ in series:
            parts.append(f"<rect x='{legend_x}' y='{height - 18}' width='12' height='12' fill='{color}' />")
            parts.append(f"<text x='{legend_x + 18}' y='{height - 8}' class='axis-label'>{html.escape(name)}</text>")
            legend_x += 150

    for _, color, points in series:
        sampled = sample_points(points, 300)
        path = " ".join(
            ("M" if idx == 0 else "L") + f" {sx(x):.2f} {sy(y):.2f}"
            for idx, (x, y) in enumerate(sampled)
        )
        parts.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2' />")

    parts.append("</svg>")
    return "".join(parts)


def infer_bursts(
    points: list[tuple[float, float]],
    threshold: float = 1.0,
    *,
    min_gap: float = 4.0,
    min_duration: float = 2.0,
) -> list[tuple[float, float]]:
    bursts: list[tuple[float, float]] = []
    start: float | None = None
    end: float | None = None
    for x, y in points:
        if y > threshold:
            if start is None:
                start = x
            end = x
        elif start is not None and end is not None:
            bursts.append((start, end))
            start = None
            end = None
    if start is not None and end is not None:
        bursts.append((start, end))
    if not bursts:
        return []
    merged: list[tuple[float, float]] = [bursts[0]]
    for start, end in bursts[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= min_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return [(start, end) for start, end in merged if (end - start) >= min_duration]


def svg_story_concurrency_figure(
    title: str,
    llm_points: list[tuple[float, float]],
    tool_points: list[tuple[float, float]],
) -> str:
    width = 1120
    panel_height = 220
    gap = 56
    header_h = 64
    footer_h = 48
    height = header_h + panel_height * 2 + gap + footer_h
    margin_left = 96
    margin_right = 24
    plot_width = width - margin_left - margin_right
    all_x = [x for x, _ in llm_points + tool_points]
    min_x = min(all_x) if all_x else 0.0
    max_x = max(all_x) if all_x else 1.0
    max_llm = max((y for _, y in llm_points), default=1.0)
    max_tool = max((y for _, y in tool_points), default=1.0)
    if max_llm <= 0:
        max_llm = 1.0
    if max_tool <= 0:
        max_tool = 1.0

    def sx(x: float) -> float:
        return margin_left + ((x - min_x) / (max_x - min_x or 1.0)) * plot_width

    def panel_path(points: list[tuple[float, float]], y_base: float, y_scale: float) -> tuple[str, str]:
        if not points:
            return "", ""
        area = [f"M {sx(points[0][0]):.2f} {y_base:.2f}"]
        line = []
        sampled = sample_points(points, 900)
        for idx, (x, y) in enumerate(sampled):
            py = y_base - (y / y_scale) * (panel_height - 28)
            area.append(f"L {sx(x):.2f} {py:.2f}")
            line.append(("M" if idx == 0 else "L") + f" {sx(x):.2f} {py:.2f}")
        area.append(f"L {sx(sampled[-1][0]):.2f} {y_base:.2f} Z")
        return " ".join(area), " ".join(line)

    top_base = header_h + panel_height
    bottom_base = header_h + panel_height + gap + panel_height
    llm_area, llm_line = panel_path(llm_points, top_base, max_llm)
    tool_area, tool_line = panel_path(tool_points, bottom_base, max_tool)
    llm_peak_y = top_base - (panel_height - 28)
    tool_peak_y = bottom_base - (panel_height - 28)

    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='chart'>",
        f"<text x='20' y='36' style='font-size:28px;font-weight:700;fill:#111827'>{html.escape(title)}</text>",
        f"<line x1='{margin_left}' y1='{top_base:.2f}' x2='{width - margin_right}' y2='{top_base:.2f}' stroke='#94a3b8' stroke-width='1' />",
        f"<line x1='{margin_left}' y1='{bottom_base:.2f}' x2='{width - margin_right}' y2='{bottom_base:.2f}' stroke='#94a3b8' stroke-width='1' />",
    ]
    for tick in range(0, int(math.floor(max_x / 5.0) * 5) + 1, 5):
        x = sx(float(tick))
        parts.append(f"<line x1='{x:.2f}' y1='{header_h}' x2='{x:.2f}' y2='{height - footer_h}' stroke='#e5e7eb' stroke-width='1' />")
        parts.append(f"<text x='{x:.2f}' y='{height - 18}' text-anchor='middle' class='axis-label'>{tick}</text>")
    parts.append(f"<text x='{margin_left + plot_width/2:.2f}' y='{height - 2:.2f}' text-anchor='middle' class='axis-label'>Time (minutes)</text>")

    if llm_area:
        parts.append(f"<text x='{margin_left + plot_width/2:.2f}' y='{header_h - 10:.2f}' text-anchor='middle' class='chart-title'>Concurrent LLM Generations Over Time</text>")
        parts.append(f"<path d='{llm_area}' fill='#7ed321' opacity='0.38' />")
        parts.append(f"<path d='{llm_line}' fill='none' stroke='#2faa3d' stroke-width='2' />")
        llm_dash_y = top_base - (max_llm / max_llm) * (panel_height - 28)
        parts.append(f"<line x1='{margin_left}' y1='{llm_dash_y:.2f}' x2='{width - margin_right}' y2='{llm_dash_y:.2f}' stroke='#2faa3d' stroke-dasharray='6 4' stroke-width='1.5' opacity='0.5' />")
        parts.append(f"<text x='{width - margin_right - 6}' y='{llm_dash_y - 6:.2f}' text-anchor='end' class='axis-label'>Peak: {int(round(max_llm))}</text>")
        parts.append(f"<text x='24' y='{header_h + panel_height/2:.2f}' text-anchor='middle' class='axis-label' transform='rotate(-90 24 {header_h + panel_height/2:.2f})'>Inflight LLM generations</text>")

    if tool_area:
        parts.append(f"<text x='{margin_left + plot_width/2:.2f}' y='{header_h + panel_height + gap - 10:.2f}' text-anchor='middle' class='chart-title'>Concurrent Tool Executions Over Time</text>")
        parts.append(f"<path d='{tool_area}' fill='#93c5fd' opacity='0.45' />")
        parts.append(f"<path d='{tool_line}' fill='none' stroke='#1d75bc' stroke-width='2' />")
        tool_dash_y = bottom_base - (max_tool / max_tool) * (panel_height - 28)
        parts.append(f"<line x1='{margin_left}' y1='{tool_dash_y:.2f}' x2='{width - margin_right}' y2='{tool_dash_y:.2f}' stroke='#1d75bc' stroke-dasharray='6 4' stroke-width='1.5' opacity='0.5' />")
        parts.append(f"<text x='{width - margin_right - 6}' y='{tool_dash_y - 6:.2f}' text-anchor='end' class='axis-label'>Peak: {int(round(max_tool))}</text>")
        parts.append(f"<text x='24' y='{header_h + panel_height + gap + panel_height/2:.2f}' text-anchor='middle' class='axis-label' transform='rotate(-90 24 {header_h + panel_height + gap + panel_height/2:.2f})'>Inflight tool executions</text>")

    bursts = infer_bursts(llm_points, threshold=max_llm * 0.08, min_gap=6.0, min_duration=3.0)
    training_windows: list[tuple[float, float]] = []
    for idx, (_, end) in enumerate(bursts):
        start = end
        next_start = bursts[idx + 1][0] if idx + 1 < len(bursts) else max_x
        if next_start - start >= 3.0:
            training_windows.append((start, next_start))
    for idx, (start, end) in enumerate(training_windows[:3], start=1):
        label_x = sx((start + end) / 2.0)
        label_y = header_h + panel_height + gap / 2 + 8
        parts.append(f"<text x='{label_x:.2f}' y='{label_y:.2f}' text-anchor='middle' style='font-size:13px;fill:#111827;font-weight:600'>Policy training step {idx}</text>")

    parts.append("</svg>")
    return "".join(parts)


def svg_small_multiple_dual_line_charts(
    title: str,
    rows: list[tuple[str, list[tuple[float, float]], list[tuple[float, float]]]],
) -> str:
    charts = []
    for label, prompt_points, completion_points in rows:
        max_turn = max([x for x, _ in prompt_points + completion_points], default=0)
        tick_max = int(math.floor(max_turn / 5.0) * 5)
        tick_values = [float(v) for v in range(0, tick_max + 1, 5)] if tick_max >= 0 else [0.0]
        charts.append(
            svg_multi_line_chart_annotated(
                shorten_task_name(label, max_len=40),
                [
                    ("Prompt tokens", "#2563eb", prompt_points),
                    ("Completion tokens", "#dc2626", completion_points),
                ],
                x_formatter=lambda x: str(int(x)),
                y_formatter=lambda y: f"{int(round(y))}",
                x_label="Turn",
                y_label="Tokens",
                x_tick_values=tick_values,
            )
        )
    return (
        f"<h3>{html.escape(title)}</h3>"
        + "<div class='chart-grid'>"
        + "".join(f"<div>{chart}</div>" for chart in charts)
        + "</div>"
    )


def svg_small_multiple_stacked_bar_turn_charts(
    title: str,
    rows: list[tuple[str, list[tuple[int, float, float]]]],
) -> str:
    def one_chart(label: str, values: list[tuple[int, float, float]]) -> str:
        width = 960
        height = 360
        margin_left = 72
        margin_right = 24
        margin_top = 40
        margin_bottom = 56
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        if not values:
            return f"<div><p class='muted'>No data for {html.escape(label)}</p></div>"
        max_turn = max(turn for turn, _, _ in values)
        max_total = max(prompt + completion for _, prompt, completion in values)
        if max_total <= 0:
            max_total = 1.0
        x_count = max_turn + 1
        bar_slot = plot_width / max(1, x_count)
        bar_width = max(2.0, min(14.0, bar_slot * 0.8))

        def sx(turn: int) -> float:
            return margin_left + turn * bar_slot + (bar_slot - bar_width) / 2.0

        def sy(tokens: float) -> float:
            return margin_top + plot_height - (tokens / max_total) * plot_height

        parts = [
            f"<svg viewBox='0 0 {width} {height}' class='chart'>",
            f"<text x='20' y='24' class='chart-title'>{html.escape(shorten_task_name(label, max_len=40))}</text>",
            f"<line x1='{margin_left}' y1='{margin_top + plot_height}' x2='{width - margin_right}' y2='{margin_top + plot_height}' stroke='#94a3b8' stroke-width='1' />",
            f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' stroke='#94a3b8' stroke-width='1' />",
            f"<text x='{margin_left + plot_width/2:.2f}' y='{height - 16}' text-anchor='middle' class='axis-label'>Turn</text>",
            f"<text x='18' y='{margin_top + plot_height/2:.2f}' text-anchor='middle' class='axis-label' transform='rotate(-90 18 {margin_top + plot_height/2:.2f})'>Tokens</text>",
            f"<rect x='{margin_left}' y='{height - 18}' width='12' height='12' fill='#2563eb' />",
            f"<text x='{margin_left + 18}' y='{height - 8}' class='axis-label'>Prompt tokens</text>",
            f"<rect x='{margin_left + 150}' y='{height - 18}' width='12' height='12' fill='#dc2626' />",
            f"<text x='{margin_left + 168}' y='{height - 8}' class='axis-label'>Completion tokens</text>",
        ]

        for tick in range(0, max_turn + 1, 5):
            x = margin_left + tick * bar_slot + bar_slot / 2.0
            parts.append(f"<line x1='{x:.2f}' y1='{margin_top}' x2='{x:.2f}' y2='{margin_top + plot_height}' stroke='#eef2f7' stroke-width='1' />")
            parts.append(f"<text x='{x:.2f}' y='{height - 30}' text-anchor='middle' class='axis-label'>{tick}</text>")

        for tick in [0.0, max_total / 2.0, max_total]:
            y = sy(tick)
            parts.append(f"<line x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' stroke='#eef2f7' stroke-width='1' />")
            parts.append(f"<text x='{margin_left - 8}' y='{y + 4:.2f}' text-anchor='end' class='axis-label'>{int(round(tick))}</text>")

        for turn, prompt, completion in values:
            x = sx(turn)
            prompt_h = (prompt / max_total) * plot_height
            completion_h = (completion / max_total) * plot_height
            y_base = margin_top + plot_height
            if prompt_h > 0:
                parts.append(f"<rect x='{x:.2f}' y='{y_base - prompt_h:.2f}' width='{bar_width:.2f}' height='{prompt_h:.2f}' fill='#2563eb' rx='1' />")
            if completion_h > 0:
                parts.append(f"<rect x='{x:.2f}' y='{y_base - prompt_h - completion_h:.2f}' width='{bar_width:.2f}' height='{completion_h:.2f}' fill='#dc2626' rx='1' />")

        parts.append("</svg>")
        return "".join(parts)

    return (
        f"<h3>{html.escape(title)}</h3>"
        + "<div class='chart-grid'>"
        + "".join(f"<div>{one_chart(label, values)}</div>" for label, values in rows)
        + "</div>"
    )


def build_visual_summary(input_dir: Path) -> str:
    summary = read_json(input_dir / "summary.json")
    trajectory_rows = read_csv(input_dir / "trajectory_summary.csv")
    task_rows = read_csv(input_dir / "task_summary.csv")
    turn_rows = read_csv(input_dir / "turn_summary.csv")
    tool_rows = read_csv(input_dir / "tool_events.csv")

    concurrency_intervals_path = input_dir / "concurrency_intervals.csv"
    concurrency_summary_path = input_dir / "concurrency_summary.csv"
    concurrency_rows = read_csv(concurrency_intervals_path) if concurrency_intervals_path.is_file() else []
    concurrency_summary_rows = read_csv(concurrency_summary_path) if concurrency_summary_path.is_file() else []

    cards = [
        ("Trace span", fmt_us_as_s(summary.get("trace_span_us"))),
        ("Tasks", str(summary.get("task_count", "-"))),
        ("Trajectories", str(summary.get("trajectory_count", "-"))),
        ("Normalized events", str(summary.get("event_count", "-"))),
        ("LLM events", str(summary.get("category_counts", {}).get("llm_generation", "-"))),
        ("Tool events", str(summary.get("category_counts", {}).get("tool_execution", "-"))),
    ]

    category_labels = ["llm_generation", "tool_execution", "framework_overhead", "queue_wait", "container_startup", "evaluation"]
    category_values = [float(summary.get("category_totals_us", {}).get(name, 0)) / 1_000_000.0 for name in category_labels]

    by_task_e2e_s: dict[str, list[float]] = defaultdict(list)
    by_task_llm_s: dict[str, list[float]] = defaultdict(list)
    by_task_tool_s: dict[str, list[float]] = defaultdict(list)
    by_task_turns: dict[str, list[int]] = defaultdict(list)
    by_task_rollouts: dict[str, int] = defaultdict(int)
    for row in trajectory_rows:
        task = row["task_name"]
        e2e_us = safe_float(row["e2e_wall_us"])
        llm_us = safe_float(row["llm_generation_us"])
        tool_us = safe_float(row["tool_execution_us"])
        turns = safe_int(row["turn_count"])
        if e2e_us is not None:
            by_task_e2e_s[task].append(e2e_us / 1_000_000.0)
        if llm_us is not None:
            by_task_llm_s[task].append(llm_us / 1_000_000.0)
        if tool_us is not None:
            by_task_tool_s[task].append(tool_us / 1_000_000.0)
        if turns is not None:
            by_task_turns[task].append(turns)
        by_task_rollouts[task] += 1

    task_e2e_rows = []
    for task, e2e_values in by_task_e2e_s.items():
        avg_e2e_s, p50_e2e_s, p90_e2e_s, p99_e2e_s = stats_row_from_seconds(e2e_values)
        avg_llm_s, _, _, _ = stats_row_from_seconds(by_task_llm_s[task])
        avg_tool_s, _, _, _ = stats_row_from_seconds(by_task_tool_s[task])
        avg_turns = mean([float(v) for v in by_task_turns[task]])
        task_e2e_rows.append(
            {
                "task": task,
                "rollouts": by_task_rollouts[task],
                "avg_e2e_s": avg_e2e_s or 0.0,
                "p50_e2e_s": p50_e2e_s or 0.0,
                "p90_e2e_s": p90_e2e_s or 0.0,
                "p99_e2e_s": p99_e2e_s or 0.0,
                "avg_llm_s": avg_llm_s or 0.0,
                "avg_tool_s": avg_tool_s or 0.0,
                "avg_turns": avg_turns or 0.0,
                "total_e2e_s": sum(e2e_values),
            }
        )
    task_e2e_rows.sort(key=lambda row: row["avg_e2e_s"], reverse=True)
    focus_task_names = select_focus_tasks(task_e2e_rows, count=5)
    top_task_e2e_rows = [row for row in task_e2e_rows if row["task"] in focus_task_names]
    top_task_box_plot = svg_box_plot_comparison(
        "Rollout E2E spread for the same 5 focus tasks",
        [(row["task"], by_task_e2e_s[row["task"]]) for row in top_task_e2e_rows],
        lambda v: f"{v:.0f}s",
    )

    tool_by_task: dict[str, list[float]] = defaultdict(list)
    tool_msg_len_by_task: dict[str, list[int]] = defaultdict(list)
    for row in tool_rows:
        task = row["task_name"]
        dur_us = safe_float(row["dur_us"])
        if dur_us is None:
            continue
        tool_by_task[task].append(dur_us)
        msg_len = safe_int(row.get("message_len_chars"))
        if msg_len is not None:
            tool_msg_len_by_task[task].append(msg_len)

    tool_share_pct_by_task: dict[str, list[float]] = defaultdict(list)
    for row in trajectory_rows:
        task = row["task_name"]
        e2e_us = safe_float(row["e2e_wall_us"]) or 0.0
        tool_us = safe_float(row["tool_execution_us"]) or 0.0
        if e2e_us > 0:
            tool_share_pct_by_task[task].append((tool_us / e2e_us) * 100.0)

    impacted_tasks = sorted(
        (
            (
                task,
                mean(values) or 0.0,
            )
            for task, values in tool_share_pct_by_task.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:8]
    tool_impact_box = svg_box_plot_comparison(
        "Tool time as % of rollout E2E for the same 5 focus tasks",
        [(task, tool_share_pct_by_task[task]) for task in focus_task_names if tool_share_pct_by_task.get(task)],
        lambda v: f"{v:.0f}%",
        min_x=0.0,
        max_x=50.0,
        tick_values=[float(v) for v in range(0, 51, 5)],
    )
    tool_msg_len_box = svg_box_plot_comparison(
        "Tool response length for the same 5 focus tasks",
        [(task, [float(v) for v in tool_msg_len_by_task.get(task, [])]) for task in focus_task_names if tool_msg_len_by_task.get(task)],
        lambda v: f"{v:.0f}",
    )

    llm_by_task_turn: dict[tuple[str, int], list[float]] = defaultdict(list)
    prompt_by_task_turn: dict[tuple[str, int], list[float]] = defaultdict(list)
    llm_by_task_total: dict[str, float] = defaultdict(float)
    turn_level_prompt_by_task: dict[str, list[float]] = defaultdict(list)
    turn_level_completion_by_task: dict[str, list[float]] = defaultdict(list)
    for row in turn_rows:
        turn = safe_int(row["turn"])
        completion = safe_float(row["completion_tokens_sum"])
        prompt = safe_float(row["prompt_tokens_sum"])
        task = row["task_name"]
        if turn is None:
            continue
        if completion is not None:
            llm_by_task_turn[(task, turn)].append(completion)
            llm_by_task_total[task] += completion
            turn_level_completion_by_task[task].append(completion)
        if prompt is not None:
            prompt_by_task_turn[(task, turn)].append(prompt)
            turn_level_prompt_by_task[task].append(prompt)

    prefill_decode_turn_rows = []
    for task in focus_task_names:
        turns = sorted(
            turn
            for (task_name, turn) in set(list(prompt_by_task_turn.keys()) + list(llm_by_task_turn.keys()))
            if task_name == task
        )
        turn_values = []
        for turn in turns:
            turn_values.append(
                (
                    int(turn),
                    mean(prompt_by_task_turn.get((task, turn), [])) or 0.0,
                    mean(llm_by_task_turn.get((task, turn), [])) or 0.0,
                )
            )
        prefill_decode_turn_rows.append((task, turn_values))

    tool_table = html_table(
        ["Task", "Rollouts", "Avg Tool Share", "p50 Tool Share", "p90 Tool Share", "Avg Tool Event"],
        [
            [
                parse_task_name(task),
                len(tool_share_pct_by_task.get(task, [])),
                f"{(mean(tool_share_pct_by_task.get(task, [])) or 0.0):.1f}%",
                f"{(percentile(tool_share_pct_by_task.get(task, []), 0.50) or 0.0):.1f}%",
                f"{(percentile(tool_share_pct_by_task.get(task, []), 0.90) or 0.0):.1f}%",
                fmt_us_as_ms(mean(values)),
            ]
            for task, values in sorted(tool_by_task.items(), key=lambda item: mean(tool_share_pct_by_task.get(item[0], [])) or 0.0, reverse=True)[:10]
        ],
    )

    if concurrency_rows:
        llm_concurrency_points = [
            ((safe_float(row["start_ts_us"]) - float(summary["first_ts_us"])) / 1_000_000.0 / 60.0, safe_float(row["active_llm_generation"]) or 0.0)
            for row in concurrency_rows
            if safe_float(row["start_ts_us"]) is not None
        ]
        tool_concurrency_points = [
            ((safe_float(row["start_ts_us"]) - float(summary["first_ts_us"])) / 1_000_000.0 / 60.0, safe_float(row["active_tool_execution"]) or 0.0)
            for row in concurrency_rows
            if safe_float(row["start_ts_us"]) is not None
        ]
        concurrency_series = [
            (
                "active_total",
                "#4a90e2",
                [
                    ((safe_float(row["start_ts_us"]) - float(summary["first_ts_us"])) / 1_000_000.0, safe_float(row["active_total"]) or 0.0)
                    for row in concurrency_rows
                    if safe_float(row["start_ts_us"]) is not None
                ],
            ),
            (
                "active_llm_generation",
                "#7ed321",
                [
                    ((safe_float(row["start_ts_us"]) - float(summary["first_ts_us"])) / 1_000_000.0, safe_float(row["active_llm_generation"]) or 0.0)
                    for row in concurrency_rows
                    if safe_float(row["start_ts_us"]) is not None
                ],
            ),
            (
                "active_tool_execution",
                "#e67e22",
                [
                    ((safe_float(row["start_ts_us"]) - float(summary["first_ts_us"])) / 1_000_000.0, safe_float(row["active_tool_execution"]) or 0.0)
                    for row in concurrency_rows
                    if safe_float(row["start_ts_us"]) is not None
                ],
            ),
            (
                "active_framework_overhead",
                "#9013fe",
                [
                    ((safe_float(row["start_ts_us"]) - float(summary["first_ts_us"])) / 1_000_000.0, safe_float(row["active_framework_overhead"]) or 0.0)
                    for row in concurrency_rows
                    if safe_float(row["start_ts_us"]) is not None
                ],
            ),
        ]
        concurrency_story_chart = svg_story_concurrency_figure(
            "GPU efficiency loss due to tool calling",
            llm_concurrency_points,
            tool_concurrency_points,
        )
        concurrency_chart = svg_multi_line_chart("Trace-level concurrency over time", concurrency_series, x_formatter=lambda x: f"{x:.0f}s")
        concurrency_table = html_table(
            ["Metric", "Mean active", "Max active", "Time-weighted p50", "Time-weighted p90"],
            [
                [
                    row["metric"],
                    fmt_num(safe_float(row["mean_active"])),
                    fmt_num(safe_float(row["max_active"])),
                    fmt_num(safe_float(row["time_weighted_p50_active"])),
                    fmt_num(safe_float(row["time_weighted_p90_active"])),
                ]
                for row in concurrency_summary_rows
                if row["metric"] in {"active_total", "active_llm_generation", "active_tool_execution", "active_framework_overhead"}
            ],
        )
    else:
        concurrency_story_chart = "<p class='muted'>No concurrency outputs found. Re-run the analyzer with the current V2 version.</p>"
        concurrency_chart = "<p class='muted'>No concurrency outputs found. Re-run the analyzer with the current V2 version.</p>"
        concurrency_table = ""
    concurrency_details = collapsible_table("Show table", concurrency_table) if concurrency_table else ""

    # Per-task E2E composition normalized to 100%
    by_task_total = defaultdict(float)
    by_task_llm = defaultdict(float)
    by_task_tool = defaultdict(float)
    by_task_env = defaultdict(float)
    by_task_framework = defaultdict(float)
    by_task_queue = defaultdict(float)
    by_task_eval = defaultdict(float)
    for row in trajectory_rows:
        task = row["task_name"]
        llm = safe_float(row["llm_generation_us"]) or 0.0
        tool = safe_float(row["tool_execution_us"]) or 0.0
        env = safe_float(row["container_startup_us"]) or 0.0
        framework = safe_float(row["framework_overhead_us"]) or 0.0
        queue = safe_float(row["queue_wait_us"]) or 0.0
        eval_time = safe_float(row["evaluation_us"]) or 0.0
        total = llm + tool + env + framework + queue + eval_time
        by_task_total[task] += total
        by_task_llm[task] += llm
        by_task_tool[task] += tool
        by_task_env[task] += env
        by_task_framework[task] += framework
        by_task_queue[task] += queue
        by_task_eval[task] += eval_time

    task_comp_rows = []
    task_comp_chart_rows = []
    for task, total in sorted(by_task_total.items(), key=lambda item: item[1], reverse=True):
        if total <= 0:
            continue
        llm_pct = by_task_llm[task] / total * 100.0
        tool_pct = by_task_tool[task] / total * 100.0
        env_pct = by_task_env[task] / total * 100.0
        framework_pct = by_task_framework[task] / total * 100.0
        queue_pct = by_task_queue[task] / total * 100.0
        eval_pct = by_task_eval[task] / total * 100.0
        task_comp_rows.append(
            [
                parse_task_name(task),
                f"{llm_pct:.1f}",
                f"{tool_pct:.1f}",
                f"{env_pct:.1f}",
                f"{framework_pct:.1f}",
                f"{queue_pct:.1f}",
                f"{eval_pct:.1f}",
            ]
        )
        task_comp_chart_rows.append(
            (
                task,
                [
                    ("Generation", llm_pct, "#2563eb"),
                    ("Tool", tool_pct, "#f59e0b"),
                    ("Env bringup", env_pct, "#10b981"),
                    ("Framework", framework_pct, "#8b5cf6"),
                    ("Queue wait", queue_pct, "#ef4444"),
                    ("Evaluation", eval_pct, "#94a3b8"),
                ],
            )
        )

    all_tasks_total = sum(by_task_total.values()) or 0.0
    if all_tasks_total > 0:
        all_tasks_llm = sum(by_task_llm.values()) / all_tasks_total * 100.0
        all_tasks_tool = sum(by_task_tool.values()) / all_tasks_total * 100.0
        all_tasks_env = sum(by_task_env.values()) / all_tasks_total * 100.0
        all_tasks_framework = sum(by_task_framework.values()) / all_tasks_total * 100.0
        all_tasks_queue = sum(by_task_queue.values()) / all_tasks_total * 100.0
        all_tasks_eval = sum(by_task_eval.values()) / all_tasks_total * 100.0
        task_comp_chart_rows = [
            (
                "All tasks average",
                [
                    ("Generation", all_tasks_llm, "#2563eb"),
                    ("Tool", all_tasks_tool, "#f59e0b"),
                    ("Env bringup", all_tasks_env, "#10b981"),
                    ("Framework", all_tasks_framework, "#8b5cf6"),
                    ("Queue wait", all_tasks_queue, "#ef4444"),
                    ("Evaluation", all_tasks_eval, "#94a3b8"),
                ],
            )
        ] + [row for row in task_comp_chart_rows if row[0] in focus_task_names]

    task_comp_chart = svg_stacked_horizontal_bar_chart(
        "Per-task E2E component breakdown",
        task_comp_chart_rows,
        lambda v: f"{v:.0f}%",
        max_total=100.0,
        tick_values=[float(v) for v in range(0, 101, 10)],
    )

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8' />",
        "<title>Multi-Turn Trace Visual Summary</title>",
        "<style>",
        ":root{--bg:#f4f6fb;--card:#ffffff;--ink:#111827;--muted:#6b7280;--accent:#0f62fe;--accent-2:#3b82f6;}",
        "body{font-family:'Space Grotesk','IBM Plex Sans','Segoe UI',sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:28px;}",
        "h1,h2,h3{margin:0 0 12px 0;} .muted{color:var(--muted);}",
        ".section{background:var(--card);border:1px solid #e7ecf3;border-radius:16px;padding:20px 22px;margin:0 0 18px 0;box-shadow:0 10px 30px rgba(15,23,42,0.06);}",
        ".section > * + *{margin-top:14px;}",
        ".card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:14px;}",
        ".card{background:linear-gradient(135deg,#f8fafc,#eef2ff);border-radius:14px;padding:14px 16px;border:1px solid #e3e8f0;} .card-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:0.04em;} .card-value{font-size:22px;font-weight:700;margin-top:6px;}",
        "table{border-collapse:collapse;width:100%;font-size:13px;} th,td{border-bottom:1px solid #edf1f6;padding:10px 12px;text-align:left;vertical-align:top;} th{background:#f8fafc;font-size:12px;letter-spacing:0.03em;text-transform:uppercase;color:var(--muted);}",
        "td.task-cell{max-width:320px;word-break:break-word;}",
        ".two-col{display:grid;grid-template-columns:1.1fr 1fr;gap:20px;} .chart{width:100%;height:auto;background:#fff;}",
        ".chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px;}",
        ".axis-label{font-size:11px;fill:#4b5563;} .chart-title{font-size:16px;font-weight:700;fill:#111827;} .value-label{font-size:11px;fill:#111827;}",
        ".table-details{margin-top:8px;}",
        ".table-details summary{display:inline-flex;align-items:center;gap:8px;cursor:pointer;list-style:none;padding:8px 12px;border:1px solid #d6deeb;border-radius:999px;background:#f8fafc;color:#334155;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;}",
        ".table-details summary::-webkit-details-marker{display:none;}",
        ".table-details[open] summary{background:#eef2ff;border-color:#bfd1ff;}",
        "@media (max-width: 1100px){.two-col,.chart-grid{grid-template-columns:1fr;}}",
        "</style></head><body>",
        "<div class='section'>",
        "<h2>Per-task E2E time composition</h2>"
        "<p class='muted'>Percent of time spent in generation, tool calls, environment bringup, framework work, queue wait, and evaluation. Each row sums to 100%.</p>"
        + task_comp_chart
        + collapsible_table("Show table", html_table(["Task", "Generation %", "Tool %", "Env bringup %", "Framework %", "Queue wait %", "Evaluation %"], task_comp_rows[:20]))
        + "</div>",
        "<div class='section'>",
        "<h2>Per-task E2E across rollouts</h2>"
        + "<p class='muted'>One row per task. Values are aggregated across that task's rollouts, so unlike tasks are not pooled into one global trajectory histogram.</p>"
        + collapsible_table(
            "Show table",
            html_table(
            ["Task", "Rollouts", "Avg E2E", "p50 E2E", "p90 E2E", "Avg LLM", "Avg Tool", "Avg Turns"],
            [
                [
                    parse_task_name(row["task"]),
                    row["rollouts"],
                    f"{row['avg_e2e_s']:.2f}s",
                    f"{row['p50_e2e_s']:.2f}s",
                    f"{row['p90_e2e_s']:.2f}s",
                    f"{row['avg_llm_s']:.2f}s",
                    f"{row['avg_tool_s']:.2f}s",
                    f"{row['avg_turns']:.1f}",
                ]
                for row in task_e2e_rows[:20]
            ],
        ))
        + "</div>",
        "<div class='section'>",
        "<h2>Rollout E2E box plots for the 5 highest-E2E tasks</h2>",
        "<p class='muted'>Each box plot shows rollout spread within one task: min/max whiskers, interquartile range, and median. The same 5 focus tasks are used across the task-specific sections below.</p>",
        top_task_box_plot,
        "</div>",
        "<div class='section'>",
        "<h2>Tool latency by task</h2>",
        "<p class='muted'>Raw tool latency matters, but the stronger signal is how much tool time consumes each rollout. This uses the same 5 focus tasks as the other task-specific sections.</p>",
        f"{tool_impact_box}",
        collapsible_table("Show table", tool_table),
        "</div>",
        "<div class='section'>",
        "<h2>Tool response length by task</h2>",
        "<p class='muted'>Box plots for tool response length on the same 5 focus tasks, so we can compare output-size spread directly against tool-impact spread.</p>",
        tool_msg_len_box,
        "</div>",
        "<div class='section'>",
        "<h2>Prefill vs decode work by turn</h2>",
        "<p class='muted'>Prompt tokens are the prefill proxy, and completion tokens are the response length for that same turn. These charts show per-task average tokens at each turn as stacked bars, with x-axis labels every 5 turns.</p>",
        svg_small_multiple_stacked_bar_turn_charts("Average prompt vs completion tokens by turn for the same top tasks", prefill_decode_turn_rows),
        "</div>",
        "<div class='section'>",
        "<h2>Trace-level concurrency</h2>"
        + concurrency_story_chart
        + concurrency_chart
        + concurrency_details
        + "</div>",
        "</body></html>",
    ]
    return "".join(html_parts)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_html = Path(args.output_html).resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    content = build_visual_summary(input_dir)
    output_html.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()

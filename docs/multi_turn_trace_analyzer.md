# Multi-Turn Trace Analyzer

`tools/multi_turn_trace_analyzer.py` packages the single-trace Chrome/Perfetto
analysis flow used for the multi-turn RL trace work. It reads one trace JSON and
emits normalized CSV, JSON, and Markdown outputs that are easier to compare
than browsing Perfetto directly.

`tools/multi_turn_trace_summary_visualizer.py` builds an HTML summary on top of
those analyzer outputs.

## CLI

Analyze one trace:

```bash
python tools/multi_turn_trace_analyzer.py \
  --trace /path/to/trace.json \
  --outdir /tmp/trace_out
```

Render the HTML summary:

```bash
python tools/multi_turn_trace_summary_visualizer.py \
  --input-dir /tmp/trace_out \
  --output-html /tmp/trace_summary.html
```

## Analyzer outputs

The analyzer writes:

- `events.csv`
- `tool_events.csv`
- `llm_events.csv`
- `trajectory_summary.csv`
- `task_summary.csv`
- `turn_summary.csv`
- `concurrency_intervals.csv`
- `concurrency_summary.csv`
- `summary.json`
- `report.md`

## Current analysis model

The packaged V2 flow focuses on:

- per-task E2E composition
- rollout E2E distribution within the focus tasks
- tool time as a percent of rollout E2E
- tool response length spread
- prompt vs completion token growth by turn
- trace-level concurrency for LLM generation and tool execution

## Tests

Unit tests for the packaged tools live in:

- `tests/unit/tools/test_multi_turn_trace_analyzer.py`
- `tests/unit/tools/test_multi_turn_trace_summary_visualizer.py`

Run them with:

```bash
pytest -q tests/unit/tools/test_multi_turn_trace_analyzer.py tests/unit/tools/test_multi_turn_trace_summary_visualizer.py
```

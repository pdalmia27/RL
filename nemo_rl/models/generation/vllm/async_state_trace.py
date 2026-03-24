import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


LEGACY_GENERATION_LOGGER_METRICS = (
    "inflight_batch_sizes",
    "num_pending_samples",
    "kv_cache_usage_perc",
    "generation_tokens",
)

ASYNC_STATE_TRACE_RAW_METRICS = (
    "decode_batch_sizes",
    "context_batch_sizes",
    "pending_requests",
    "avg_kv_slens",
    "p50_kv_slens",
    "p95_kv_slens",
    "max_kv_slens",
    "new_prefills_per_tick",
    "decode_tokens_per_tick",
)

ALL_GENERATION_LOGGER_METRICS = (
    *LEGACY_GENERATION_LOGGER_METRICS,
    *ASYNC_STATE_TRACE_RAW_METRICS,
)

WANDB_EXCLUDED_GENERATION_LOGGER_METRICS = frozenset(ASYNC_STATE_TRACE_RAW_METRICS)


def is_generation_state_metrics_enabled(vllm_cfg: Mapping[str, Any]) -> bool:
    return bool(
        vllm_cfg.get("async_engine", False)
        and (
            vllm_cfg.get("enable_vllm_metrics_logger", False)
            or vllm_cfg.get("enable_async_state_trace", False)
        )
    )


def get_generation_state_metrics_interval(vllm_cfg: Mapping[str, Any]) -> float:
    interval = vllm_cfg.get("async_state_trace_interval")
    if interval is None:
        interval = vllm_cfg.get("vllm_metrics_logger_interval")

    assert interval is not None, (
        "Either async_state_trace_interval or vllm_metrics_logger_interval must be "
        "set when generation state metrics are enabled"
    )
    interval = float(interval)
    assert interval > 0, (
        f"Generation state metrics interval must be a positive float, got {interval}"
    )
    return interval


def filter_generation_logger_metrics_for_wandb(
    generation_logger_metrics: Mapping[str, dict[int, list[Any]]],
) -> dict[str, dict[int, list[Any]]]:
    return {
        metric_name: metric_values
        for metric_name, metric_values in generation_logger_metrics.items()
        if metric_name not in WANDB_EXCLUDED_GENERATION_LOGGER_METRICS
    }


@dataclass
class AsyncStateTraceContext:
    step: int
    generation_pass_idx: int
    trace_dir: str


@dataclass
class AsyncStateRequestState:
    submit_time_s: float
    prompt_len: int
    sampled_output_len: int | None
    current_generated_tokens: int = 0
    first_token_seen: bool = False
    last_update_time_s: float = 0.0


def _empty_metric_series() -> dict[str, list[int | float]]:
    return {metric_name: [] for metric_name in ALL_GENERATION_LOGGER_METRICS}


def _compute_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


class AsyncStateTraceRecorder:
    def __init__(self, worker_idx: int):
        self.worker_idx = worker_idx
        self._context: AsyncStateTraceContext | None = None
        self._trace_path: str | None = None
        self._request_states: dict[str, AsyncStateRequestState] = {}
        self._metric_series = _empty_metric_series()
        self._prev_prefill_running_request_ids: set[str] = set()
        self._prev_generation_tokens_counter: int | None = None
        self._latest_generation_tokens_counter: int | None = None

    def set_context(
        self, step: int, generation_pass_idx: int, trace_dir: str
    ) -> None:
        self._context = AsyncStateTraceContext(
            step=step,
            generation_pass_idx=generation_pass_idx,
            trace_dir=trace_dir,
        )
        os.makedirs(trace_dir, exist_ok=True)
        self._trace_path = os.path.join(
            trace_dir, f"worker-idx{self.worker_idx}.jsonl"
        )

    def on_submit(
        self,
        request_id: str,
        prompt_len: int,
        sampled_output_len: int | None,
        wall_time_s: float | None = None,
    ) -> None:
        now = time.time() if wall_time_s is None else wall_time_s
        self._request_states[request_id] = AsyncStateRequestState(
            submit_time_s=now,
            prompt_len=prompt_len,
            sampled_output_len=sampled_output_len,
            last_update_time_s=now,
        )

    def on_output(
        self, request_id: str, current_generated_tokens: int, wall_time_s: float | None = None
    ) -> None:
        request_state = self._request_states.get(request_id)
        if request_state is None:
            return

        request_state.current_generated_tokens = current_generated_tokens
        request_state.first_token_seen = (
            request_state.first_token_seen or current_generated_tokens > 0
        )
        request_state.last_update_time_s = (
            time.time() if wall_time_s is None else wall_time_s
        )

    def on_finish(self, request_id: str) -> None:
        self._request_states.pop(request_id, None)

    def clear_step_local_metrics(self) -> None:
        self._metric_series = _empty_metric_series()
        self._prev_prefill_running_request_ids = set()
        self._prev_generation_tokens_counter = self._latest_generation_tokens_counter

    def get_metric_series(self) -> dict[str, list[int | float]]:
        return copy.deepcopy(self._metric_series)

    def sample(
        self,
        running_total: int,
        waiting_total: int,
        kv_cache_usage_perc: float,
        generation_tokens_counter: int,
        interval_s: float,
        wall_time_s: float | None = None,
    ) -> dict[str, Any] | None:
        now = time.time() if wall_time_s is None else wall_time_s
        self._latest_generation_tokens_counter = generation_tokens_counter

        decode_request_ids: list[str] = []
        zero_token_request_ids: list[str] = []
        request_items = sorted(
            self._request_states.items(), key=lambda item: item[1].submit_time_s
        )
        for request_id, request_state in request_items:
            if request_state.current_generated_tokens > 0:
                decode_request_ids.append(request_id)
            else:
                zero_token_request_ids.append(request_id)

        decode_batch_size = len(decode_request_ids)
        requested_context_batch_size = max(running_total - decode_batch_size, 0)
        prefill_running_ids = set(
            zero_token_request_ids[
                : min(requested_context_batch_size, len(zero_token_request_ids))
            ]
        )
        context_batch_size = len(prefill_running_ids)

        kv_lengths: list[float] = []
        for request_id, request_state in request_items:
            if request_id in prefill_running_ids:
                kv_lengths.append(float(request_state.prompt_len))
            elif request_state.current_generated_tokens > 0:
                kv_lengths.append(
                    float(
                        request_state.prompt_len + request_state.current_generated_tokens
                    )
                )

        avg_kv_slen = float(np.mean(kv_lengths)) if kv_lengths else 0.0
        p50_kv_slen = _compute_percentile(kv_lengths, 50.0)
        p95_kv_slen = _compute_percentile(kv_lengths, 95.0)
        max_kv_slen = float(max(kv_lengths)) if kv_lengths else 0.0

        previous_generation_tokens_counter = self._prev_generation_tokens_counter
        decode_tokens_this_tick = (
            max(generation_tokens_counter - previous_generation_tokens_counter, 0)
            if previous_generation_tokens_counter is not None
            else 0
        )
        self._prev_generation_tokens_counter = generation_tokens_counter

        new_prefills_this_tick = len(
            prefill_running_ids - self._prev_prefill_running_request_ids
        )
        self._prev_prefill_running_request_ids = prefill_running_ids

        self._metric_series["inflight_batch_sizes"].append(int(running_total))
        self._metric_series["num_pending_samples"].append(int(waiting_total))
        self._metric_series["kv_cache_usage_perc"].append(float(kv_cache_usage_perc))
        self._metric_series["generation_tokens"].append(int(generation_tokens_counter))
        self._metric_series["decode_batch_sizes"].append(int(decode_batch_size))
        self._metric_series["context_batch_sizes"].append(int(context_batch_size))
        self._metric_series["pending_requests"].append(int(waiting_total))
        self._metric_series["avg_kv_slens"].append(float(avg_kv_slen))
        self._metric_series["p50_kv_slens"].append(float(p50_kv_slen))
        self._metric_series["p95_kv_slens"].append(float(p95_kv_slen))
        self._metric_series["max_kv_slens"].append(float(max_kv_slen))
        self._metric_series["new_prefills_per_tick"].append(int(new_prefills_this_tick))
        self._metric_series["decode_tokens_per_tick"].append(
            int(decode_tokens_this_tick)
        )

        if self._context is None or self._trace_path is None:
            return None

        row = {
            "step": self._context.step,
            "generation_pass_idx": self._context.generation_pass_idx,
            "wall_time_s": float(now),
            "interval_s": float(interval_s),
            "worker_idx": int(self.worker_idx),
            "running_total": int(running_total),
            "waiting_total": int(waiting_total),
            "decode_batch_size": int(decode_batch_size),
            "context_batch_size": int(context_batch_size),
            "pending_requests": int(waiting_total),
            "avg_kv_slen": float(avg_kv_slen),
            "p50_kv_slen": float(p50_kv_slen),
            "p95_kv_slen": float(p95_kv_slen),
            "max_kv_slen": float(max_kv_slen),
            "new_prefills_this_tick": int(new_prefills_this_tick),
            "decode_tokens_this_tick": int(decode_tokens_this_tick),
            "kv_cache_usage_perc": float(kv_cache_usage_perc),
        }
        with open(self._trace_path, "a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(row, sort_keys=True) + "\n")
        return row

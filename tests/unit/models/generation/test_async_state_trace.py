import json

from nemo_rl.algorithms.utils import (
    log_generation_metrics_to_wandb,
    print_performance_metrics,
)
from nemo_rl.models.generation.vllm.async_state_trace import (
    AsyncStateTraceRecorder,
    filter_generation_logger_metrics_for_wandb,
)


def test_async_state_trace_recorder_rebases_generation_tokens_on_clear(tmp_path):
    recorder = AsyncStateTraceRecorder(worker_idx=3)
    recorder.set_context(step=2, generation_pass_idx=1, trace_dir=str(tmp_path))
    recorder.on_submit("req-a", prompt_len=16, sampled_output_len=64, wall_time_s=1.0)
    recorder.sample(
        running_total=1,
        waiting_total=0,
        kv_cache_usage_perc=12.5,
        generation_tokens_counter=100,
        interval_s=0.5,
        wall_time_s=2.0,
    )
    recorder.clear_step_local_metrics()
    recorder.on_submit("req-b", prompt_len=8, sampled_output_len=32, wall_time_s=3.0)
    recorder.on_output("req-b", current_generated_tokens=5, wall_time_s=3.2)

    recorder.sample(
        running_total=1,
        waiting_total=0,
        kv_cache_usage_perc=13.0,
        generation_tokens_counter=107,
        interval_s=0.5,
        wall_time_s=4.0,
    )

    metric_series = recorder.get_metric_series()
    assert metric_series["decode_tokens_per_tick"] == [7]
    assert metric_series["decode_batch_sizes"] == [1]
    assert metric_series["context_batch_sizes"] == [0]


def test_async_state_trace_recorder_writes_jsonl_and_classifies_prefill(tmp_path):
    recorder = AsyncStateTraceRecorder(worker_idx=5)
    recorder.set_context(step=7, generation_pass_idx=2, trace_dir=str(tmp_path))
    recorder.on_submit("older", prompt_len=20, sampled_output_len=80, wall_time_s=1.0)
    recorder.on_submit("decode", prompt_len=10, sampled_output_len=40, wall_time_s=2.0)
    recorder.on_output("decode", current_generated_tokens=6, wall_time_s=2.5)
    recorder.on_submit("queued", prompt_len=14, sampled_output_len=50, wall_time_s=3.0)

    recorder.sample(
        running_total=2,
        waiting_total=1,
        kv_cache_usage_perc=25.0,
        generation_tokens_counter=15,
        interval_s=0.5,
        wall_time_s=4.0,
    )

    trace_path = tmp_path / "worker-idx5.jsonl"
    assert trace_path.exists()

    row = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert row["step"] == 7
    assert row["generation_pass_idx"] == 2
    assert row["worker_idx"] == 5
    assert row["decode_batch_size"] == 1
    assert row["context_batch_size"] == 1
    assert row["pending_requests"] == 1
    assert row["new_prefills_this_tick"] == 1
    assert row["avg_kv_slen"] == 18.0
    assert row["max_kv_slen"] == 20.0


def test_filter_generation_logger_metrics_for_wandb_excludes_state_trace_series():
    filtered = filter_generation_logger_metrics_for_wandb(
        {
            "inflight_batch_sizes": {0: [1, 2]},
            "decode_batch_sizes": {0: [1, 1]},
            "pending_requests": {0: [0, 2]},
        }
    )
    assert "inflight_batch_sizes" in filtered
    assert "decode_batch_sizes" not in filtered
    assert "pending_requests" not in filtered


class _FakeLogger:
    def __init__(self):
        self.names: list[str] = []

    def log_plot_per_worker_timeline_metrics(self, metrics, step, prefix, name, timeline_interval):
        self.names.append(name)


def test_log_generation_metrics_to_wandb_filters_state_trace_series():
    fake_logger = _FakeLogger()
    log_generation_metrics_to_wandb(
        {
            "inflight_batch_sizes": {0: [1, 2]},
            "decode_batch_sizes": {0: [1, 1]},
        },
        step=3,
        timeline_interval=0.5,
        logger=fake_logger,
    )
    assert fake_logger.names == ["inflight_batch_sizes"]


def test_print_performance_metrics_emits_state_trace_summaries():
    master_config = {
        "policy": {
            "generation": {
                "vllm_cfg": {
                    "async_engine": True,
                    "enable_async_state_trace": True,
                    "async_state_trace_interval": 0.5,
                },
                "colocated": {
                    "enabled": True,
                    "resources": {"num_nodes": None, "gpus_per_node": None},
                },
            }
        },
        "cluster": {"num_nodes": 1, "gpus_per_node": 2},
        "grpo": {"num_prompts_per_step": 2, "num_generations_per_prompt": 4},
    }
    metrics = {
        "total_num_tokens": 128,
        "mean_total_tokens_per_sample": 16.0,
        "vllm_logger_metrics": {
            "inflight_batch_sizes": {0: [1, 2, 3]},
            "num_pending_samples": {0: [0, 1, 0]},
            "decode_batch_sizes": {0: [1, 2, 2]},
            "context_batch_sizes": {0: [1, 0, 1]},
            "pending_requests": {0: [0, 1, 0]},
            "avg_kv_slens": {0: [16.0, 24.0, 20.0]},
            "new_prefills_per_tick": {0: [1, 0, 1]},
            "decode_tokens_per_tick": {0: [8, 12, 10]},
        },
    }
    timing_metrics = {
        "policy_and_reference_logprobs": 1.0,
        "policy_training": 2.0,
        "total_step_time": 4.0,
        "generation": 1.5,
        "prepare_for_generation/total": 0.5,
    }

    performance_metrics = print_performance_metrics(
        train_results={},
        metrics=metrics,
        timing_metrics=timing_metrics,
        master_config=master_config,
    )

    assert performance_metrics["state_trace/decode_batch_size/mean"] == 5 / 3
    assert performance_metrics["state_trace/context_batch_size/max"] == 1.0
    assert performance_metrics["state_trace/pending_requests/max"] == 1.0
    assert performance_metrics["state_trace/avg_kv_slen/mean"] == 20.0
    assert performance_metrics["state_trace/decode_tokens_per_tick/max"] == 12.0

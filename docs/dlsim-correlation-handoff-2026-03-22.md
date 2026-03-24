# DLSim Correlation Handoff

## Summary

This note is a handoff for the NeMo-RL work needed to correlate DLSim async RL operating points with real silicon runs.

The original objective was not "just profiling." The objective was:

1. make NeMo-RL generate with the same output-length distribution that DLSim uses,
2. then make the real async silicon run observable in the same state space that DLSim uses,
3. then use that state matching to choose the right profiling windows.

The main conclusion so far is:

- the output-sequence-length distribution alignment is implemented and verified,
- the async state-trace instrumentation is implemented and unit-tested,
- the first 2-node async Llama runtime validation of the new state trace is now running.

## Initial Problem

DLSim does not simulate a single averaged async RL regime. It samples request output sequence lengths (OSLs), constructs rollout batches, and reasons about concurrency at multiple operating points. For the async RL correlation problem, the relevant DLSim state tuple is effectively:

- `gen_batch_size`
- `context_batch_size`
- `gen_kv_slen`
- `ifb_sample_weight`

across a small set of representative concurrency points.

The initial NeMo-RL gap was that real training runs were not using the same response-length distribution as the simulator, so any later attempt to compare DLSim operating points to real silicon would already be biased at the request-mix layer.

## What I Did First: Match The Output-Length Distribution

The first implementation target was to make NeMo-RL use a simulator-compatible output-length distribution on the async vLLM path.

The reasoning was:

1. for `L=1`, the first thing to align is the OSL distribution itself,
2. async vLLM already has a natural per-request max-token cap point,
3. if sampled target output lengths are not enforced, later concurrency-state matching is less meaningful.

The distribution work was implemented by copying the simulator's truncated-lognormal behavior into NeMo-RL, rather than importing the simulator directly at runtime.

### Files Changed For Distribution Alignment

- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/models/generation/output_length_samplers.py`
  - new helper implementing the DLSim-style truncated-lognormal output-length sampler
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/models/generation/interfaces.py`
  - generation config surface extended with `ignore_eos` and synthetic output-length config
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/models/generation/__init__.py`
  - generation config normalization updated so EOS is only auto-added when `ignore_eos` is false
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/models/generation/vllm/vllm_worker.py`
  - sync path rejects unsupported per-request distribution configs in v1
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/models/generation/vllm/vllm_worker_async.py`
  - async path samples a per-request output length and uses it to cap generation
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/experience/rollouts.py`
  - rollout metrics extended to persist sampled-vs-actual output-length summaries
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/algorithms/grpo.py`
  - console summary extended so synthetic output-length metrics are visible in the main training flow
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tests/unit/models/generation/test_synthetic_output_lengths.py`
  - focused unit coverage for the sampler/config path

### Why These Distribution Changes Were Made

- `output_length_samplers.py` exists so the simulator math is locally reproducible and version-controlled inside NeMo-RL.
- `interfaces.py` and `__init__.py` were changed so the feature is opt-in and explicit in config.
- `vllm_worker_async.py` was the real execution hook because async vLLM already has per-request generation control.
- `rollouts.py` and `grpo.py` were changed because proving the feature works requires sampled-vs-actual observability, not only code-path confidence.

## Verification Of Distribution Alignment

This part is already verified on the stock 2-node async Llama benchmark path.

### Constant-Cap Shakeout

Run artifacts:

- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tests/test_suites/llm/performance/grpo-llama3.1-8b-instruct-2n4g-async-1off-osl-const64-v3-20260319-162603/metrics.json`

Result:

- sampled output length mean, p50, p95, and max all matched actual generation length exactly at `64`
- clipping counters were zero

### Truncated-Lognormal Shakeout

Run artifacts:

- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tests/test_suites/llm/performance/grpo-llama3.1-8b-instruct-2n4g-async-1off-osl-trunclogn-v1-20260319-170208/metrics.json`

Result:

- sampled output-length summaries matched actual generation-length summaries exactly across the debug run
- clipping counters were zero

This means the first planned milestone is complete:

- NeMo-RL can now run the same kind of output-length distribution that DLSim expects, on a real async run.

## What The Plan Became After Distribution Matching

Once the distribution piece was verified, the next question was what to do before profiling.

The answer was: **state matching, not more profiling.**

The thought process was:

1. DLSim does not just define a distribution; it defines operating points in a state space.
2. The existing NeMo-RL metrics were too coarse to recover those operating points directly.
3. If we profile before the real run is observable in the same state space, the resulting trace windows are hard to interpret against the simulator.

So the new plan became:

1. keep the OSL distribution aligned,
2. add async state-trace observability to the real run,
3. later build offline matching from DLSim points to real runtime windows,
4. only after that choose targeted profiling windows.

The high-level plan was written here:

- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/docs/dlsim-silicon-correlation-plan-2026-03-20.md`

There is also a generated explainer artifact here:

- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/docs/async_state_trace_explainer.png`
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/docs/async_state_trace_explainer.json`

## What Was Implemented Next: Async State Trace

The next implementation was a structured async state trace built as an additive extension of the existing async vLLM metrics path.

The design goal was to make the real run observable in approximately the same state space as DLSim, using lightweight runtime telemetry rather than a profiler.

The canonical state trace records:

- `decode_batch_size`
- `context_batch_size`
- `pending_requests`
- `avg_kv_slen`
- `new_prefills_this_tick`
- `decode_tokens_this_tick`

with worker-local JSONL output and additive scalar summaries in the main performance metrics path.

### Files Changed For Async State Trace

- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/models/generation/vllm/async_state_trace.py`
  - new helper owning request-state bookkeeping, per-tick reconstruction, JSONL row creation, and step-local raw-series accumulation
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/models/generation/vllm/config.py`
  - config surface extended with:
    - `enable_async_state_trace`
    - `async_state_trace_interval`
    - `async_state_trace_dir`
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/models/generation/vllm/vllm_worker_async.py`
  - async worker now:
    - uses a combined enable predicate for legacy vLLM metrics logging and new state trace
    - tracks request lifecycle in `generate_async()`
    - samples one atomic row per tick
    - writes per-worker JSONL files
    - rebases counters on clear
    - stops the metrics thread cleanly on shutdown
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/models/generation/vllm/vllm_generation.py`
  - driver-side aggregation expanded to collect the new raw state-trace series
  - added `set_async_state_trace_context(step, generation_pass_idx, trace_dir)`
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/algorithms/grpo.py`
  - training loop injects:
    - `step`
    - `generation_pass_idx`
    - `trace_dir`
  - done immediately before per-generation logger-state clearing
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/algorithms/utils.py`
  - performance reporting extended with additive `state_trace/...` scalar summaries
  - W&B raw generation-metric logging filters out the new raw state-trace series to avoid noise
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tests/unit/models/generation/test_async_state_trace.py`
  - focused unit tests for recorder behavior, rebase semantics, JSONL output shape, and summary/logging filtering

### Why These State-Trace Changes Were Made

- `async_state_trace.py` exists to keep the core state-reconstruction logic pure and testable rather than burying everything inside `vllm_worker_async.py`.
- `vllm_worker_async.py` was the right hook because request lifecycle and the metrics thread both live there already.
- `vllm_generation.py` was changed so the driver can fetch and clear the expanded metrics payload without inventing a new transport path.
- `grpo.py` was changed because the worker-local JSONL rows need a meaningful `step` and `generation_pass_idx`.
- `utils.py` was changed so the state trace becomes visible in the same reporting surface as the existing performance metrics, without breaking the legacy visualizations.

## Verification Of Async State Trace So Far

Unit verification is complete.

Focused test file:

- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tests/unit/models/generation/test_async_state_trace.py`

Test command used:

- `uv run --group test pytest -q --noconftest /lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tests/unit/models/generation/test_async_state_trace.py`

Result:

- `5 passed`

This was run with `--noconftest` because the repo's local Ray test fixture is noisy on this host. The async state-trace tests are pure and did not require that fixture.

The production files also passed syntax compilation.

## Current Status Of The Overall Plan

### Complete

- output-length distribution alignment
- distribution verification on stock async Llama
- high-level DLSim-to-silicon plan
- async state-trace implementation
- async state-trace focused unit tests

### In Progress

- no longer in progress for Llama
- the first real runtime validation of the new async state trace on the stock 2-node async Llama benchmark is now complete
- the next in-progress item is deciding whether to clean up a small JSONL tail artifact before moving to DeepSeek

### Not Started Yet

- offline matching from DLSim operating points to real state-trace windows
- targeted profiling driven by those matched windows

## Runtime Validation Result

The final successful runtime validation run for Sub-Plan 1 is:

- job id: `1266175`
- job name: `coreai_dlalgo_llm:llama-async-state-trace`
- partition: `gb200`
- final state: `COMPLETED`

Primary run artifacts:

- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/slurm-1266175-llama-async-state-trace.out`
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/1266175-logs`
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tests/test_suites/llm/performance/grpo-llama3.1-8b-instruct-2n4g-async-1off-state-trace-20260322-165602`

Merged metrics:

- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tests/test_suites/llm/performance/grpo-llama3.1-8b-instruct-2n4g-async-1off-state-trace-20260322-165602/metrics.json`

Per-worker JSONL traces:

- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tests/test_suites/llm/performance/grpo-llama3.1-8b-instruct-2n4g-async-1off-state-trace-20260322-165602/logs/exp_001/async_state_trace/worker-idx0.jsonl`
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tests/test_suites/llm/performance/grpo-llama3.1-8b-instruct-2n4g-async-1off-state-trace-20260322-165602/logs/exp_001/async_state_trace/worker-idx1.jsonl`
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tests/test_suites/llm/performance/grpo-llama3.1-8b-instruct-2n4g-async-1off-state-trace-20260322-165602/logs/exp_001/async_state_trace/worker-idx2.jsonl`
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tests/test_suites/llm/performance/grpo-llama3.1-8b-instruct-2n4g-async-1off-state-trace-20260322-165602/logs/exp_001/async_state_trace/worker-idx3.jsonl`

### What Was Verified In This Runtime Run

1. `performance/state_trace/...` scalar summaries were present in `metrics.json`.
2. The driver printed the new `Async State Trace` block during training.
3. Four per-worker JSONL files were emitted under the expected shared log directory.
4. The JSONL row schema matched the intended design:
   - `step`
   - `generation_pass_idx`
   - `wall_time_s`
   - `interval_s`
   - `worker_idx`
   - `running_total`
   - `waiting_total`
   - `decode_batch_size`
   - `context_batch_size`
   - `pending_requests`
   - `avg_kv_slen`
   - `p50_kv_slen`
   - `p95_kv_slen`
   - `max_kv_slen`
   - `new_prefills_this_tick`
   - `decode_tokens_this_tick`
   - `kv_cache_usage_perc`
5. Recomputed summaries from JSONL matched the merged `performance/state_trace/...` metrics to numerical tolerance for the logged training steps.

### Important Fix That Made JSONL Work

The critical bug was not in the worker helper itself. It was in the async GRPO call sites in:

- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/algorithms/grpo.py`

The state-trace context had initially been wired into a sync-style generation block, but the successful benchmark path was using the async GRPO training loop. Once `set_async_state_trace_context(...)` was injected into the actual async clear/restart points, JSONL emission started working.

### Residual Quirk

The JSONL files contain a very small `step=4` tail:

- steps present in JSONL: `1, 2, 3, 4`
- steps present in merged metrics: `1, 2, 3`

This appears to come from the async post-refit path setting a new trace context after the final logged training step. It does not break the main use case, because:

- the real training steps `1, 2, 3` are fully traced and summarized,
- the `step=4` tail is very small,
- the JSONL files remain usable for offline matching.

This is a cleanup candidate, not a blocker.

## Recommended Next Steps

1. optionally clean up the small `step=4` JSONL tail in:
   - `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/algorithms/grpo.py`
2. move the same instrumentation unchanged to the reduced-work DeepSeek async path
3. once a stable DeepSeek state trace exists, add the offline DLSim-to-silicon matcher
4. only after the matcher exists, return to targeted NSYS profiling

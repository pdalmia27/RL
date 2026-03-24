# DLSim-to-Silicon Correlation Plan

## Summary

The goal is to correlate DLSim's 5 async RL operating points with real silicon runs.

The first prerequisite is already satisfied:

- NeMo-RL can now run the same output-length distribution on real async Llama.

The next step is **state matching**, not more profiling.

The plan is:

1. make the silicon run observable in the same state space as DLSim,
2. match DLSim points to real runtime windows offline,
3. only then collect targeted NSYS traces for those matched windows.

Chosen defaults:

- canonical trace sink: per-worker JSONL plus existing metrics summary
- state source for v1: heuristic reconstruction from existing vLLM gauges plus per-request lifecycle tracking
- first rollout target: small async Llama, then move the same logger unchanged to DeepSeek

## High-Level Phases

### Phase 1: Silicon State Observability

Add a lightweight async generation state logger that emits time-indexed state snapshots for each generation worker. This logger should be cheap enough for debug runs and should not depend on profiling.

Target state tuple per sample interval:

- `decode_batch_size`
- `context_batch_size`
- `avg_kv_slen`
- `p50_kv_slen`
- `p95_kv_slen`
- `max_kv_slen`
- `pending_requests`
- `new_prefills_this_tick`
- `decode_tokens_this_tick`
- metadata: `wall_time`, `step`, `worker_id`, `dp_idx`, `interval_s`

### Phase 2: DLSim-to-Silicon Matching

For each DLSim point:

- `S_i = (gen_batch_size_i, context_batch_size_i, gen_kv_slen_i)`

Build silicon time-indexed states:

- `X_t = (decode_batch_size_t, context_batch_size_t, avg_kv_slen_t)`

Use a normalized distance:

- `d_i(t) = 0.4*|decode_t-g_i|/max(1,g_i) + 0.3*|context_t-c_i|/max(1,c_i) + 0.3*|kv_t-k_i|/max(1,k_i)`

Then select, for each DLSim point:

- the best contiguous window of at least 3 consecutive samples
- with the lowest average distance
- record the matched window and carry forward `ifb_sample_weight`

### Phase 3: Targeted Profiling

Only after phase 2 is working:

- either trigger profiling when the logger enters a matched state window
- or manually profile the matched timestamps/windows
- analyze silicon traces at those matched points, weighted by `ifb_sample_weight`

## Sub-Plan 1: Async State Logger

### Scope

Implement a worker-local state trace for async vLLM generation. Do not add any profiler triggers yet.

### Hook points

Use these existing surfaces:

- `nemo_rl/models/generation/vllm/vllm_worker_async.py`
- `nemo_rl/models/generation/vllm/vllm_generation.py`
- `nemo_rl/algorithms/utils.py`

### Data model

Maintain an in-memory request-state table on the model-owner async generation worker. For each active request, track:

- `request_id`
- `submit_time`
- `prompt_len`
- `sampled_output_len` if available
- `current_generated_tokens`
- `first_token_seen`
- `finished`

### State reconstruction rule

At each sampling interval:

- `running_total` comes from the existing vLLM gauge `num_requests_running`
- `waiting_total` comes from the existing vLLM gauge `num_requests_waiting`
- `decode_batch_size` = count of active requests with `current_generated_tokens > 0`
- `context_batch_size` = `max(running_total - decode_batch_size, 0)`

Because vLLM does not currently expose exact prefill/decode split or KV lengths directly, reconstruct them heuristically:

- active zero-token requests are ordered by `submit_time`
- the oldest `context_batch_size` zero-token active requests are treated as prefill-running
- remaining zero-token active requests are treated as queue-waiting and are excluded from running KV statistics

KV-length estimate:

- decode request KV length = `prompt_len + current_generated_tokens`
- prefill-running request KV length = `prompt_len`
- waiting request = excluded from running-state KV statistics

Derived interval counters:

- `pending_requests` = `waiting_total`
- `new_prefills_this_tick` = count of requests that newly entered the prefill-running classification since the previous snapshot
- `decode_tokens_this_tick` = delta of the existing vLLM `generation_tokens` counter since the previous snapshot

### Output format

Write one JSONL file per worker under the run log tree:

- `logger.log_dir/async_state_trace/worker-dp<idx>.jsonl`

Each line is one interval snapshot with:

- time fields: `wall_time_s`, `step`, `interval_s`
- identity: `worker_id`, `dp_idx`
- direct state: `running_total`, `waiting_total`, `decode_batch_size`, `context_batch_size`
- KV stats: `avg_kv_slen`, `p50_kv_slen`, `p95_kv_slen`, `max_kv_slen`
- deltas: `new_prefills_this_tick`, `decode_tokens_this_tick`
- optional debug: `kv_cache_usage_perc`

### Summary metrics

Keep the existing performance block unchanged in v1. Add summary metrics at the normal end-of-step metrics surface:

- `state_trace/decode_batch_size/{mean,p50,p95,max}`
- `state_trace/context_batch_size/{mean,p50,p95,max}`
- `state_trace/avg_kv_slen/{mean,p50,p95,max}`
- `state_trace/pending_requests/{mean,p50,p95,max}`
- `state_trace/new_prefills_per_tick/{mean,p50,p95,max}`
- `state_trace/decode_tokens_per_tick/{mean,p50,p95,max}`

Do not add a live ASCII histogram yet. The canonical debug artifact is the JSONL state trace.

## Implementation Subtasks

### 1. Worker-side trace collection

- extend the async worker's existing metrics logger thread to sample a request-state table plus the already-collected vLLM gauges/counters
- update request state during request submission, incremental async output updates, and request completion
- write per-interval JSONL snapshots from the model-owner actor only

### 2. Aggregation and exposure

- extend worker-to-driver metric collection to include summary statistics for the state trace
- keep JSONL files local to workers and use the existing log-sync path to collect them with the run logs

### 3. Offline matcher

- add a small offline script or notebook-free Python entrypoint that:
  - loads DLSim point JSON
  - loads the worker JSONL traces
  - computes distances
  - returns the best window per DLSim point
- emit a machine-readable result file with matched windows and scores

### 4. Validation ladder

- validate first on small async Llama with the already-working synthetic OSL setup
- confirm the logger produces stable JSONL and sensible summaries
- then run the same logger on DeepSeek before any new NSYS collection

## Test Plan

- unit-test the request-state reconstruction logic with synthetic request timelines:
  - decode-only
  - prefill plus decode overlap
  - queued requests plus running requests
  - request completion and zero-token edge cases
- unit-test interval summarization:
  - correct JSONL fields
  - correct mean/p50/p95/max summaries
  - correct `decode_tokens_this_tick` counter behavior
- runtime acceptance on async Llama:
  - JSONL files exist for generation workers
  - summary metrics exist in merged metrics
  - sampled OSL metrics still match actual generation lengths
  - state traces show nontrivial variation in decode/context/KV state across the run
- offline matcher acceptance:
  - returns one contiguous window per DLSim point
  - no NaNs or empty-window failures when points are reachable

## Assumptions

- v1 targets `L=1` correlation semantics. DLSim lag-state slicing is not being reimplemented in NeMo-RL.
- real prompt lengths remain the silicon ISL source; only the output-length distribution is currently aligned
- the prefill/decode/KV reconstruction is heuristic by design for v1 and is acceptable as the first matching layer
- the existing per-worker imbalance and vLLM logger visuals remain useful, but they are not the canonical correlation artifact; the JSONL state trace is

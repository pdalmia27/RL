# Add Internal-Worker NVTX for Inflight Changes + Synthetic Response-Length Sampling

## Summary

Implement the next profiling run on the current `v0.5.0` DeepSeek async benchmark, not on a new random-dataset runner.

The patch should do two things:

- add synthetic response-length sampling to the existing DeepSeek async generation path, using direct lognormal parameters from the simulator while keeping the current prompt/input side unchanged
- add NSYS-visible NVTX regions inside the vLLM internal worker processes whenever inflight concurrency changes, so the ranges appear in the same `worker_process_*` traces as the attention and FC kernels

Do not bring in the full `PR #1453` random-dataset path for this change. Reuse only the relevant ideas from that PR:

- `ignore_eos`
- synthetic output-length generation
- a sequence-length generator helper

## Key Changes

### 1. Extend generation config for synthetic response lengths

Add these config fields to the generation interface and config normalization path:

- `policy.generation.ignore_eos: bool = false`
- `policy.generation.output_len_or_output_len_generator: int | dict | null`
- `policy.generation.vllm_cfg.enable_inflight_nvtx_regions: bool = false`

Supported generator schema for this work:

```yaml
policy:
  generation:
    ignore_eos: true
    output_len_or_output_len_generator:
      type: lognormal
      mu: <float>
      sigma: <float>
      min_value: 1
      max_value: ${policy.generation.max_new_tokens}
      seed: ${grpo.seed}
```

Implementation choices:

- use a dedicated helper to build a sampler from config
- support `int` and `type: lognormal`
- keep the default path unchanged when the field is unset
- in `configure_generation_config`, only auto-fill EOS into `stop_token_ids` when `ignore_eos` is `false`

### 2. Apply synthetic output lengths in the existing vLLM generation path

Adapt the current vLLM worker path so the async DeepSeek benchmark can stay on the existing perf recipe.

Behavior:

- parse the generator once during vLLM worker init
- for each async request, compute:
  - `remaining_ctx`
  - `allowed_new_tokens = min(max_new_tokens, remaining_ctx, sampled_output_len)` when a sampler is configured
- when `ignore_eos=true`, pass empty `stop_token_ids` and set `ignore_eos=True` in vLLM sampling params
- implement the same output-length handling in the sync worker too, so the interface stays coherent even though the current target run is async

This keeps the current prompt dataset and current benchmark script, while making response-length stragglers synthetic.

### 3. Put NVTX where the kernels actually run

Do not rely on actor-only NVTX in `vllm_async_generation_worker` for the main signal.

Reason:

- the useful NSYS reports for async vLLM are the internal `worker_process_*` traces
- actor-only NVTX likely will not land in the same process as the attention and FC kernels

Instead:

- add two methods to the vLLM internal worker extension:
  - `update_inflight_nvtx_region(label: str) -> None`
  - `clear_inflight_nvtx_region() -> None`
- implement them with `torch.cuda.nvtx.range_push/pop`
- keep exactly one active stable-region range per internal worker process
- on update:
  - pop the previous region if one exists
  - push a new one with the current label
- on clear/shutdown:
  - pop any outstanding region

### 4. Drive internal-worker NVTX from the async metrics logger

Use the existing async vLLM metrics logger as the control loop.

Behavior in the model-owner async actor:

- keep the existing inflight, pending, and KV usage collection
- add an active-request tracker keyed by request id with:
  - prompt length
  - sampled target output length
  - target total length
- on every metrics-logger tick:
  - read `vllm:num_requests_running`
  - read `vllm:num_requests_waiting`
  - read `vllm:kv_cache_usage_perc`
  - compute coarse active-request stats from the tracker:
    - `target_total_len_p50`
    - `target_total_len_max`
- if the region signature changed, RPC into the internal workers and update the NVTX region label

Use a compact label format like:

```text
vllm_inflight/running=7 waiting=3 kv=61.2 tgt_p50=1184 tgt_max=1536
```

Notes:

- this is a coarse first approximation of KV-length context
- it uses target total lengths derived from prompt length plus sampled output budget
- it does not attempt exact live per-request KV progression inside vLLM kernels in this patch

### 5. Keep the next profiling run operationally repeatable

For the next NSYS run:

- keep using the current DeepSeek perf benchmark
- keep using a fresh `EXP_DIR_OVERRIDE`
- enable:
  - `policy.generation.ignore_eos=True`
  - `policy.generation.output_len_or_output_len_generator=...`
  - `policy.generation.vllm_cfg.enable_inflight_nvtx_regions=True`
- keep:
  - `NRL_NSYS_WORKER_PATTERNS=vllm_async_generation_worker`
  - `NRL_NSYS_PROFILE_STEP_RANGE=2:3`

Also fix the log-sync path in `ray.sub` so profile artifacts reliably leave `/tmp/ray`:

- make `RAY_LOG_SYNC_FREQUENCY` explicit in the `srun` environment for head and worker containers rather than relying on implicit export behavior
- preserve the existing sidecar logic; only make the export deterministic

## Test Plan

### Static and unit

- generator helper:
  - `int` input returns a constant sampler
  - `type: lognormal` samples positive lengths
  - `min_value` and `max_value` clamping works
- generation config:
  - default behavior unchanged when new fields are unset
  - `ignore_eos=false` still auto-populates EOS stop token ids
  - `ignore_eos=true` does not force EOS stopping
- internal-worker NVTX state machine:
  - repeated identical signatures do not push duplicate ranges
  - changed signatures pop then push exactly once
  - clear/shutdown pops any open range

### Targeted runtime

- 1-worker or small async smoke:
  - synthetic response-length sampling affects `allowed_new_tokens`
  - request completes with `ignore_eos=true`
  - actor-side inflight metrics still collect normally
- small NSYS smoke:
  - internal worker receives region updates
  - `.nsys-rep` files sync out of `/tmp/ray`
  - trace contains `vllm_inflight/...` ranges in the same `worker_process_*` trace as GPU kernels

### Acceptance for the real run

- the 64-node DeepSeek async profiling run no longer skips due to old state
- it uses the current prompt dataset
- it samples response lengths from the configured lognormal
- NSYS traces contain stepwise inflight NVTX regions
- those regions can be used to segment attention and FC kernel time by concurrency regime

## Assumptions and Defaults

- Work against the current `nemo-rl-v0.5.0` tree and current DeepSeek async perf recipe.
- Keep the current prompt/input side unchanged for the next run.
- The simulator handoff to NeMo-RL is direct lognormal parameters, not a raw trace file.
- Exact live KV-length progression is out of scope for this first patch; use coarse target-total-length stats in the NVTX label.
- All new behavior is opt-in; default benchmark behavior must remain unchanged.

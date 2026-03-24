# Branch Handoff: `feat/dlsim-correlation-stageA`

## Baseline Commit
- branch: `feat/dlsim-correlation-stageA`
- baseline checkpoint commit: `de0b3dfdd5cde566bb52ab3ab2fb09ece8bf1561`
- commit message: `Checkpoint DeepSeek profiling, DLSim sampling, and state tracing`

## What This Commit Contains
This checkpoint bundles four threads that were developed together:

1. DeepSeek startup stabilization
- resolves the HF model/config race by resolving the model once to a shared local snapshot under a lock
- standardizes the local DeepSeek materialized checkpoint path workflow

2. NSYS / Ray / vLLM profiling work
- worker selection and async profiling controls
- Ray-side nsight patching and submit helpers
- tests for async profiling completion gating and vLLM worker profiling behavior

3. DLSim-style output-length sampling
- `ignore_eos`
- DLSim-compatible truncated-lognormal output-length generation
- sampled-vs-actual output-length metrics and tests

4. Async state trace logging
- per-tick worker-local JSONL trace
- additive `performance/state_trace/...` summaries
- tests for reconstruction, rebasing, W&B filtering, and summary logging

## Main Files In The Baseline Commit
Code:
- `nemo_rl/models/megatron/community_import.py`
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- `nemo_rl/distributed/worker_group_utils.py`
- `nemo_rl/distributed/worker_groups.py`
- `nemo_rl/models/generation/output_length_samplers.py`
- `nemo_rl/models/generation/__init__.py`
- `nemo_rl/models/generation/interfaces.py`
- `nemo_rl/models/generation/vllm/config.py`
- `nemo_rl/models/generation/vllm/vllm_worker.py`
- `nemo_rl/models/generation/vllm/vllm_worker_async.py`
- `nemo_rl/models/generation/vllm/vllm_generation.py`
- `nemo_rl/models/generation/vllm/async_state_trace.py`
- `nemo_rl/experience/rollouts.py`
- `nemo_rl/algorithms/grpo.py`
- `nemo_rl/algorithms/utils.py`
- `ray.sub`
- `tests/test_suites/llm/performance/common.env`

Tests:
- `tests/unit/distributed/test_worker_groups.py`
- `tests/unit/models/generation/test_vllm_generation.py`
- `tests/unit/models/generation/test_async_state_trace.py`
- `tests/unit/models/generation/test_synthetic_output_lengths.py`

Tools:
- `tools/materialize_deepseek_v3_bf16.sh`
- `tools/patch_ray_nsight.py`
- `tools/submit_v050_perf.sh`
- `tools/submit_v050_nsys_smoke.sh`
- `tools/check_v050_job.sh`
- `tools/monitor_v050_job.sh`
- `tools/inspect_nsys_container.sh`
- `tools/inspect_ray_nsight_runtime.py`
- `tools/ray_nsys_actor_smoke.py`

Docs:
- `docs/nsys-profiling.md`
- `docs/nsys-profiling-journey-2026-03-19.md`
- `docs/dlsim-correlation-handoff-2026-03-22.md`
- `docs/dlsim-silicon-correlation-plan-2026-03-20.md`
- `docs/verify-ray-workers-use-nsight-hypothesis-2026-03-24.md`
- `docs/async_state_trace_explainer.json`

## Current Runtime Status
Recent validated behavior before this handoff:
- small async Llama runs validated DLSim-style output-length generation and async state-trace JSONL emission
- reduced-work DeepSeek async runs also emitted per-worker state-trace JSONL and aggregated state-trace summaries
- a heavier Stage A DeepSeek run is now queued/running separately to match TP8/DP16-style DLSim topology more closely

## Important Local State Not Included In The Baseline Commit
The following were intentionally left out of `de0b3dfd` because they are drafts, generated artifacts, backups, or one-off experiment outputs:
- `.codex-backups/`
- generated experiment directories under `tests/test_suites/llm/performance/`
- raw draft diagrams:
  - `docs/Nemo_rl.png`
  - `docs/nemo rl 2.png`
- extra explainer image not yet committed at baseline:
  - `docs/async_state_trace_explainer.png`
- temporary long-prompt file used for one observability test:
  - `examples/prompts/cot_long_debug.txt`
- local planning note:
  - `docs/inflight-nvtx-nsys-plan-2026-03-11.md`

## Recommended Workflow For Another Codex
1. Start from branch `feat/dlsim-correlation-stageA`
2. Treat `de0b3dfd` as the functional checkpoint
3. Pull or cherry-pick any follow-up cleanup commit made after this note
4. Avoid touching generated experiment directories
5. Coordinate carefully on the hot files:
   - `nemo_rl/algorithms/grpo.py`
   - `nemo_rl/models/generation/vllm/vllm_worker_async.py`
   - `nemo_rl/models/generation/vllm/vllm_generation.py`
   - `nemo_rl/algorithms/utils.py`

## Suggested Division Of Labor
- this machine / runtime operator:
  - launch jobs
  - inspect traces and metrics
  - validate DeepSeek behavior on cluster
- other machine / write-capable Codex:
  - matcher implementation
  - cleanup/refactor
  - docs and tests
  - config surfacing

# NSYS Profiling Journey: Llama Baseline to DeepSeek Worker-Process Reports

## Goal

Document the path from the first failing async Llama NSYS runs to the final DeepSeek run that emitted real `worker_process_*.nsys-rep` files.

This note is intentionally operational. It focuses on:

- what failed
- what changed
- what was finally proven

## Final Working References

### Llama baseline that emitted reports

- Job: `1216090`
- Path:
  - `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/1216090-logs`
- What was verified:
  - `Step 2/3` profiling started and stopped cleanly
  - reports were emitted under the Ray session `logs/nsight/` trees
- Example reports:
  - `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/1216090-logs/ray/session_2026-03-17_14-42-12_190654_2069860/logs/nsight/vllm_generation_worker_2:3_2090672.nsys-rep`
  - `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/1216090-logs/ray/lyris0176/session_2026-03-17_14-42-12_190654_2069860/logs/nsight/vllm_generation_worker_2:3_1824368.nsys-rep`

### DeepSeek model-parallel run that emitted reports

- Job: `1234735`
- Path:
  - `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/1234735-logs`
- What was verified:
  - `Step 2/3` profiling started and stopped cleanly
  - reports were emitted under the Ray session `logs/nsight/` trees as `worker_process_*.nsys-rep`
  - the final effective fix was Ray-side, not direct vLLM file patching:
    - `patched_ray=True matched_ray=2 patched_vllm=False matched_vllm=0`
    - in `ray-head.log` and worker logs
- Example reports:
  - `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/1234735-logs/ray/lyris0132/session_2026-03-18_16-38-25_195060_380597/logs/nsight/worker_process_3098906.nsys-rep`
  - `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/1234735-logs/ray/lyris0057/session_2026-03-18_16-38-25_195060_380597/logs/nsight/worker_process_180977.nsys-rep`
  - `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/1234735-logs/ray/lyris0155/session_2026-03-18_16-38-25_195060_380597/logs/nsight/worker_process_2518183.nsys-rep`

## What Happened

### 1. Early async Llama smoke with the older container `nsys`

- Reference job: `1163771`
- What happened:
  - profiling reached graph-mode replay and then crashed
- Verified failure markers in `ray-driver.log`:
  - `Caught signal 11`
  - `cuGraphLaunch`
  - `at::cuda::CUDAGraph::replay()`
- Example lines:
  - `1163771-logs/ray-driver.log:1671`
  - `1163771-logs/ray-driver.log:1691`

Interpretation:

- the older `nsys` path was interacting badly with CUDA graph replay on this async smoke path

### 2. Forcing the newer `/usr/local/bin/nsys`

- Reference jobs:
  - `1202920`
  - `1203217`
- What changed:
  - forced Ray to use `/usr/local/bin/nsys`
  - this is `2025.6.1.190`
- What improved:
  - the old `cuGraphLaunch` / `CUDAGraph::replay()` crash disappeared
- What still failed:
  - full-step async profiling hung and timed out instead
- Verified failure markers:
  - `Timeout waiting for worker results after 1200.0s`
  - repeated `Error generating response for sample ...`

Interpretation:

- switching to newer `nsys` removed the immediate replay crash, but full-step profiling on the async smoke path was still too heavy and did not finish cleanly

### 3. The first important correction: reports were not always missing, sometimes we were checking the wrong place or too early

- The working Llama doc-style run `1216090` proved that:
  - reports land under the Ray session `logs/nsight/` trees
  - they can appear only after profiling stop and NSYS post-processing
- Practical lesson:
  - never declare “no reports” until:
    - start and stop both happened
    - post-processing had time to finish
    - the full session `logs/nsight/` trees were searched

### 4. The doc-style Llama run established the known-good capture model

- Reference job: `1216090`
- Benchmark path:
  - `examples/run_grpo_math.py`
  - `examples/configs/grpo_math_8B.yaml`
- Profiling settings that worked:
  - `NRL_NSYS_WORKER_PATTERNS="*vllm*"`
  - `NRL_NSYS_PROFILE_STEP_RANGE=2:3`
  - `CONTAINER_PATH_PREFIX=/usr/local/bin`
  - `RAY_NSYS_BINARY=/usr/local/bin/nsys`
- Verified behavior:
  - `Starting GPU profiling`
  - `Stopping GPU profiling`
  - `vllm_generation_worker_2:3_*.nsys-rep` emitted

Interpretation:

- the standard outer-worker path with capture-range-driven start/stop was healthy

### 5. DeepSeek model-parallel profiling reached start/stop but produced no reports

- Reference jobs:
  - `1224869`
  - `1225381`
  - `1226368`
- What was verified:
  - `Starting GPU profiling`
  - `Stopping GPU profiling`
  - but no `worker_process_*.nsys-rep` files were emitted
- Most important diagnostic point:
  - model-parallel vLLM runs the real GPU work in internal Ray worker processes named `worker_process_%p`
  - on this path, the internal workers were not being launched with the same capture-range-aware NSYS defaults as the working Llama outer-worker path

Interpretation:

- start/stop was happening in NeMo-RL
- but the profiled model-parallel worker processes were not armed with the right NSYS capture behavior, so no reports were flushed

### 6. A separate DeepSeek issue: outer-actor local profiler start could OOM

- Reference job: `1224796`
- What was verified:
  - after the batch-shape fix, DeepSeek async reached profiling start
  - then `torch.cuda.profiler.start()` on outer async actors hit OOM
- Practical fix:
  - non-model-owner outer actors must no-op on profiling start/stop
  - only the internal vLLM worker processes should actually be profiled

## Root Causes

### Root cause 1: wrong `nsys` binary on the early async smoke path

- `/usr/local/cuda/bin/nsys` was older and correlated with the graph replay crash
- `/usr/local/bin/nsys` was newer and removed that immediate crash

### Root cause 2: full-step async smoke profiling was too heavy

- `NRL_NSYS_PROFILE_STEP_RANGE=2:3` means one full training step
- on the async smoke path, that could still hang or time out

### Root cause 3: model-parallel DeepSeek used a different effective profiling process

- the useful reports come from internal `worker_process_%p` executors
- not from the outer NeMo generation actor names used in the simpler Llama case

### Root cause 4: the effective NSYS launch recipe for DeepSeek MP was incomplete until Ray was patched

- the final successful run showed:
  - `patched_ray=True`
  - `patched_vllm=False`
- this matters because it means the decisive fix surface was Ray’s `nsight.py`, not a direct vLLM file patch

### Root cause 5: DeepSeek HF checkpoint loading had its own independent race

- separate from NSYS report generation, DeepSeek previously hit:
  - `No architectures found in model config`
- that was fixed by resolving to a shared local snapshot under a lock
- in later DeepSeek profiling runs we also used the materialized local path:
  - `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/DeepSeek-V3-BF16`
- this was necessary operational hardening, but it was not the actual report-generation fix

## Final Fix Surface

### File

- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tools/patch_ray_nsight.py`

### What the final patch does

- makes Ray honor:
  - `RAY_NSYS_BINARY`
- injects these NSYS defaults when absent:
  - `t=cuda,cudnn,cublas,nvtx`
  - `capture-range=cudaProfilerApi`
  - `capture-range-end=stop`
  - `stop-on-exit=true`
- preserves `cuda-graph-trace=node`
- improves the runtime log message so the actual profiler command is visible

### Why this mattered

- the Llama baseline succeeded because the profiled process was using capture-range-aware NSYS arguments
- DeepSeek MP only started emitting `worker_process_*.nsys-rep` once the Ray-side NSYS defaults were patched to match that capture model

## Other Code Changes That Helped

### Outer-actor profiling no-op

- Files:
  - `nemo_rl/models/generation/vllm/vllm_worker.py`
  - `nemo_rl/models/generation/vllm/vllm_worker_async.py`
- Change:
  - non-model-owner outer actors no-op on profiling start/stop instead of calling local `torch.cuda.profiler.start()`
- Why:
  - avoids OOM on large async DeepSeek runs
  - keeps profiling scoped to the internal vLLM worker processes that actually matter

## Practical Rules Going Forward

- Use `/usr/local/bin/nsys`, not the older CUDA-bundled path, for these runs.
- For DeepSeek profiling, keep using the materialized local checkpoint path and offline HF mode.
- Search under the full Ray session `logs/nsight/` trees before declaring “no reports”.
- On model-parallel runs, expect `worker_process_*.nsys-rep`, not `vllm_generation_worker_*.nsys-rep`.
- If profiling start/stop logs appear but no DeepSeek reports land, inspect the effective Ray-side NSYS command first.

## Verified Reference Jobs

- `1163771`
  - older async Llama smoke
  - crashed in `cuGraphLaunch` / `CUDAGraph::replay()`
- `1203217`
  - newer `nsys`
  - avoided the replay crash but hung in async timeouts
- `1216090`
  - doc-style Llama baseline
  - emitted `vllm_generation_worker_2:3_*.nsys-rep`
- `1226368`
  - DeepSeek MP
  - profiling start/stop but no reports
- `1234735`
  - DeepSeek MP final successful report-emitting run
  - emitted `worker_process_*.nsys-rep`

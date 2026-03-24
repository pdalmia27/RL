# Handoff: Verify `ray_workers_use_nsight` TP>1 Hang Hypothesis and Proposed Fix

## Goal
Verify a new hypothesis about the TP>1 NSYS issue in NeMo-RL/vLLM and determine whether the proposed fix is:
- technically correct,
- sufficient to avoid the hang/crash,
- and acceptable for our profiling goals.

The proposed fix is:
- in `nemo_rl/models/generation/vllm/vllm_worker.py`
- stop setting:
  - `vllm_kwargs["ray_workers_use_nsight"] = True`

## External Claim To Verify
A teammate reported the following model:

1. The outer `VllmGenerationWorker` is already profiled via NeMo RL `runtime_env.nsight`.
2. When TP > 1, NeMo also sets `ray_workers_use_nsight=True`.
3. That causes vLLM's internal Ray TP workers to also run under `nsys`.
4. Those internal workers communicate through Ray compiled DAG/shared memory channels.
5. `nsys` instrumentation on those internal workers interferes with compiled DAG communication and causes a timeout/hang.
6. With TP = 1, there are no internal Ray TP workers in the same sense, so the problem does not appear.
7. Therefore the workaround/fix is to stop setting `ray_workers_use_nsight=True` and rely on profiling from the outer worker only.

## What We Already Know Locally

### Verified local code facts
- In our tree, NeMo does set `ray_workers_use_nsight=True` on the vLLM Ray distributed executor path:
  - `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/models/generation/vllm/vllm_worker.py`
- The outer generation worker is also profiled through NeMo's `runtime_env.nsight` path:
  - same file, around the worker `runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}` setup
- Our local DeepSeek TP>1 profiling work patched the Ray-side nsight launcher so internal `worker_process_%p` workers run with the full capture-range config:
  - `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tools/patch_ray_nsight.py`

### Verified local runtime result
We **did** get successful internal DeepSeek worker traces on our stack:
- DeepSeek async/model-parallel produced real `worker_process_*.nsys-rep` files after our Ray-side nsight patch.
- So the claim "TP>1 internal-worker profiling is impossible" is false on our setup.

But this does **not** disprove the teammate's hang hypothesis.
It only shows that the path can work in at least some short diagnostic runs.

### Current local interpretation
Our current best interpretation is:
- the teammate's explanation is technically plausible,
- but their proposed fix likely trades profiling fidelity for stability,
- because profiling only the outer worker is not equivalent to profiling the internal TP worker processes that actually execute the GPU kernels.

## What Needs To Be Verified
There are really **two** questions.

### Question 1: Is the hang hypothesis technically correct?
Does `ray_workers_use_nsight=True` on TP>1 actually correlate with:
- hangs,
- compiled DAG timeouts,
- or other TP-worker failures
that disappear when that flag is disabled?

### Question 2: Is the proposed fix acceptable for our use case?
If we disable `ray_workers_use_nsight=True`:
- do we avoid the hang,
- and what profiling artifacts remain?
- specifically: do we lose the internal `worker_process_%p` traces we needed for DeepSeek MP analysis?

## Why This Matters
Our goal is not just "make the run not hang."
Our goal is:
- stable profiling **and**
- profiling data from the actual GPU execution path that matters for TP>1 DeepSeek.

A fix that disables inner profiling may be a good default or workaround,
but it may be unacceptable if it removes the real worker-process traces.

## Suggested Validation Matrix
Use the same DeepSeek TP>1 path for all comparisons.
Keep everything else as fixed as possible.

### A. Baseline: current local behavior
Run with our current local code as-is:
- outer worker profiled
- internal workers also profiled via `ray_workers_use_nsight=True`
- Ray-side nsight patch still active

Capture:
- whether run hangs/crashes
- whether `worker_process_*.nsys-rep` are produced
- whether the reports are usable

### B. Proposed fix: disable internal TP-worker nsight
Patch only this behavior:
- do **not** set `vllm_kwargs["ray_workers_use_nsight"] = True`

Keep outer `runtime_env.nsight` profiling enabled.

Capture:
- whether hang/crash disappears
- whether any `.nsys-rep` files are produced
- whether they are only outer-worker traces
- whether internal `worker_process_%p` traces disappear

### C. Optional control: TP=1 path
If needed, compare with a TP=1 path to confirm the teammate's TP>1-specific claim.
This is secondary; the critical path is the TP>1 comparison above.

## Minimum Evidence To Collect
For each run, collect:
- run config summary
- whether TP > 1
- whether `ray_workers_use_nsight` is set
- whether run completes or hangs
- exact failure mode if it hangs
- presence/absence of:
  - outer-worker `.nsys-rep`
  - internal `worker_process_*.nsys-rep`
- if reports exist, note whether they contain the expected profiling window

If possible, also capture:
- relevant Ray / compiled DAG error messages
- whether the hang is deterministic or intermittent

## Expected Outcomes And How To Interpret Them

### Outcome 1
- disabling `ray_workers_use_nsight` removes the hang
- but internal `worker_process_%p` traces disappear

Interpretation:
- teammate's workaround is probably valid as a stability workaround
- but not equivalent to our higher-fidelity profiling path
- likely best as a configurable knob, not a blanket replacement

### Outcome 2
- disabling `ray_workers_use_nsight` removes the hang
- and profiling quality remains sufficient for our needs

Interpretation:
- teammate's fix may be preferable
- but this needs real artifact inspection, not just "run completed"

### Outcome 3
- disabling `ray_workers_use_nsight` does not materially change the hang/crash behavior

Interpretation:
- teammate's hypothesis is incomplete or wrong on our stack
- our current Ray-side nsight path remains the more relevant fix

### Outcome 4
- current local behavior remains stable and continues to emit internal worker traces
- teammate's issue does not reproduce on our setup

Interpretation:
- their issue may be environment/version dependent
- we should still consider a safety knob, but not necessarily replace our current path

## Local Files To Inspect
Primary code path:
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/nemo_rl/models/generation/vllm/vllm_worker.py`

Relevant local profiler patch:
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tools/patch_ray_nsight.py`

Related docs/handoff:
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/docs/nsys-profiling-journey-2026-03-19.md`
- `/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/docs/dlsim-correlation-handoff-2026-03-22.md`

## Recommendation To The Verifier
Do **not** treat this as a pure code-style cleanup.
This is a behavior tradeoff question:
- stability versus fidelity.

The most important thing is to determine whether disabling `ray_workers_use_nsight`:
1. really fixes a TP>1-specific hang on our stack,
2. and whether it destroys the internal traces we actually care about.

That answer should drive whether this becomes:
- a default,
- a fallback/workaround,
- or a configurable option.

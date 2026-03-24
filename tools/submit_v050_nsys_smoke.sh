#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"

BENCHMARK_BASENAME="${BENCHMARK_BASENAME:-grpo-llama3.1-8b-instruct-2n4g-async-1off}"
export BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-tests/test_suites/llm/performance/${BENCHMARK_BASENAME}.sh}"
export NUM_NODES="${NUM_NODES:-2}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
export JOB_LABEL="${JOB_LABEL:-nemorl.v050.nsys-smoke}"
export NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS:-vllm_async_generation_worker}"
export NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE:-2:3}"
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-30}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-}"

if [[ -z "${EXP_DIR_OVERRIDE:-}" ]]; then
  export EXP_DIR_OVERRIDE="$REPO_ROOT/tests/test_suites/llm/performance/${BENCHMARK_BASENAME}-nsys-smoke-${RUN_STAMP}"
fi

if [[ -n "$VLLM_ENFORCE_EAGER" ]]; then
  EAGER_OVERRIDE="policy.generation.vllm_cfg.enforce_eager=${VLLM_ENFORCE_EAGER}"
  if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    export EXTRA_OVERRIDES="${EXTRA_OVERRIDES} ${EAGER_OVERRIDE}"
  else
    export EXTRA_OVERRIDES="$EAGER_OVERRIDE"
  fi
fi

exec "$REPO_ROOT/tools/submit_v050_perf.sh" "$@"

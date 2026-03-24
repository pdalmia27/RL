#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/submit_v050_perf.sh [--submit] [--dry-run]

Environment overrides:
  SLURM_ACCOUNT           Default: coreai_dlalgo_llm
  SLURM_PARTITION         Default: gb200
  CONTAINER               Default: nvcr.io/nvidia/nemo-rl:v0.5.0
  GPUS_PER_NODE           Default: 4
  NUM_NODES               Default: 64
  TIME_LIMIT              Default: 04:00:00
  JOB_LABEL               Default: nemorl.v050
  HF_HOME                 Default: /lustre/fsw/coreai_dlalgo_llm/users/pdalmia/hf_home
  UV_CACHE_DIR            Default: /lustre/fsw/coreai_dlalgo_llm/users/pdalmia/uv_cache
  NRL_DEEPSEEK_V3_HF_CKPT Default: unset (benchmark default is live HF repo ID)
  HF_HUB_OFFLINE          Default: 0
  TRANSFORMERS_OFFLINE    Default: 0
  NRL_NSYS_WORKER_PATTERNS Default: unset
  NRL_NSYS_PROFILE_STEP_RANGE Default: unset
  RAY_LOG_SYNC_FREQUENCY  Default: unset
  BENCHMARK_SCRIPT        Default: tests/test_suites/llm/performance/grpo-deepseek-v3-64n4g-async-1off.sh
  EXP_DIR_OVERRIDE        Default: unset (benchmark uses its fixed per-script directory)
  WANDB_ENABLED           Default: False
  EXTRA_OVERRIDES         Extra CLI overrides appended to the benchmark command

Notes:
  - Default mode is dry-run. Use --submit to actually call sbatch.
  - HF_TOKEN must already be exported in the shell.
EOF
}

MODE="dry-run"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)
      MODE="submit"
      ;;
    --dry-run)
      MODE="dry-run"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set" >&2
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

SLURM_ACCOUNT="${SLURM_ACCOUNT:-coreai_dlalgo_llm}"
SLURM_PARTITION="${SLURM_PARTITION:-gb200}"
CONTAINER="${CONTAINER:-nvcr.io/nvidia/nemo-rl:v0.5.0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_NODES="${NUM_NODES:-64}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
JOB_LABEL="${JOB_LABEL:-nemorl.v050}"
HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/hf_home}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/uv_cache}"
NRL_DEEPSEEK_V3_HF_CKPT="${NRL_DEEPSEEK_V3_HF_CKPT:-}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS:-}"
NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE:-}"
RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-tests/test_suites/llm/performance/grpo-deepseek-v3-64n4g-async-1off.sh}"
EXP_DIR_OVERRIDE="${EXP_DIR_OVERRIDE:-}"
WANDB_ENABLED="${WANDB_ENABLED:-False}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
MEGATRON_ROOT_BASE="${MEGATRON_ROOT_BASE:-$HF_HOME/nemo_rl_runs_v050}"
RUN_STAMP=$(date +%Y%m%d-%H%M%S)
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-$MEGATRON_ROOT_BASE/$RUN_STAMP}"

if [[ -n "$NRL_DEEPSEEK_V3_HF_CKPT" && ! -d "$NRL_DEEPSEEK_V3_HF_CKPT" ]]; then
  echo "NRL_DEEPSEEK_V3_HF_CKPT does not exist: $NRL_DEEPSEEK_V3_HF_CKPT" >&2
  exit 2
fi

if [[ -n "$NRL_NSYS_WORKER_PATTERNS" || -n "$NRL_NSYS_PROFILE_STEP_RANGE" ]]; then
  if [[ -z "$NRL_NSYS_WORKER_PATTERNS" || -z "$NRL_NSYS_PROFILE_STEP_RANGE" ]]; then
    echo "Set both NRL_NSYS_WORKER_PATTERNS and NRL_NSYS_PROFILE_STEP_RANGE, or neither." >&2
    exit 2
  fi
fi

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR" "$NRL_MEGATRON_CHECKPOINT_DIR"

MOUNTS="${MOUNTS:-/lustre:/lustre:ro,$REPO_ROOT:$REPO_ROOT,$HF_HOME:$HF_HOME,$UV_CACHE_DIR:$UV_CACHE_DIR}"

if [[ "$NUM_NODES" -gt 16 ]]; then
  EXTRA_SLURM_ARGS=(--segment 16)
else
  EXTRA_SLURM_ARGS=()
fi

COMMAND="HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE HF_DATASETS_CACHE=$HF_DATASETS_CACHE HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE UV_CACHE_DIR=$UV_CACHE_DIR HF_TOKEN=$HF_TOKEN NRL_MEGATRON_CHECKPOINT_DIR=$NRL_MEGATRON_CHECKPOINT_DIR"
if [[ -n "$NRL_DEEPSEEK_V3_HF_CKPT" ]]; then
  COMMAND+=" NRL_DEEPSEEK_V3_HF_CKPT=$NRL_DEEPSEEK_V3_HF_CKPT"
fi
if [[ -n "$EXP_DIR_OVERRIDE" ]]; then
  COMMAND+=" EXP_DIR_OVERRIDE=$EXP_DIR_OVERRIDE"
fi
if [[ -n "$NRL_NSYS_WORKER_PATTERNS" ]]; then
  COMMAND+=" NRL_NSYS_WORKER_PATTERNS=$NRL_NSYS_WORKER_PATTERNS NRL_NSYS_PROFILE_STEP_RANGE=$NRL_NSYS_PROFILE_STEP_RANGE"
fi
COMMAND+=" uv run bash $BENCHMARK_SCRIPT logger.wandb_enabled=$WANDB_ENABLED"
if [[ -n "$EXTRA_OVERRIDES" ]]; then
  COMMAND+=" $EXTRA_OVERRIDES"
fi

SBATCH_CMD=(
  sbatch
  --chdir="$REPO_ROOT"
  --nodes="$NUM_NODES"
  --account="$SLURM_ACCOUNT"
  --job-name="${SLURM_ACCOUNT}-${JOB_LABEL}"
  --partition="$SLURM_PARTITION"
  "${EXTRA_SLURM_ARGS[@]}"
  --time="$TIME_LIMIT"
  "$REPO_ROOT/ray.sub"
)

printf 'repo_root=%s\n' "$REPO_ROOT"
printf 'mode=%s\n' "$MODE"
printf 'container=%s\n' "$CONTAINER"
printf 'account=%s\n' "$SLURM_ACCOUNT"
printf 'partition=%s\n' "$SLURM_PARTITION"
printf 'num_nodes=%s\n' "$NUM_NODES"
printf 'gpus_per_node=%s\n' "$GPUS_PER_NODE"
printf 'hf_home=%s\n' "$HF_HOME"
printf 'hf_hub_cache=%s\n' "$HF_HUB_CACHE"
printf 'hf_datasets_cache=%s\n' "$HF_DATASETS_CACHE"
printf 'uv_cache_dir=%s\n' "$UV_CACHE_DIR"
printf 'local_checkpoint=%s\n' "${NRL_DEEPSEEK_V3_HF_CKPT:-<benchmark default>}"
printf 'hf_hub_offline=%s\n' "$HF_HUB_OFFLINE"
printf 'transformers_offline=%s\n' "$TRANSFORMERS_OFFLINE"
printf 'nsys_worker_patterns=%s\n' "${NRL_NSYS_WORKER_PATTERNS:-<unset>}"
printf 'nsys_profile_step_range=%s\n' "${NRL_NSYS_PROFILE_STEP_RANGE:-<unset>}"
printf 'ray_log_sync_frequency=%s\n' "${RAY_LOG_SYNC_FREQUENCY:-<unset>}"
printf 'benchmark_script=%s\n' "$BENCHMARK_SCRIPT"
printf 'exp_dir_override=%s\n' "${EXP_DIR_OVERRIDE:-<benchmark default>}"
printf 'wandb_enabled=%s\n' "$WANDB_ENABLED"
printf 'checkpoint_dir=%s\n' "$NRL_MEGATRON_CHECKPOINT_DIR"
printf 'mounts=%s\n' "$MOUNTS"
REDACTED_COMMAND=$(printf '%s\n' "$COMMAND" | sed -E 's/(HF_TOKEN=)[^[:space:]]+/\1***REDACTED***/g')
printf 'command=%s\n' "$REDACTED_COMMAND"
printf 'sbatch='
printf '%q ' "${SBATCH_CMD[@]}"
printf '\n'

if [[ "$MODE" != "submit" ]]; then
  exit 0
fi

export CONTAINER
export GPUS_PER_NODE
export MOUNTS
export COMMAND
if [[ -n "$RAY_LOG_SYNC_FREQUENCY" ]]; then
  export RAY_LOG_SYNC_FREQUENCY
fi
"${SBATCH_CMD[@]}"

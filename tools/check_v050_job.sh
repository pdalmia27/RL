#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/check_v050_job.sh <job_id>

Environment:
  LINES=<n>    Number of tail lines to print from logs. Default: 120
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

JOB_ID="$1"
LINES="${LINES:-120}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$(cd -- "$REPO_ROOT/.." && pwd)
REPO_LOG_DIR="$REPO_ROOT/${JOB_ID}-logs"
WORKSPACE_LOG_DIR="$WORKSPACE_ROOT/${JOB_ID}-logs"
if [[ -d "$REPO_LOG_DIR" ]]; then
  LOG_DIR="$REPO_LOG_DIR"
else
  LOG_DIR="$WORKSPACE_LOG_DIR"
fi
DRIVER_LOG="$LOG_DIR/ray-driver.log"
HEAD_LOG="$LOG_DIR/ray-head.log"
REPO_SLURM_OUT="$REPO_ROOT/slurm-${JOB_ID}.out"
WORKSPACE_SLURM_OUT="$WORKSPACE_ROOT/slurm-${JOB_ID}.out"
if [[ -f "$REPO_SLURM_OUT" ]]; then
  SLURM_OUT="$REPO_SLURM_OUT"
else
  SLURM_OUT="$WORKSPACE_SLURM_OUT"
fi

print_header() {
  printf '\n== %s ==\n' "$1"
}

redact_stream() {
  sed -E \
    -e 's/(HF_TOKEN=)[^[:space:]]+/\1***REDACTED***/g' \
    -e 's/("HF_TOKEN"[[:space:]]*:[[:space:]]*")[^"]+/\1***REDACTED***/g' \
    -e "s/('HF_TOKEN'[[:space:]]*:[[:space:]]*')[^']+/\\1***REDACTED***/g"
}

print_header "squeue"
if command -v squeue >/dev/null 2>&1; then
  squeue -j "$JOB_ID" -o '%.18i %.9P %.40j %.8T %.10M %.9l %R' || true
else
  echo "squeue not found"
fi

print_header "sacct"
if command -v sacct >/dev/null 2>&1; then
  sacct -j "$JOB_ID" --format=JobID,JobName%40,Partition,State,ExitCode,Elapsed -n -P \
    | sed -n '1,80p' || true
else
  echo "sacct not found"
fi

print_header "paths"
printf 'repo_root=%s\n' "$REPO_ROOT"
printf 'workspace_root=%s\n' "$WORKSPACE_ROOT"
printf 'log_dir=%s\n' "$LOG_DIR"
printf 'driver_log=%s\n' "$DRIVER_LOG"
printf 'head_log=%s\n' "$HEAD_LOG"
printf 'slurm_out=%s\n' "$SLURM_OUT"

print_header "log dir"
if [[ -d "$LOG_DIR" ]]; then
  ls -ld "$LOG_DIR"
  find "$LOG_DIR" -maxdepth 1 -type f | sort | sed -n '1,80p'
else
  echo "log directory not found"
fi

print_header "driver key lines"
if [[ -f "$DRIVER_LOG" ]]; then
  rg -n \
    "ActorDiedError|RuntimeError:|ValueError:|Traceback|Architectures: None|No architectures|gradient_accumulation_fusion|fused_weight_gradient|Watchdog|NCCL|timeout|Model architecture not supported|Using vLLM backend|Loaded configuration from|Overrides:" \
    "$DRIVER_LOG" | tail -n 120 || true
else
  echo "ray-driver.log not found"
fi

print_header "driver tail"
if [[ -f "$DRIVER_LOG" ]]; then
  tail -n "$LINES" "$DRIVER_LOG" | redact_stream
else
  echo "ray-driver.log not found"
fi

print_header "head tail"
if [[ -f "$HEAD_LOG" ]]; then
  tail -n "$LINES" "$HEAD_LOG" | redact_stream
else
  echo "ray-head.log not found"
fi

print_header "slurm tail"
if [[ -f "$SLURM_OUT" ]]; then
  tail -n "$LINES" "$SLURM_OUT" | redact_stream
else
  echo "slurm output not found"
fi

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/monitor_v050_job.sh <job_id> [--interval-seconds N] [--log-file PATH] [--max-polls N]

Environment:
  LINES=<n>    Number of tail lines to print in each appended snapshot. Default: 120

Behavior:
  - Appends timestamped snapshots from check_v050_job.sh to a log file.
  - Stops automatically once the primary Slurm job reaches a terminal state.
  - If --max-polls is set, stops after N polls even if the job is still active.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

JOB_ID="$1"
shift

INTERVAL_SECONDS=600
MAX_POLLS=0
LINES="${LINES:-120}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$(cd -- "$REPO_ROOT/.." && pwd)
CHECK_SCRIPT="$SCRIPT_DIR/check_v050_job.sh"
LOG_FILE="${WORKSPACE_ROOT}/job-monitor-${JOB_ID}.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval-seconds)
      INTERVAL_SECONDS="$2"
      shift
      ;;
    --log-file)
      LOG_FILE="$2"
      shift
      ;;
    --max-polls)
      MAX_POLLS="$2"
      shift
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

if [[ ! -x "$CHECK_SCRIPT" ]]; then
  echo "Check script is missing or not executable: $CHECK_SCRIPT" >&2
  exit 2
fi

terminal_state() {
  case "$1" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED|REVOKED)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

get_primary_state() {
  local state=""

  if command -v sacct >/dev/null 2>&1; then
    state=$(
      sacct -j "$JOB_ID" --format=JobIDRaw,State -n -P 2>/dev/null \
        | awk -F'|' -v id="$JOB_ID" '$1 == id {print $2; exit}'
    )
    state="${state%% *}"
    state="${state%%+*}"
    if [[ -n "$state" ]]; then
      printf '%s\n' "$state"
      return 0
    fi
  fi

  if command -v squeue >/dev/null 2>&1; then
    state=$(squeue -h -j "$JOB_ID" -o '%T' 2>/dev/null | head -1 || true)
    if [[ -n "$state" ]]; then
      printf '%s\n' "$state"
      return 0
    fi
  fi

  printf 'UNKNOWN\n'
}

append_snapshot() {
  local state="$1"
  {
    printf '\n===== %s job=%s state=%s =====\n' "$(date -Is)" "$JOB_ID" "$state"
    LINES="$LINES" "$CHECK_SCRIPT" "$JOB_ID"
  } >> "$LOG_FILE" 2>&1
}

mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

{
  printf 'monitor_pid=%s\n' "$$"
  printf 'job_id=%s\n' "$JOB_ID"
  printf 'interval_seconds=%s\n' "$INTERVAL_SECONDS"
  printf 'log_file=%s\n' "$LOG_FILE"
  printf 'started_at=%s\n' "$(date -Is)"
} >> "$LOG_FILE"

poll_count=0
while true; do
  state="$(get_primary_state)"
  append_snapshot "$state"
  poll_count=$((poll_count + 1))

  if terminal_state "$state"; then
    printf '\n===== %s terminal_state=%s =====\n' "$(date -Is)" "$state" >> "$LOG_FILE"
    exit 0
  fi

  if [[ "$MAX_POLLS" -gt 0 && "$poll_count" -ge "$MAX_POLLS" ]]; then
    printf '\n===== %s max_polls_reached=%s =====\n' "$(date -Is)" "$MAX_POLLS" >> "$LOG_FILE"
    exit 0
  fi

  sleep "$INTERVAL_SECONDS"
done

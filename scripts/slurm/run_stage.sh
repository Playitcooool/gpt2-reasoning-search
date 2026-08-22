#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-}"
CONFIG_FILE="${SSH_TRAIN_CONFIG:-$PROJECT_ROOT/config/ssh.env}"

case "$STAGE" in
  prepare|smoke|proxies|pretrain|sft|rl|all) ;;
  *) echo "Unknown Slurm stage: $STAGE" >&2; exit 2 ;;
esac

[[ -f "$CONFIG_FILE" ]] || {
  echo "Missing $CONFIG_FILE. Run ./train-ssh setup and edit config/ssh.env first." >&2
  exit 2
}

cd "$PROJECT_ROOT"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-unknown}; host $(hostname); stage $STAGE"
echo "Started $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

if [[ "$STAGE" != "prepare" ]]; then
  nvidia-smi || true
  env SSH_TRAIN_CONFIG="$CONFIG_FILE" scripts/ssh/worker.sh doctor
fi

exec env SSH_TRAIN_CONFIG="$CONFIG_FILE" scripts/ssh/worker.sh "$STAGE"

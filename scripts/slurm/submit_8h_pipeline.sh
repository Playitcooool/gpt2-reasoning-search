#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${SSH_TRAIN_CONFIG:-$PROJECT_ROOT/config/ssh.env}"
WORKER_SCRIPT="$PROJECT_ROOT/scripts/ssh/worker.sh"
SUBMIT_STAGE="$PROJECT_ROOT/scripts/slurm/submit_stage.sh"

[[ -f "$CONFIG_FILE" ]] || {
  echo "Missing $CONFIG_FILE. Run ./train-ssh setup and edit config/ssh.env first." >&2
  exit 2
}
[[ -x "$WORKER_SCRIPT" ]] || {
  echo "Missing executable $WORKER_SCRIPT. Check out the complete repository." >&2
  exit 2
}
[[ -x "$SUBMIT_STAGE" ]] || {
  echo "Missing executable $SUBMIT_STAGE. Check out the complete repository." >&2
  exit 2
}
command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is not available on this server." >&2
  exit 2
}

cd "$PROJECT_ROOT"
mkdir -p logs
set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a
export SSH_TRAIN_CONFIG="$CONFIG_FILE"

echo "Preparing data and fixed training inputs before submitting GPU jobs..."
"$WORKER_SCRIPT" prepare

pretrain_job="$(SSH_TRAIN_CONFIG="$CONFIG_FILE" "$SUBMIT_STAGE" pretrain)"
sft_job="$(SSH_TRAIN_CONFIG="$CONFIG_FILE" "$SUBMIT_STAGE" sft "$pretrain_job")"
rl_job="$(SSH_TRAIN_CONFIG="$CONFIG_FILE" "$SUBMIT_STAGE" rl "$sft_job")"

cat <<EOF
Submitted the separate eight-hour stages:
  pretrain: $pretrain_job
  SFT:      $sft_job (afterok:$pretrain_job)
  RL:       $rl_job (afterok:$sft_job)

Monitor with: squeue -j $pretrain_job,$sft_job,$rl_job
If a stage times out: scancel $sft_job $rl_job, then rerun this wrapper to create a fresh chain.
EOF

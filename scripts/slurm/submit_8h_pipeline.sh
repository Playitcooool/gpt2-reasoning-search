#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${SSH_TRAIN_CONFIG:-$PROJECT_ROOT/config/ssh.env}"
BATCH_SCRIPT="$PROJECT_ROOT/scripts/slurm/train_h100.sbatch"

[[ -f "$CONFIG_FILE" ]] || {
  echo "Missing $CONFIG_FILE. Run ./train-ssh setup and edit config/ssh.env first." >&2
  exit 2
}
command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is not available on this server." >&2
  exit 2
}

cd "$PROJECT_ROOT"
mkdir -p logs
export SSH_TRAIN_CONFIG="$CONFIG_FILE"

submit() {
  local dependency="$1"
  local stage="$2"
  local output
  local args=(--parsable --export="ALL,SSH_TRAIN_CONFIG=$CONFIG_FILE")
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:$dependency")
  fi
  output="$(sbatch "${args[@]}" "$BATCH_SCRIPT" "$stage")"
  output="${output%%;*}"
  [[ "$output" =~ ^[0-9]+$ ]] || {
    echo "Could not parse Slurm job id from: $output" >&2
    exit 3
  }
  printf '%s\n' "$output"
}

pretrain_job="$(submit "" pretrain)"
sft_job="$(submit "$pretrain_job" sft)"
rl_job="$(submit "$sft_job" rl)"

cat <<EOF
Submitted the independent eight-hour stages:
  pretrain: $pretrain_job
  SFT:      $sft_job (afterok:$pretrain_job)
  RL:       $rl_job (afterok:$sft_job)

Monitor with: squeue -j $pretrain_job,$sft_job,$rl_job
If a stage times out: scancel $sft_job $rl_job, then rerun this wrapper to create a fresh chain.
EOF

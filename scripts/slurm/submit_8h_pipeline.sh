#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${SSH_TRAIN_CONFIG:-$PROJECT_ROOT/config/ssh.env}"
BATCH_SCRIPT="$PROJECT_ROOT/scripts/slurm/train_h100.sbatch"
WORKER_SCRIPT="$PROJECT_ROOT/scripts/ssh/worker.sh"

[[ -f "$CONFIG_FILE" ]] || {
  echo "Missing $CONFIG_FILE. Run ./train-ssh setup and edit config/ssh.env first." >&2
  exit 2
}
[[ -x "$WORKER_SCRIPT" ]] || {
  echo "Missing executable $WORKER_SCRIPT. Check out the complete repository." >&2
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

submit() {
  local dependency="$1"
  local stage="$2"
  local output
  local args=(
    --parsable
    --job-name="${SESSION_PREFIX:-grs}-$stage"
    --gres="${SLURM_GRES:-gpu:1}"
    --cpus-per-task="${SLURM_CPUS:-16}"
    --mem="${SLURM_MEMORY:-128G}"
    --time="${SLURM_TIME:-08:00:00}"
    --export="ALL,SSH_TRAIN_CONFIG=$CONFIG_FILE"
  )
  [[ -n "${SLURM_PARTITION:-}" ]] && args+=(--partition="$SLURM_PARTITION")
  [[ -n "${SLURM_ACCOUNT:-}" ]] && args+=(--account="$SLURM_ACCOUNT")
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
Submitted the separate eight-hour stages:
  pretrain: $pretrain_job
  SFT:      $sft_job (afterok:$pretrain_job)
  RL:       $rl_job (afterok:$sft_job)

Monitor with: squeue -j $pretrain_job,$sft_job,$rl_job
If a stage times out: scancel $sft_job $rl_job, then rerun this wrapper to create a fresh chain.
EOF

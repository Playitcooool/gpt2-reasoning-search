#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${SSH_TRAIN_CONFIG:-$PROJECT_ROOT/config/ssh.env}"
STAGE="${1:-}"
DEPENDENCY="${2:-}"

case "$STAGE" in
  prepare|smoke|proxies|pretrain|sft|rl|all) ;;
  *)
    echo "Choose a stage: prepare, smoke, proxies, pretrain, sft, rl, or all." >&2
    exit 2
    ;;
esac
[[ -f "$CONFIG_FILE" ]] || {
  echo "Missing $CONFIG_FILE. Run ./train-ssh setup and edit config/ssh.env first." >&2
  exit 2
}
command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is not available on this server." >&2
  exit 2
}

cd "$PROJECT_ROOT"
set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

script="$PROJECT_ROOT/scripts/slurm/$STAGE.sbatch"
[[ -f "$script" ]] || { echo "Missing stage script: $script" >&2; exit 2; }

args=(
  --parsable
  --job-name="${SESSION_PREFIX:-grs}-$STAGE"
  --cpus-per-task="${SLURM_CPUS:-16}"
  --mem="${SLURM_MEMORY:-128G}"
  --time="${SLURM_TIME:-08:00:00}"
  --export="ALL,SSH_TRAIN_CONFIG=$CONFIG_FILE"
)
if [[ "$STAGE" != "prepare" ]]; then
  args+=(--gpus="${SLURM_GPUS:-h100}")
fi
[[ -n "${SLURM_PARTITION:-}" ]] && args+=(--partition="$SLURM_PARTITION")
[[ -n "${SLURM_ACCOUNT:-}" ]] && args+=(--account="$SLURM_ACCOUNT")
[[ -n "$DEPENDENCY" ]] && args+=(--dependency="afterok:$DEPENDENCY")

output="$(sbatch "${args[@]}" "$script")"
output="${output%%;*}"
[[ "$output" =~ ^[0-9]+$ ]] || {
  echo "Could not parse Slurm job id from: $output" >&2
  exit 3
}
printf '%s\n' "$output"

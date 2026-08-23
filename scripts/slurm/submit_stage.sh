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

token_file_ready() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  [[ "$path" == *.npy ]] || [[ -f "${path%.*}.manifest.json" ]]
}

pretrain_inputs_ready() {
  local reasoning_tokens="${REASONING_TOKENS:-data/processed/reasoning.bin}"
  local general_tokens="${GENERAL_TOKENS:-data/processed/general.bin}"
  local tokenizer_path="${TOKENIZER_PATH:-artifacts/tokenizer.json}"
  token_file_ready "$reasoning_tokens" && token_file_ready "$general_tokens" \
    && [[ -s "$tokenizer_path" ]] \
    && [[ -s "$(dirname "$reasoning_tokens")/preparation-manifest.json" ]]
}

submit_job() {
  local stage="$1"
  local dependency="${2:-}"
  local script="$PROJECT_ROOT/scripts/slurm/$stage.sbatch"
  local output
  local args=(
    --parsable
    --job-name="${SESSION_PREFIX:-grs}-$stage"
    --cpus-per-task="${SLURM_CPUS:-16}"
    --mem="${SLURM_MEMORY:-128G}"
    --time="${SLURM_TIME:-08:00:00}"
    --export="ALL,SSH_TRAIN_CONFIG=$CONFIG_FILE"
  )

  [[ -f "$script" ]] || { echo "Missing stage script: $script" >&2; exit 2; }
  if [[ "$stage" != "prepare" ]]; then
    args+=(--gpus="${SLURM_GPUS:-h100}")
  fi
  [[ -n "${SLURM_PARTITION:-}" ]] && args+=(--partition="$SLURM_PARTITION")
  [[ -n "${SLURM_ACCOUNT:-}" ]] && args+=(--account="$SLURM_ACCOUNT")
  [[ -n "$dependency" ]] && args+=(--dependency="afterok:$dependency")

  output="$(sbatch "${args[@]}" "$script")"
  output="${output%%;*}"
  [[ "$output" =~ ^[0-9]+$ ]] || {
    echo "Could not parse Slurm job id from: $output" >&2
    exit 3
  }
  printf '%s\n' "$output"
}

if [[ "$STAGE" == "pretrain" && -z "$DEPENDENCY" && "${AUTO_PREPARE:-1}" == "1" ]] \
  && ! pretrain_inputs_ready; then
  prepare_job="$(submit_job prepare)"
  echo "Queued data preparation CPU job $prepare_job; pretraining will wait for it." >&2
  DEPENDENCY="$prepare_job"
fi

submit_job "$STAGE" "$DEPENDENCY"

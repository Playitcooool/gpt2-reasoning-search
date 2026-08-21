#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${SSH_TRAIN_CONFIG:-$PROJECT_ROOT/config/ssh.env}"
STAGE="${1:-}"

case "$STAGE" in
  doctor|prepare|smoke|proxies|pretrain|sft|rl|all) ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac

[[ -f "$CONFIG_FILE" ]] || { echo "Missing $CONFIG_FILE" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

# Existing config/ssh.env files are intentionally not overwritten by setup. Make an older config
# follow the new eight-hour default unless the user explicitly selects a custom profile.
if [[ "${TRAIN_PROFILE:-8h}" == "8h" ]]; then
  : "${PREPARE_IN_JOB:=0}"
  : "${RUN_SMOKE:=1}"
  : "${RUN_PROXIES:=0}"
  : "${RUN_PRETRAIN:=1}"
  : "${RUN_SFT:=1}"
  : "${RUN_RL:=1}"
  MAIN_TOKEN_CAP=2500000000
  MAIN_HOURS=7.5
  SFT_HOURS=7.5
  RL_HOURS=7.5
  SFT_EPOCHS=4
  RL_EPOCHS=8
  RL_GROUP_SIZE=4
fi

: "${SFT_HOURS:=7.5}"
: "${RL_HOURS:=7.5}"

cd "$PROJECT_ROOT"
mkdir -p logs artifacts checkpoints data/processed "$TRAIN_CACHE"
export UV_CACHE_DIR="$TRAIN_CACHE/uv"
export HF_HOME="$TRAIN_CACHE/huggingface"
export TORCH_HOME="$TRAIN_CACHE/torch"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
RESUME_ARGS=()

if [[ "$STAGE" != "doctor" ]]; then
  trap 'rm -f "$PROJECT_ROOT/logs/$STAGE.pid"' EXIT
  exec > >(tee -a "$PROJECT_ROOT/logs/$STAGE.log") 2>&1
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Starting stage: $STAGE"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$PROJECT_ROOT/logs/training.lock"
    flock -n 9 || {
      echo "Another training/preparation stage is running. Check ./train-ssh status." >&2
      exit 3
    }
  fi
fi

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; exit 2; }
}

latest_checkpoint() {
  local output="$1"
  local candidate
  shopt -s nullglob
  local candidates=("$output"/step-*)
  shopt -u nullglob
  ((${#candidates[@]} > 0)) || return 1
  while IFS= read -r candidate; do
    if [[ -f "$candidate/model.safetensors" && -f "$candidate/optimizer.pt" \
      && -f "$candidate/scheduler.pt" && -f "$candidate/rng.pt" \
      && -f "$candidate/state.json" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(printf '%s\n' "${candidates[@]}" | sort -r)
  return 1
}

set_resume_args() {
  local output="$1"
  RESUME_ARGS=()
  if [[ "${AUTO_RESUME:-1}" == "1" && ! -e "$output/final" ]]; then
    local checkpoint
    checkpoint="$(latest_checkpoint "$output" || true)"
    if [[ -n "$checkpoint" ]]; then
      RESUME_ARGS=(--resume-from "$checkpoint")
    fi
  fi
  return 0
}

run_doctor() {
  local failed=0
  echo "Project: $PROJECT_ROOT"
  echo "Config:  $CONFIG_FILE"
  echo "Cache:   $TRAIN_CACHE"
  command -v uv >/dev/null 2>&1 || { echo "MISSING: uv (run ./train-ssh setup)"; failed=1; }
  command -v nvidia-smi >/dev/null 2>&1 || { echo "MISSING: nvidia-smi"; failed=1; }
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  fi
  df -h "$PROJECT_ROOT" "$TRAIN_CACHE" | awk 'NR == 1 || !seen[$1]++'
  for path in "$REASONING_TOKENS" "$GENERAL_TOKENS" "$TOKENIZER_PATH" \
    "$WIKIPEDIA_JSONL" "$GROUNDED_QUESTIONS" "$RL_PROMPTS"; do
    if [[ -e "$path" ]]; then echo "OK:      $path"; else echo "PENDING: $path"; fi
  done
  if command -v uv >/dev/null 2>&1; then
    local cuda_report
    if ! cuda_report="$(uv run --locked python -c \
      'import torch; cuda = torch.cuda.is_available(); bf16 = cuda and torch.cuda.is_bf16_supported(); print("CUDA:", cuda, "bf16:", bf16); raise SystemExit(0 if cuda and bf16 else 3)' 2>&1)"; then
      echo "$cuda_report"
      echo "NOT READY: CUDA and bf16 support are required."
      failed=1
    elif [[ "$cuda_report" == *"CUDA: True bf16: True"* ]]; then
      echo "$cuda_report"
    else
      echo "$cuda_report"
      echo "NOT READY: CUDA and bf16 support are required."
      failed=1
    fi
  fi
  return "$failed"
}

run_prepare() {
  uv run --locked gpt2-reasoning-search bootstrap-data \
    --data-root "$PROJECT_ROOT/data" \
    --manifest-output "$PROJECT_ROOT/artifacts/auto-data-manifest.json"
  if [[ ! -f "$TOKENIZER_PATH" ]]; then
    local tokenizer_inputs=()
    while IFS= read -r tokenizer_input; do
      tokenizer_inputs+=("$tokenizer_input")
    done < <(find "$TOKENIZER_INPUT_DIR" -type f -name '*.txt' -print | sort)
    ((${#tokenizer_inputs[@]} > 0)) || {
      echo "Put representative .txt files in $TOKENIZER_INPUT_DIR first." >&2
      exit 2
    }
    uv run --locked gpt2-reasoning-search train-tokenizer "${tokenizer_inputs[@]}" \
      --output "$TOKENIZER_PATH" --vocab-size 50304
  fi
  if [[ ! -f "$REASONING_TOKENS" || ! -f "$GENERAL_TOKENS" ]]; then
    local evaluation_args=()
    if [[ -f "$EVALUATION_PROMPTS" ]]; then
      evaluation_args=(--evaluation-prompts "$EVALUATION_PROMPTS")
    fi
    local args=(uv run --locked gpt2-reasoning-search prepare-data \
      --tokenizer-path "$TOKENIZER_PATH" --manifest config/datasets.json \
      --output "$(dirname "$REASONING_TOKENS")" \
      --reasoning-token-cap "$REASONING_TOKEN_CAP" \
      --general-token-cap "$GENERAL_TOKEN_CAP")
    if ((${#evaluation_args[@]} > 0)); then args+=("${evaluation_args[@]}"); fi
    "${args[@]}"
  fi
  if [[ ! -d "$WIKI_INDEX" ]]; then
    require_file "$WIKIPEDIA_JSONL"
    local index_args=()
    if [[ "${LEXICAL_ONLY:-0}" == "1" ]]; then
      index_args=(--lexical-only)
    fi
    local args=(uv run --locked gpt2-reasoning-search build-index "$WIKIPEDIA_JSONL" \
      --output "$WIKI_INDEX" --embedding-device cpu)
    if ((${#index_args[@]} > 0)); then args+=("${index_args[@]}"); fi
    "${args[@]}"
  fi
  if [[ ! -f "$TRAJECTORIES" ]]; then
    require_file "$GROUNDED_QUESTIONS"
    uv run --locked gpt2-reasoning-search make-trajectories "$GROUNDED_QUESTIONS" \
      --output "$TRAJECTORIES"
  fi
}

require_prepared() {
  if [[ "${RUN_PRETRAIN:-1}" == "1" || "${RUN_PROXIES:-0}" == "1" ]]; then
    require_file "$REASONING_TOKENS"
    require_file "$GENERAL_TOKENS"
  fi
  if [[ "${RUN_SFT:-1}" == "1" ]]; then
    require_file "$TOKENIZER_PATH"
    require_file "$TRAJECTORIES"
  fi
  if [[ "${RUN_RL:-1}" == "1" ]]; then
    require_file "$TOKENIZER_PATH"
    require_dir "$WIKI_INDEX"
    require_file "$RL_PROMPTS"
  fi
}

run_smoke() {
  uv run --locked gpt2-reasoning-search smoke-overfit --device cuda
}

run_pretrain_job() {
  local name="$1" preset="$2" ratio="$3" token_cap="$4" hours="$5" output="$6"
  if [[ -d "$output/final" ]]; then
    echo "$name already complete: $output/final"
    return
  fi
  require_file "$REASONING_TOKENS"
  require_file "$GENERAL_TOKENS"
  set_resume_args "$output"
  local args=(uv run --locked gpt2-reasoning-search pretrain \
    --reasoning-tokens "$REASONING_TOKENS" --general-tokens "$GENERAL_TOKENS" \
    --output "$output" --preset "$preset" --reasoning-ratio "$ratio" \
    --max-tokens "$token_cap" --time-budget-hours "$hours")
  if ((${#RESUME_ARGS[@]} > 0)); then args+=("${RESUME_ARGS[@]}"); fi
  "${args[@]}"
  if [[ ! -d "$output/final" ]]; then
    echo "$name reached its time budget without a final checkpoint. Re-submit the same stage to resume." >&2
    return 75
  fi
}

run_proxies() {
  run_pretrain_job proxy-r0 proxy-124m 0.0 "$PROXY_TOKEN_CAP" "$PROXY_HOURS" \
    "$CHECKPOINT_ROOT/proxy-r0"
  run_pretrain_job proxy-r30 proxy-124m 0.3 "$PROXY_TOKEN_CAP" "$PROXY_HOURS" \
    "$CHECKPOINT_ROOT/proxy-r30"
  run_pretrain_job proxy-r70 proxy-124m 0.7 "$PROXY_TOKEN_CAP" "$PROXY_HOURS" \
    "$CHECKPOINT_ROOT/proxy-r70"
}

run_pretrain() {
  run_pretrain_job main-r70 main-350m 0.7 "$MAIN_TOKEN_CAP" "$MAIN_HOURS" "$MAIN_OUTPUT"
}

run_sft() {
  if [[ -f "$SFT_OUTPUT/model.safetensors" ]]; then
    echo "Tool SFT already complete: $SFT_OUTPUT"
    return
  fi
  require_dir "$MAIN_OUTPUT/final"
  require_file "$TOKENIZER_PATH"
  require_file "$TRAJECTORIES"
  set_resume_args "$SFT_OUTPUT"
  local args=(uv run --locked gpt2-reasoning-search sft-tools \
    --checkpoint "$MAIN_OUTPUT/final" --tokenizer-path "$TOKENIZER_PATH" \
    --trajectories "$TRAJECTORIES" --output "$SFT_OUTPUT" \
    --epochs "$SFT_EPOCHS" --time-budget-hours "$SFT_HOURS" \
    --micro-batch-size "$SFT_MICRO_BATCH" \
    --gradient-accumulation-steps "$SFT_GRAD_ACCUM")
  if ((${#RESUME_ARGS[@]} > 0)); then args+=("${RESUME_ARGS[@]}"); fi
  "${args[@]}"
  if [[ ! -f "$SFT_OUTPUT/model.safetensors" ]]; then
    echo "Tool SFT reached its time budget without a final checkpoint. Re-submit the same stage to resume." >&2
    return 75
  fi
}

run_rl() {
  if [[ -d "$RL_OUTPUT/final" ]]; then
    echo "Search RL already complete: $RL_OUTPUT/final"
    return
  fi
  require_dir "$SFT_OUTPUT"
  require_file "$TOKENIZER_PATH"
  require_file "$RL_PROMPTS"
  require_dir "$WIKI_INDEX"
  set_resume_args "$RL_OUTPUT"
  local judge_args=(--llm-judge --judge-device cuda)
  if [[ "${USE_LLM_JUDGE:-1}" == "0" ]]; then
    judge_args=(--no-llm-judge)
  fi
  local args=(uv run --locked gpt2-reasoning-search rl-search \
    --checkpoint "$SFT_OUTPUT" --tokenizer-path "$TOKENIZER_PATH" \
    --prompts "$RL_PROMPTS" --index "$WIKI_INDEX" --output "$RL_OUTPUT" \
    --epochs "$RL_EPOCHS" --time-budget-hours "$RL_HOURS" \
    --group-size "$RL_GROUP_SIZE" --max-searches 3 \
    "${judge_args[@]}")
  if ((${#RESUME_ARGS[@]} > 0)); then args+=("${RESUME_ARGS[@]}"); fi
  "${args[@]}"
  if [[ ! -d "$RL_OUTPUT/final" ]]; then
    echo "Search RL reached its time budget without a final checkpoint. Re-submit the same stage to resume." >&2
    return 75
  fi
}

case "$STAGE" in
  doctor) run_doctor ;;
  prepare) run_prepare ;;
  smoke) run_smoke ;;
  proxies) run_proxies ;;
  pretrain) run_pretrain ;;
  sft) run_sft ;;
  rl) run_rl ;;
  all)
    if [[ "${TRAIN_PROFILE:-8h}" == "8h" && "${ALLOW_COMBINED_JOB:-0}" != "1" ]]; then
      echo "The 8-hour profile does not run multiple GPU stages in one job." >&2
      echo "Submit scripts/slurm/submit_8h_pipeline.sh, or set ALLOW_COMBINED_JOB=1 for a custom reservation." >&2
      exit 2
    fi
    if [[ "${PREPARE_IN_JOB:-0}" == "1" ]]; then run_prepare; else require_prepared; fi
    if [[ "${RUN_SMOKE:-1}" == "1" ]]; then run_smoke; fi
    if [[ "${RUN_PROXIES:-0}" == "1" ]]; then run_proxies; fi
    if [[ "${RUN_PRETRAIN:-1}" == "1" ]]; then run_pretrain; fi
    if [[ "${RUN_SFT:-1}" == "1" ]]; then run_sft; fi
    if [[ "${RUN_RL:-1}" == "1" ]]; then run_rl; fi
    ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac

if [[ "$STAGE" != "doctor" ]]; then
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Completed stage: $STAGE"
fi

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

validate_workflow_paths() {
  [[ "${ALLOW_CUSTOM_PATHS:-0}" == "1" ]] && return 0
  local specification name expected actual invalid=0
  for specification in \
    "TOKENIZER_INPUT_DIR|data/tokenizer-sample" \
    "TOKENIZER_PATH|artifacts/tokenizer.json" \
    "EVALUATION_PROMPTS|data/evaluation/contamination-prompts.jsonl" \
    "REASONING_TOKENS|data/processed/reasoning.bin" \
    "GENERAL_TOKENS|data/processed/general.bin" \
    "WIKIPEDIA_JSONL|data/raw/wikipedia.jsonl" \
    "GROUNDED_QUESTIONS|data/raw/grounded-questions.jsonl" \
    "RL_PROMPTS|data/rl/search-qa.jsonl" \
    "TRAJECTORIES|data/processed/tool-trajectories.jsonl" \
    "WIKI_INDEX|artifacts/wiki-index" \
    "CHECKPOINT_ROOT|checkpoints" \
    "MAIN_OUTPUT|checkpoints/main-350m" \
    "SFT_OUTPUT|checkpoints/tool-sft" \
    "RL_OUTPUT|checkpoints/search-rl"; do
    name="${specification%%|*}"
    expected="${specification#*|}"
    actual="${!name:-}"
    if [[ "$actual" != "$expected" && "$actual" != "$PROJECT_ROOT/$expected" ]]; then
      echo "Fixed workflow path changed: $name=$actual (expected $expected)." >&2
      invalid=1
    fi
  done
  if ((invalid)); then
    echo "Do not edit data/checkpoint paths in config/ssh.env; the automatic pipeline owns them." >&2
    echo "Set ALLOW_CUSTOM_PATHS=1 only for an explicitly managed custom layout." >&2
    return 2
  fi
}

validate_workflow_paths || exit $?

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

require_nonempty_file() {
  [[ -s "$1" ]] || { echo "Missing or empty required file: $1" >&2; exit 2; }
}

token_file_ready() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  [[ "$path" == *.npy ]] || [[ -f "${path%.*}.manifest.json" ]]
}

corpora_ready() {
  token_file_ready "$REASONING_TOKENS" && token_file_ready "$GENERAL_TOKENS" \
    && [[ -s "$(dirname "$REASONING_TOKENS")/preparation-manifest.json" ]]
}

wiki_index_ready() {
  [[ -d "$WIKI_INDEX/lexical" && -f "$WIKI_INDEX/metadata.sqlite3" \
    && -f "$WIKI_INDEX/retrieval-manifest.json" ]]
}

checkpoint_complete() {
  local directory="$1"
  [[ -f "$directory/model.safetensors" && -f "$directory/optimizer.pt" \
    && -f "$directory/scheduler.pt" && -f "$directory/rng.pt" \
    && -f "$directory/state.json" ]]
}

require_corpora() {
  if ! token_file_ready "$REASONING_TOKENS" || ! token_file_ready "$GENERAL_TOKENS"; then
    echo "Missing token file or manifest." >&2
    echo "Run ./train-ssh prepare before training." >&2
    exit 2
  fi
  [[ -s "$(dirname "$REASONING_TOKENS")/preparation-manifest.json" ]] || {
    echo "Missing preparation manifest; run ./train-ssh prepare again." >&2
    exit 2
  }
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
    if [[ "$path" == *.bin ]]; then
      if token_file_ready "$path"; then echo "OK:      $path"; else echo "PENDING: $path"; fi
    elif [[ -s "$path" ]]; then
      echo "OK:      $path"
    else
      echo "PENDING: $path"
    fi
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
  if [[ ! -s "$TOKENIZER_PATH" ]]; then
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
  if ! corpora_ready; then
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
  if ! wiki_index_ready; then
    require_nonempty_file "$WIKIPEDIA_JSONL"
    local index_args=()
    if [[ "${LEXICAL_ONLY:-0}" == "1" ]]; then
      index_args=(--lexical-only)
    fi
    local args=(uv run --locked gpt2-reasoning-search build-index "$WIKIPEDIA_JSONL" \
      --output "$WIKI_INDEX" --embedding-device cpu)
    if ((${#index_args[@]} > 0)); then args+=("${index_args[@]}"); fi
    "${args[@]}"
  fi
  if [[ ! -s "$TRAJECTORIES" ]]; then
    require_nonempty_file "$GROUNDED_QUESTIONS"
    uv run --locked gpt2-reasoning-search make-trajectories "$GROUNDED_QUESTIONS" \
      --output "$TRAJECTORIES"
  fi
}

require_prepared() {
  if [[ "${RUN_PRETRAIN:-1}" == "1" || "${RUN_PROXIES:-0}" == "1" ]]; then
    require_corpora
  fi
  if [[ "${RUN_SFT:-1}" == "1" ]]; then
    require_nonempty_file "$TOKENIZER_PATH"
    require_nonempty_file "$TRAJECTORIES"
  fi
  if [[ "${RUN_RL:-1}" == "1" ]]; then
    require_nonempty_file "$TOKENIZER_PATH"
    wiki_index_ready || {
      echo "Missing or incomplete Wikipedia index: $WIKI_INDEX" >&2
      exit 2
    }
    require_nonempty_file "$RL_PROMPTS"
  fi
}

run_smoke() {
  uv run --locked gpt2-reasoning-search smoke-overfit --device cuda
}

run_smoke_gate() {
  [[ "${RUN_SMOKE:-1}" == "1" ]] || return 0
  local marker="$PROJECT_ROOT/artifacts/smoke-overfit.ok"
  if [[ -f "$marker" ]]; then
    echo "Smoke gate already passed: $marker"
    return 0
  fi
  run_smoke
  printf 'passed\n' > "$marker"
}

run_pretrain_job() {
  local name="$1" preset="$2" ratio="$3" token_cap="$4" hours="$5" output="$6"
  if checkpoint_complete "$output/final"; then
    echo "$name already complete: $output/final"
    return
  fi
  require_corpora
  set_resume_args "$output"
  local args=(uv run --locked gpt2-reasoning-search pretrain \
    --reasoning-tokens "$REASONING_TOKENS" --general-tokens "$GENERAL_TOKENS" \
    --output "$output" --preset "$preset" --reasoning-ratio "$ratio" \
    --max-tokens "$token_cap" --time-budget-hours "$hours")
  if ((${#RESUME_ARGS[@]} > 0)); then args+=("${RESUME_ARGS[@]}"); fi
  "${args[@]}"
  if ! checkpoint_complete "$output/final"; then
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
  if checkpoint_complete "$MAIN_OUTPUT/final"; then
    echo "main-r70 already complete: $MAIN_OUTPUT/final"
    return
  fi
  run_smoke_gate
  run_pretrain_job main-r70 main-350m 0.7 "$MAIN_TOKEN_CAP" "$MAIN_HOURS" "$MAIN_OUTPUT"
}

run_sft() {
  if checkpoint_complete "$SFT_OUTPUT"; then
    echo "Tool SFT already complete: $SFT_OUTPUT"
    return
  fi
  checkpoint_complete "$MAIN_OUTPUT/final" || {
    echo "Missing or incomplete pretraining checkpoint: $MAIN_OUTPUT/final" >&2
    exit 2
  }
  require_nonempty_file "$TOKENIZER_PATH"
  require_nonempty_file "$TRAJECTORIES"
  set_resume_args "$SFT_OUTPUT"
  local args=(uv run --locked gpt2-reasoning-search sft-tools \
    --checkpoint "$MAIN_OUTPUT/final" --tokenizer-path "$TOKENIZER_PATH" \
    --trajectories "$TRAJECTORIES" --output "$SFT_OUTPUT" \
    --epochs "$SFT_EPOCHS" --time-budget-hours "$SFT_HOURS" \
    --micro-batch-size "$SFT_MICRO_BATCH" \
    --gradient-accumulation-steps "$SFT_GRAD_ACCUM")
  if ((${#RESUME_ARGS[@]} > 0)); then args+=("${RESUME_ARGS[@]}"); fi
  "${args[@]}"
  if ! checkpoint_complete "$SFT_OUTPUT"; then
    echo "Tool SFT reached its time budget without a final checkpoint. Re-submit the same stage to resume." >&2
    return 75
  fi
}

run_rl() {
  if checkpoint_complete "$RL_OUTPUT/final"; then
    echo "Search RL already complete: $RL_OUTPUT/final"
    return
  fi
  checkpoint_complete "$SFT_OUTPUT" || {
    echo "Missing or incomplete tool-SFT checkpoint: $SFT_OUTPUT" >&2
    exit 2
  }
  require_nonempty_file "$TOKENIZER_PATH"
  require_nonempty_file "$RL_PROMPTS"
  wiki_index_ready || {
    echo "Missing or incomplete Wikipedia index: $WIKI_INDEX" >&2
    exit 2
  }
  set_resume_args "$RL_OUTPUT"
  local judge_args=(--llm-judge --judge-device cuda)
  if [[ "${USE_LLM_JUDGE:-1}" == "0" ]]; then
    judge_args=(--no-llm-judge)
  fi
  local retrieval_args=(--lexical-only)
  if [[ "${RL_LEXICAL_ONLY:-1}" == "0" ]]; then
    retrieval_args=(--hybrid-retrieval)
  fi
  local search_args=(--search-mode local)
  if [[ -n "${BRAVE_SEARCH_API_KEY:-}" ]]; then
    search_args=(--search-mode web)
    echo "Live Brave search enabled for RL; cached results and local fallback protect this run."
  fi
  local args=(uv run --locked gpt2-reasoning-search rl-search \
    --checkpoint "$SFT_OUTPUT" --tokenizer-path "$TOKENIZER_PATH" \
    --prompts "$RL_PROMPTS" --index "$WIKI_INDEX" --output "$RL_OUTPUT" \
    --epochs "$RL_EPOCHS" --time-budget-hours "$RL_HOURS" \
    --group-size "$RL_GROUP_SIZE" --max-searches 3 \
    "${retrieval_args[@]}" \
    "${search_args[@]}" \
    "${judge_args[@]}")
  if ((${#RESUME_ARGS[@]} > 0)); then args+=("${RESUME_ARGS[@]}"); fi
  "${args[@]}"
  if ! checkpoint_complete "$RL_OUTPUT/final"; then
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
    if [[ "${RUN_SMOKE:-1}" == "1" ]]; then run_smoke_gate; fi
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

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${SSH_TRAIN_CONFIG:-$PROJECT_ROOT/config/ssh.env}"
set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a
cd "$PROJECT_ROOT"

echo "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader
else
  echo "nvidia-smi unavailable"
fi

echo
echo "RUNNING"
if command -v tmux >/dev/null 2>&1; then
  tmux list-sessions -F '#{session_name}  #{session_created_string}' 2>/dev/null \
    | awk -v prefix="${SESSION_PREFIX:-grs}-" 'index($1, prefix) == 1' || true
fi
for pid_file in "$PROJECT_ROOT"/logs/*.pid; do
  [[ -e "$pid_file" ]] || continue
  pid="$(<"$pid_file")"
  kill -0 "$pid" 2>/dev/null && echo "$(basename "$pid_file" .pid)  PID $pid (nohup)"
done

echo
echo "CHECKPOINTS"
for output in "$CHECKPOINT_ROOT"/proxy-r0 "$CHECKPOINT_ROOT"/proxy-r30 \
  "$CHECKPOINT_ROOT"/proxy-r70 "$MAIN_OUTPUT" "$SFT_OUTPUT" "$RL_OUTPUT"; do
  [[ -e "$output" ]] || continue
  latest="$(find "$output" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort | tail -1)"
  if [[ -d "$output/final" || -f "$output/model.safetensors" ]]; then
    state="complete"
  else
    state="${latest##*/}"
  fi
  printf '%-36s %s\n' "${output#$PROJECT_ROOT/}" "${state:-started}"
done

echo
echo "RECENT METRICS"
for metrics in "$MAIN_OUTPUT/metrics.jsonl" "$SFT_OUTPUT/metrics.jsonl" \
  "$RL_OUTPUT/metrics.jsonl"; do
  [[ -s "$metrics" ]] || continue
  echo "${metrics#$PROJECT_ROOT/}"
  tail -n 1 "$metrics"
done

echo
echo "Recent logs:"
find "$PROJECT_ROOT/logs" -maxdepth 1 -type f -name '*.log' -print 2>/dev/null \
  | sort | tail -8 | sed "s|$PROJECT_ROOT/||"

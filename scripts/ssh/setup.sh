#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${SSH_TRAIN_CONFIG:-$PROJECT_ROOT/config/ssh.env}"

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_URL="https://astral.sh/uv/0.11.21/install.sh"
  echo "uv is not installed; installing pinned uv 0.11.21 in your user account (no sudo)."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "$UV_INSTALL_URL" | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$UV_INSTALL_URL" | sh
  else
    echo "Install curl or wget, then run this command again." >&2
    exit 2
  fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  cp "$PROJECT_ROOT/config/ssh.env.example" "$CONFIG_FILE"
  echo "Created $CONFIG_FILE"
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a
mkdir -p "$TRAIN_CACHE" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/artifacts" \
  "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/data/processed"
export UV_CACHE_DIR="$TRAIN_CACHE/uv"
export HF_HOME="$TRAIN_CACHE/huggingface"
export TORCH_HOME="$TRAIN_CACHE/torch"

cd "$PROJECT_ROOT"
uv sync --dev --locked
uv run --locked gpt2-reasoning-search version

echo
echo "Setup complete. Next:"
echo "  1. Edit config/ssh.env for SLURM_ACCOUNT, SLURM_TIME, and any required partition/GPU name."
echo "  2. Submit scripts/slurm/submit_8h_pipeline.sh (it prepares data automatically)."

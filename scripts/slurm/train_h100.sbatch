#!/usr/bin/env bash
# Submit one stage from the repository root (each job has its own 8-hour limit):
#   sbatch scripts/slurm/train_h100.sbatch pretrain
# Or submit the dependent pretrain -> SFT -> RL pipeline:
#   scripts/slurm/submit_8h_pipeline.sh
# To resume one stage:
#   sbatch scripts/slurm/train_h100.sbatch pretrain

#SBATCH --job-name=grs-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Uncomment and replace these only if your school requires them.
##SBATCH --partition=REPLACE_ME
##SBATCH --account=REPLACE_ME
# Some clusters also require: #SBATCH --constraint=h100

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-pretrain}"
CONFIG_FILE="${SSH_TRAIN_CONFIG:-$PROJECT_ROOT/config/ssh.env}"

[[ -f "$CONFIG_FILE" ]] || {
  echo "Missing $CONFIG_FILE. Run ./train-ssh setup and edit config/ssh.env first." >&2
  exit 2
}

cd "$PROJECT_ROOT"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-unknown}; host $(hostname); stage $STAGE"
echo "Started $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
nvidia-smi || true
env SSH_TRAIN_CONFIG="$CONFIG_FILE" scripts/ssh/worker.sh doctor

exec env SSH_TRAIN_CONFIG="$CONFIG_FILE" scripts/ssh/worker.sh "$STAGE"

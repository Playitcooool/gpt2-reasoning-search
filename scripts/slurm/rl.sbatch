#!/usr/bin/env bash
# Run or resume local-search reinforcement learning:
#   sbatch scripts/slurm/rl.sbatch

#SBATCH --job-name=grs-rl
#SBATCH --gpus=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
exec "$PROJECT_ROOT/scripts/slurm/run_stage.sh" rl

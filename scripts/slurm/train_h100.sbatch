#!/usr/bin/env bash
# Compatibility shim. Prefer the explicit stage scripts:
#   sbatch scripts/slurm/pretrain.sbatch
#   sbatch scripts/slurm/sft.sbatch
#   sbatch scripts/slurm/rl.sbatch

#SBATCH --job-name=grs-train
#SBATCH --gpus=h100
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

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
STAGE="${1:-pretrain}"
exec "$PROJECT_ROOT/scripts/slurm/run_stage.sh" "$STAGE"

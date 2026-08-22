#!/usr/bin/env bash
# Run a deliberately custom combined profile (blocked by the default 8-hour profile):
#   sbatch scripts/slurm/all.sbatch

#SBATCH --job-name=grs-all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$PROJECT_ROOT/scripts/slurm/run_stage.sh" all

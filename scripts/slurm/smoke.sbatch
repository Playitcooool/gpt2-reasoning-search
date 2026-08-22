#!/usr/bin/env bash
# Run the small CUDA learning gate:
#   sbatch scripts/slurm/smoke.sbatch

#SBATCH --job-name=grs-smoke
#SBATCH --gpus=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$PROJECT_ROOT/scripts/slurm/run_stage.sh" smoke

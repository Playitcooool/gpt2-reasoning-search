#!/usr/bin/env bash
# Prepare pinned data and all fixed local training inputs on a Slurm CPU job:
#   sbatch scripts/slurm/prepare.sbatch

#SBATCH --job-name=grs-prepare
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$PROJECT_ROOT/scripts/slurm/run_stage.sh" prepare

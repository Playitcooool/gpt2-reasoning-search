#!/usr/bin/env bash
# Run the optional 0% / 30% / 70% proxy ablations:
#   sbatch scripts/slurm/proxies.sbatch

#SBATCH --job-name=grs-proxies
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$PROJECT_ROOT/scripts/slurm/run_stage.sh" proxies

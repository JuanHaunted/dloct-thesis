#!/bin/bash
# Evaluate a finished run on the test split.
#
#   sbatch scripts/eval.slurm runs/unet_full [extra dloct.eval args]
#SBATCH --job-name=dloct-eval
#SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
RUN=${1:?usage: sbatch scripts/eval.slurm runs/<name>}
shift
cd "${SLURM_SUBMIT_DIR}"
uv run --frozen python -m dloct.eval --run "${RUN}" --data-root "${DLOCT_DATA:-data/prepared}" "$@"

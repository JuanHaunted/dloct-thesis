#!/bin/bash
# Evaluate a finished run on the test split.
#
#   sbatch scripts/eval.slurm runs/unet_full [extra dloct.eval args]
#SBATCH --job-name=dloct-eval
#SBATCH --partition=accel-2          # Apolo V100 partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
RUN=${1:?usage: sbatch scripts/eval.slurm runs/<name>}
shift
cd "${SLURM_SUBMIT_DIR}"
source scripts/env.sh
python -m dloct.eval --run "${RUN}" --data-root "${DLOCT_DATA:-data/prepared}" "$@"

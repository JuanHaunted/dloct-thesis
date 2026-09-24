#!/bin/bash
# One-time conversion of raw tomograms to the training layout (CPU only).
#
#   sbatch scripts/prepare.slurm /path/to/raw/train /scratch/$USER/dloct/prepared
#SBATCH --job-name=dloct-prep
#SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
uv run --frozen python -m dloct.prepare_data --src "${1:?raw dir}" --out "${2:?output dir}"

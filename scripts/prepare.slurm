#!/bin/bash
# One-time conversion of raw tomograms to the training layout (CPU only).
#
#   sbatch scripts/prepare.slurm /path/to/raw/train /scratch/$USER/dloct/prepared
#SBATCH --job-name=dloct-prep
#SBATCH --partition=longjobs         # CPU only
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/env.sh
python -m dloct.prepare_data --src "${1:?raw dir}" --out "${2:?output dir}" --split-file configs/split.yaml

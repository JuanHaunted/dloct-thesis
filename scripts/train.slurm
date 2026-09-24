#!/bin/bash
# Train one experiment on all GPUs of one node.
#
#   sbatch scripts/train.slurm configs/unet_full.yaml [--set key=value ...]
#
# Resubmitting the same command resumes from runs/<name>/latest.pt.
# Defaults are for EAFIT Apolo (accel-2, V100). Check GPUs per node with `sinfo -p accel-2 -o "%N %G"`.
#SBATCH --job-name=dloct
#SBATCH --partition=accel-2          # Apolo V100 partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00          # mandatory on Apolo (D-HH:MM:SS); ask staff for the max
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch scripts/train.slurm <config.yaml> [--set k=v ...]}
shift

# Point at the prepared data on fast storage, e.g. export DLOCT_DATA=/scratch/$USER/dloct/prepared
DATA_ROOT=${DLOCT_DATA:-data/prepared}

cd "${SLURM_SUBMIT_DIR}"
source scripts/env.sh
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")   # GPUs allocated to this job
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} / NGPU - 1 ))
echo "host=$(hostname) gpus=${NGPU} workers/gpu=${WORKERS} config=${CONFIG} data=${DATA_ROOT}"
nvidia-smi

export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
torchrun --standalone --nproc_per_node="${NGPU}" -m dloct.train \
    --config "${CONFIG}" --set data.root="${DATA_ROOT}" data.num_workers="${WORKERS}" "$@"

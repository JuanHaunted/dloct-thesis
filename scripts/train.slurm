#!/bin/bash
# Train one experiment on all GPUs of one node.
#
#   sbatch scripts/train.slurm configs/unet_full.yaml [--set key=value ...]
#
# Resubmitting the same command resumes from runs/<name>/latest.pt.
# Edit the #SBATCH lines below for your cluster (partition/account/GPU type).
#SBATCH --job-name=dloct
#SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch scripts/train.slurm <config.yaml> [--set k=v ...]}
shift

# Point at the prepared data on fast storage, e.g. export DLOCT_DATA=/scratch/$USER/dloct/prepared
DATA_ROOT=${DLOCT_DATA:-data/prepared}

cd "${SLURM_SUBMIT_DIR}"
NGPU=$(nvidia-smi -L | wc -l)
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} / NGPU - 1 ))
echo "host=$(hostname) gpus=${NGPU} workers/gpu=${WORKERS} config=${CONFIG} data=${DATA_ROOT}"
nvidia-smi

export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
uv run --frozen torchrun --standalone --nproc_per_node="${NGPU}" -m dloct.train \
    --config "${CONFIG}" --set data.root="${DATA_ROOT}" data.num_workers="${WORKERS}" "$@"

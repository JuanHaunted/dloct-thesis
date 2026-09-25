#!/bin/bash
# Train one experiment on all GPUs of one node.
#
#   sbatch scripts/train.slurm configs/unet_full.yaml [--set key=value ...]
#
# Resubmitting the same command resumes from runs/<name>/latest.pt.
# Defaults are for EAFIT Apolo-3: partition accel is a single node with 2 GPUs.
#SBATCH --job-name=dloct
#SBATCH --partition=accel            # Apolo-3 GPU node a3-accel-0 (2 GPUs, 64 CPUs, 257 GB)
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00          # accel partition maximum
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL          # email when the job ends or fails; address from SBATCH_MAIL_USER

set -euo pipefail
CONFIG=${1:?usage: sbatch scripts/train.slurm <config.yaml> [--set k=v ...]}
shift

# Point at the prepared data on fast storage, e.g. export DLOCT_DATA=/scratch/$USER/dloct/prepared
DATA_ROOT=${DLOCT_DATA:-data/prepared}

cd "${SLURM_SUBMIT_DIR}"
source scripts/env.sh
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")   # GPUs allocated to this job
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} / NGPU - 1 ))
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} gpus=${NGPU} workers/gpu=${WORKERS} config=${CONFIG} data=${DATA_ROOT}"
nvidia-smi

export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
torchrun --standalone --nproc_per_node="${NGPU}" -m dloct.train \
    --config "${CONFIG}" --set data.root="${DATA_ROOT}" data.num_workers="${WORKERS}" "$@"

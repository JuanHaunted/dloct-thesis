#!/bin/bash
# One-time environment setup on the cluster login node. Run from the repo root:
#   bash scripts/setup_cluster.sh
#
# Online (login node can reach PyPI): installs uv and syncs the locked environment.
# Offline: if ./wheelhouse exists (built locally with scripts/build_wheelhouse.sh and
# uploaded), creates a venv from Apolo's Python 3.12 module and installs from it.
set -euo pipefail
MODULE_PY=${MODULE_PY:-python/3.12_miniconda-24.7.1}

if curl -s --max-time 10 -o /dev/null https://pypi.org/simple/; then
    echo "internet: yes -> uv"
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv sync --frozen --no-dev
elif [ -d wheelhouse ]; then
    echo "internet: no -> offline install from ./wheelhouse with module ${MODULE_PY}"
    module load "${MODULE_PY}"
    python3 -m venv .venv
    echo "${MODULE_PY}" > .venv/.module-python
    .venv/bin/pip install --no-index --find-links wheelhouse -r wheelhouse/requirements.txt
    .venv/bin/pip install --no-index --find-links wheelhouse --no-deps -e .
else
    echo "No internet and no ./wheelhouse. Build it locally: bash scripts/build_wheelhouse.sh" >&2
    exit 1
fi

source scripts/env.sh
python -c "import torch, dloct; print('torch', torch.__version__, 'CUDA build', torch.version.cuda, 'archs', torch.cuda.get_arch_list())"
echo "OK. On a GPU node, check the driver (needs >= 525):"
echo "  srun -p accel --gres=gpu:1 -t 0-00:10:00 --pty nvidia-smi"

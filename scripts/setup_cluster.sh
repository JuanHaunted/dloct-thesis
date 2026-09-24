#!/bin/bash
# Run once on a cluster login node (needs internet): installs uv and the environment.
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
uv run python -c "import torch, dloct; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
echo "Driver must support CUDA >= 12.8 (driver >= 570). Check with: nvidia-smi on a GPU node."

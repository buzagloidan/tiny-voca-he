#!/bin/bash
# Runs ON the pod, in /workspace/tiny-voca-he, after launch.sh uploaded the project.
# Installs uv and a .venv (on top of the image's CUDA torch), and links the
# Mac path that ANNOTATIONS/annotations.json records as acoustic_root.
# Usage: setup_pod.sh [ANNOTATIONS]   (default work/annotations)
set -euo pipefail
cd "$(dirname "$0")/../.."
HERE=$(pwd); ANNOTATIONS=${1:-work/annotations}

ROOT=$(python3 -c "import json;print(json.load(open('$ANNOTATIONS/annotations.json'))['acoustic_root'])")
PROJECT=$(dirname "$(dirname "$ROOT")")
if [ "$PROJECT" != "$HERE" ]; then
  mkdir -p "$(dirname "$PROJECT")"
  ln -sfn "$HERE" "$PROJECT"
fi

# The RunPod image already has a CUDA torch (>= 2.4, as pyproject requires); reuse it.
# Installing torch on the pod is slow and timed out on pypi.nvidia.com, so it is
# only the fallback.
SYSTEM_TORCH_OK=$(python3 -c "import torch;v=tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]));print(int(torch.cuda.is_available() and v>=(2,4)))" 2>/dev/null || echo 0)
export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=600
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ "$SYSTEM_TORCH_OK" = 1 ]; then
  [ -x .venv/bin/python ] || uv venv -q --system-site-packages --python "$(command -v python3)" .venv
else
  [ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
  uv pip install -q --python .venv/bin/python torch
fi
uv pip install -q --python .venv/bin/python "numpy>=1.24,<3" "soundfile>=0.12,<1"
.venv/bin/python -c "import torch;assert torch.cuda.is_available();print('torch',torch.__version__,'on',torch.cuda.get_device_name(0))"

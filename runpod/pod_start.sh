#!/usr/bin/env bash
set -euo pipefail

echo "==== NVIDIA SMI ===="
nvidia-smi || { echo "nvidia-smi failed"; exit 1; }

echo "==== PyTorch / FAISS / Python ===="
python3 - <<'PY'
import platform

try:
    import torch
except ImportError:
    torch = None

try:
    import faiss
except ImportError:
    faiss = None

print("python_version:", platform.python_version())
print("torch_available:", bool(torch))
if torch:
    print("torch_version:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
print("faiss_available:", bool(faiss))
if faiss:
    print("faiss_version:", faiss.__version__)
PY

echo "==== PySceneDetect ===="
scenedetect --version

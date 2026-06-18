#!/usr/bin/env bash
set -euo pipefail

# Expected machine image:
# Hyperstack Server 22.04 LTS R550 CUDA 12.4
# GPU: NVIDIA L40

nvidia-smi
which nvcc || true
nvcc --version || true

# Install Miniconda if needed
if [ ! -d "$HOME/miniconda3" ]; then
  cd "$HOME"
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p "$HOME/miniconda3"
fi

eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"

# If needed on a fresh machine:
# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n sparge_celerity python=3.10 pip -y || true
conda activate sparge_celerity

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9"

python -m pip install --upgrade pip

python -m pip install \
  torch==2.6.0+cu124 \
  torchvision==0.21.0+cu124 \
  torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install \
  transformers==4.46.3 \
  tokenizers==0.20.3 \
  huggingface_hub==0.36.2 \
  accelerate==1.14.0 \
  safetensors \
  ninja \
  einops \
  packaging \
  setuptools \
  wheel

# Matplotlib for plots
python -m pip install matplotlib --no-deps
python -m pip install contourpy cycler fonttools kiwisolver pyparsing python-dateutil

# Install SpargeAttention
if [ ! -d "$HOME/SpargeAttn" ]; then
  cd "$HOME"
  git clone https://github.com/thu-ml/SpargeAttn.git
fi

cd "$HOME/SpargeAttn"
python -m pip install -e . --no-build-isolation

python - <<'PY'
import sys, torch, transformers, tokenizers, huggingface_hub, matplotlib
import spas_sage_attn
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("matplotlib:", matplotlib.__version__)
print("SpargeAttn import OK")
PY

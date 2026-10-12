#!/bin/bash
# ============================================================
# Circuit Tracing 项目 — 一键环境搭建
# 适用于：8x A100 集群 (CUDA 12.x)
# 用法：bash setup.sh
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="circuit_tracing"
PYTHON_VERSION="3.11"
MODELS_DIR="${PROJECT_DIR}/models"
TRANSCODERS_DIR="${PROJECT_DIR}/transcoders"

echo "============================================"
echo "  Circuit Tracing 环境搭建"
echo "  项目目录: ${PROJECT_DIR}"
echo "============================================"

# ------------------------------------------------------------
# 1. 创建 Conda 环境
# ------------------------------------------------------------
echo ""
echo "[1/6] 创建 Conda 环境 (${ENV_NAME}, Python ${PYTHON_VERSION})..."

if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "  环境 ${ENV_NAME} 已存在，跳过创建"
else
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# 激活环境 (在脚本中需要 source)
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "  Python: $(python --version)"
echo "  环境路径: $(which python)"

# ------------------------------------------------------------
# 2. 安装 PyTorch (CUDA 12.4)
# ------------------------------------------------------------
echo ""
echo "[2/6] 安装 PyTorch + CUDA..."

pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -c "import torch as T; print(f'  PyTorch {T.__version__}, CUDA available: {T.cuda.is_available()}, Devices: {T.cuda.device_count()}')"

# ------------------------------------------------------------
# 3. 安装 circuit-tracer 和依赖
# ------------------------------------------------------------
echo ""
echo "[3/6] 安装 circuit-tracer 和核心依赖..."

# circuit-tracer (Anthropic 开源)
if [ ! -d "${PROJECT_DIR}/circuit-tracer" ]; then
    git clone https://github.com/safety-research/circuit-tracer.git "${PROJECT_DIR}/circuit-tracer"
fi
cd "${PROJECT_DIR}/circuit-tracer"
git checkout 4bb8c0e   # v0.4.1, the revision the reported runs used
pip install -e .
cd "${PROJECT_DIR}"

# 核心依赖
pip install \
    transformers>=4.46.0 \
    accelerate \
    safetensors \
    datasets \
    huggingface_hub \
    sae-lens \
    nnsight \
    transformer_lens \
    einops \
    jaxtyping \
    plotly \
    networkx \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    tqdm \
    wandb \
    jupyter \
    ipywidgets

echo "  circuit-tracer 安装完成"

# ------------------------------------------------------------
# 4. 下载模型权重
# ------------------------------------------------------------
echo ""
echo "[4/6] 下载模型权重..."

mkdir -p ${MODELS_DIR}

# 需要先登录 HuggingFace (Gemma 需要 license 同意)
echo "  注意: Gemma 模型需要先在 https://huggingface.co/google/gemma-3-4b-it 点击同意 license"
echo "  然后运行: huggingface-cli login"
echo ""

# Qwen3 系列 (无需 license)
echo "  下载 Qwen3 模型..."
python -c "
from huggingface_hub import snapshot_download
import os

models = [
    'Qwen/Qwen3-0.6B',
    'Qwen/Qwen3-1.7B',
    'Qwen/Qwen3-4B',
    'Qwen/Qwen3-8B',
    'Qwen/Qwen3-14B',
]

cache_dir = '${MODELS_DIR}'
for model in models:
    print(f'  Downloading {model}...')
    try:
        snapshot_download(model, cache_dir=cache_dir)
        print(f'  ✓ {model}')
    except Exception as e:
        print(f'  ✗ {model}: {e}')
"

# Gemma3 系列 (需要 license)
echo "  下载 Gemma3 模型..."
python -c "
from huggingface_hub import snapshot_download

models = [
    'google/gemma-3-1b-it',
    'google/gemma-3-4b-it',
    'google/gemma-3-12b-it',
    'google/gemma-3-27b-it',
]

cache_dir = '${MODELS_DIR}'
for model in models:
    print(f'  Downloading {model}...')
    try:
        snapshot_download(model, cache_dir=cache_dir)
        print(f'  ✓ {model}')
    except Exception as e:
        print(f'  ✗ {model}: {e} (可能需要 huggingface-cli login 并同意 license)')
"

# ------------------------------------------------------------
# 5. 下载 Transcoder / SAE 权重
# ------------------------------------------------------------
echo ""
echo "[5/6] 下载 Transcoder 和 SAE 权重..."

mkdir -p ${TRANSCODERS_DIR}

# Qwen3 PLTs (mwhanna)
echo "  下载 Qwen3 Per-Layer Transcoders..."
python -c "
from huggingface_hub import snapshot_download

transcoders = [
    'mwhanna/qwen3-0.6b-transcoders',
    'mwhanna/qwen3-1.7b-transcoders',
    'mwhanna/qwen3-4b-transcoders',
    'mwhanna/qwen3-8b-transcoders',
    'mwhanna/qwen3-14b-transcoders',
]

cache_dir = '${TRANSCODERS_DIR}'
for tc in transcoders:
    print(f'  Downloading {tc}...')
    try:
        snapshot_download(tc, cache_dir=cache_dir)
        print(f'  ✓ {tc}')
    except Exception as e:
        print(f'  ✗ {tc}: {e}')
"

# Qwen3 CLTs (BluelightAI)
echo "  下载 Qwen3 Cross-Layer Transcoders (BluelightAI)..."
python -c "
from huggingface_hub import snapshot_download

clts = [
    'bluelightai/qwen3-0.6b-base-clt',
    'bluelightai/qwen3-1.7b-base-clt',
]

cache_dir = '${TRANSCODERS_DIR}'
for clt in clts:
    print(f'  Downloading {clt}...')
    try:
        snapshot_download(clt, cache_dir=cache_dir)
        print(f'  ✓ {clt}')
    except Exception as e:
        print(f'  ✗ {clt}: {e}')
"

# Gemma Scope 2 SAEs + Transcoders
echo "  下载 Gemma Scope 2 (SAE + Transcoder)..."
echo "  注意: 完整下载非常大 (数百 GB)，这里只下载推荐的 64k/medium 配置"
python -c "
from huggingface_hub import snapshot_download

# 下载 Gemma Scope 2 的核心文件
# 注意: 这些 repo 很大，我们选择性下载
scope_repos = [
    'google/gemma-scope-2-4b-it',
    'google/gemma-scope-2-12b-it',
    'google/gemma-scope-2-27b-it',
]

cache_dir = '${TRANSCODERS_DIR}'
for repo in scope_repos:
    print(f'  Downloading {repo} (transcoder_all only)...')
    try:
        snapshot_download(
            repo,
            cache_dir=cache_dir,
            allow_patterns=['transcoder_all/*width_64k*medium*'],
        )
        print(f'  ✓ {repo}')
    except Exception as e:
        print(f'  ✗ {repo}: {e}')
"

# ------------------------------------------------------------
# 6. 创建项目结构
# ------------------------------------------------------------
echo ""
echo "[6/6] 创建项目结构..."

mkdir -p "${PROJECT_DIR}/experiments"
mkdir -p "${PROJECT_DIR}/experiments/01_hello_world"
mkdir -p "${PROJECT_DIR}/experiments/02_reproduce_anthropic"
mkdir -p "${PROJECT_DIR}/experiments/03_cross_lingual"
mkdir -p "${PROJECT_DIR}/experiments/04_thinking_mode"
mkdir -p "${PROJECT_DIR}/experiments/05_cross_model"
mkdir -p "${PROJECT_DIR}/results"
mkdir -p "${PROJECT_DIR}/results/attribution_graphs"
mkdir -p "${PROJECT_DIR}/results/figures"
mkdir -p "${PROJECT_DIR}/notes"
mkdir -p "${PROJECT_DIR}/scripts"
mkdir -p "${PROJECT_DIR}/paper"

echo "============================================"
echo "  环境搭建完成!"
echo ""
echo "  激活环境: conda activate ${ENV_NAME}"
echo ""
echo "  项目结构:"
echo "  ├── circuit-tracer/        # Anthropic 开源库"
echo "  ├── models/                # 模型权重缓存"
echo "  ├── transcoders/           # Transcoder/SAE 权重"
echo "  ├── experiments/"
echo "  │   ├── 01_hello_world/    # 跑通 demo"
echo "  │   ├── 02_reproduce_anthropic/  # 复现经典 case"
echo "  │   ├── 03_cross_lingual/  # 中英跨语言实验"
echo "  │   ├── 04_thinking_mode/  # Qwen thinking mode"
echo "  │   └── 05_cross_model/    # Qwen vs Gemma 对比"
echo "  ├── results/               # 实验结果"
echo "  ├── notes/                 # 阅读笔记"
echo "  ├── scripts/               # SLURM 脚本"
echo "  └── paper/                 # 论文 LaTeX"
echo "============================================"

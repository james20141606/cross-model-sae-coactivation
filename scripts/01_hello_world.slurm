#!/bin/bash
#SBATCH --job-name=ct-hello
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/01_hello_world_%j.out
#SBATCH --error=logs/01_hello_world_%j.err

# ============================================================
# 实验 01: Hello World — 环境验证
# GPU: 1x A100, 预计 10-15 分钟
# ============================================================

set -e
PROJECT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd ${PROJECT_DIR}
mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Project: ${PROJECT_DIR}"
echo ""

# 激活环境
source activate circuit_tracing 2>/dev/null || conda activate circuit_tracing

# 运行
cd experiments/01_hello_world
python run.py --model qwen3-0.6b --device cuda:0 --prompt "The opposite of small is"

echo ""
echo "Hello world test with Qwen3-4B..."
python run.py --model qwen3-4b --device cuda:0 --prompt "The opposite of small is"

echo ""
# Phase 1.4 加测: 验证 Qwen3 tokenizer 是否支持 enable_thinking kwarg
# 这步零 GPU 成本预先 de-risk 实验 04 的 thinking-mode prompt 构造
echo ""
echo "==== Tokenizer enable_thinking pre-check ===="
cd ${PROJECT_DIR}
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
msgs = [{'role': 'user', 'content': 'What is 2+2?'}]
try:
    out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    has_think = '<think>' in out or '<|think' in out
    print(f'  enable_thinking kwarg: SUPPORTED')
    print(f'  Contains <think> marker: {has_think}')
    print(f'  Sample: {out[:200]}...')
except TypeError as e:
    print(f'  enable_thinking kwarg: UNSUPPORTED ({e})')
    print(f'  Experiment 04 will use the manual <think> fallback path')
" 2>&1
echo "==== End pre-check ===="

echo ""
echo "01_hello_world 完成!"

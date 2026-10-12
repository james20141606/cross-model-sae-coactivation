#!/bin/bash
#SBATCH --job-name=ct-repr-qwen
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/02_reproduce_qwen_%j.out
#SBATCH --error=logs/02_reproduce_qwen_%j.err

# ============================================================
# 实验 02: 在 Qwen3 上复现 Anthropic 经典 Case Study
# GPU: 2x A100
#   - GPU 0: Qwen3-4B (antonyms + multistep + hallucination)
#   - GPU 1: Qwen3-8B (antonyms + multistep + hallucination)
# 预计 4-6 小时
# ============================================================

set -e
PROJECT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd ${PROJECT_DIR}
mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

source activate circuit_tracing 2>/dev/null || conda activate circuit_tracing

cd experiments/02_reproduce_anthropic

# 并行运行不同模型
echo "Starting Qwen3-4B on GPU 0..."
python run.py --model qwen3-4b --experiment all --device cuda:0 &
PID1=$!

echo "Starting Qwen3-8B on GPU 1..."
python run.py --model qwen3-8b --experiment all --device cuda:1 &
PID2=$!

# 等待完成
wait $PID1
echo "Qwen3-4B 完成"
wait $PID2
echo "Qwen3-8B 完成"

echo "02_reproduce_qwen 全部完成!"

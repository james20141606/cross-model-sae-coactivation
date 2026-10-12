#!/bin/bash
#SBATCH --job-name=ct-think
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/04_thinking_mode_%j.out
#SBATCH --error=logs/04_thinking_mode_%j.err

# ============================================================
# 实验 04: Qwen3 Thinking Mode 电路对比
# GPU: 2x A100
#   - GPU 0: Qwen3-4B
#   - GPU 1: Qwen3-8B
# 预计 4-6 小时
# ============================================================

set -e
PROJECT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd ${PROJECT_DIR}
mkdir -p logs

source activate circuit_tracing 2>/dev/null || conda activate circuit_tracing

cd experiments/04_thinking_mode

python run.py --model qwen3-4b --device cuda:0 &
PID1=$!

python run.py --model qwen3-8b --device cuda:1 &
PID2=$!

wait $PID1 && echo "Qwen3-4B 完成"
wait $PID2 && echo "Qwen3-8B 完成"

echo "04_thinking_mode 全部完成!"

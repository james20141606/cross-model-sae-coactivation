#!/bin/bash
#SBATCH --job-name=ct-repr-gemma
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/02_reproduce_gemma_%j.out
#SBATCH --error=logs/02_reproduce_gemma_%j.err

# ============================================================
# 实验 02: 在 Gemma3 上复现 Anthropic 经典 Case Study
# GPU: 2x A100
#   - GPU 0: Gemma3-4B
#   - GPU 1: Gemma3-12B
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

echo "Starting Gemma3-4B on GPU 0..."
python run.py --model gemma3-4b --experiment all --device cuda:0 &
PID1=$!

echo "Starting Gemma3-12B on GPU 1..."
python run.py --model gemma3-12b --experiment all --device cuda:1 &
PID2=$!

wait $PID1
echo "Gemma3-4B 完成"
wait $PID2
echo "Gemma3-12B 完成"

echo "02_reproduce_gemma 全部完成!"

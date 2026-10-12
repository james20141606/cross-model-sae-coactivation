#!/bin/bash
#SBATCH --job-name=ct-crossling
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/03_cross_lingual_%j.out
#SBATCH --error=logs/03_cross_lingual_%j.err

# ============================================================
# 实验 03: 跨语言 Feature Sharing 分析 (核心贡献)
# GPU: 4x A100
#   - GPU 0: Qwen3-4B
#   - GPU 1: Qwen3-14B
#   - GPU 2: Gemma3-4B
#   - GPU 3: Gemma3-12B
# 预计 8-12 小时
# ============================================================

set -e
PROJECT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd ${PROJECT_DIR}
mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

source activate circuit_tracing 2>/dev/null || conda activate circuit_tracing

cd experiments/03_cross_lingual

# 4 个模型并行
echo "Starting cross-lingual analysis on 4 models..."

python run.py --model qwen3-4b --device cuda:0 --include-chinese-reasoning &
PID1=$!

python run.py --model qwen3-14b --device cuda:1 &
PID2=$!

python run.py --model gemma3-4b --device cuda:2 &
PID3=$!

python run.py --model gemma3-12b --device cuda:3 &
PID4=$!

wait $PID1 && echo "Qwen3-4B 完成"
wait $PID2 && echo "Qwen3-14B 完成"
wait $PID3 && echo "Gemma3-4B 完成"
wait $PID4 && echo "Gemma3-12B 完成"

echo "03_cross_lingual 全部完成!"

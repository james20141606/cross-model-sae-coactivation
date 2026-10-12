#!/bin/bash
#SBATCH --job-name=ct-xmodel
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/05_cross_model_%j.out
#SBATCH --error=logs/05_cross_model_%j.err

# ============================================================
# 实验 05: 跨模型 Scaling 分析 (最大实验)
# GPU: 8x A100 全部使用
# 运行顺序: Qwen scaling → Gemma scaling → matched pairs
#
# 注意: 这个实验会依次加载从小到大的模型，
# 每次只用所需 GPU 数量，但申请 8 张以便跑大模型。
# 预计 16-24 小时
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

cd experiments/05_cross_model

# 阶段 1: Qwen3 Scaling (0.6B → 14B)
echo ""
echo "=========================================="
echo "  阶段 1/3: Qwen3 Scaling Analysis"
echo "=========================================="
python run.py --mode qwen_scaling --device cuda:0 --skip-large

# 阶段 2: Gemma3 Scaling (4B → 27B)
echo ""
echo "=========================================="
echo "  阶段 2/3: Gemma3 Scaling Analysis"
echo "=========================================="
python run.py --mode gemma_scaling --device cuda:0 --skip-large

# 阶段 3: 同规模 Head-to-Head
echo ""
echo "=========================================="
echo "  阶段 3/3: Matched Pair Comparison"
echo "=========================================="
python run.py --mode matched_pairs --device cuda:0 --skip-large

echo ""
echo "05_cross_model 全部完成!"

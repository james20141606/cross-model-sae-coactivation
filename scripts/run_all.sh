#!/bin/bash
# ============================================================
# 主调度脚本 — 按顺序提交所有 SLURM 作业
#
# 用法:
#   bash scripts/run_all.sh          # 提交所有实验
#   bash scripts/run_all.sh --dev    # 只跑 dev mode 验证
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DEV_MODE=false
if [ "$1" = "--dev" ]; then
    DEV_MODE=true
fi

echo "============================================"
echo "  Circuit Tracing 实验调度"
echo "  项目: ${PROJECT_DIR}"
echo "  DEV_MODE: ${DEV_MODE}"
echo "============================================"

if [ "$DEV_MODE" = true ]; then
    echo "提交 DEV 验证作业..."
    JOB1=$(sbatch --parsable ${SCRIPT_DIR}/01_hello_world.slurm)
    echo "  01_hello_world: Job ${JOB1}"
    echo "  完成! 等待 Job ${JOB1} 运行结束后检查输出。"
    exit 0
fi

# --- 阶段 1: Hello World 验证 (1 GPU, ~10min) ---
echo ""
echo "[阶段 1] Hello World 验证..."
JOB1=$(sbatch --parsable ${SCRIPT_DIR}/01_hello_world.slurm)
echo "  Job ${JOB1} submitted"

# --- 阶段 2: 复现 Anthropic (依赖阶段 1) ---
echo ""
echo "[阶段 2] 复现 Anthropic 经典 Case Study..."
# Qwen3 系列 (并行在不同 GPU 上)
JOB2a=$(sbatch --parsable --dependency=afterok:${JOB1} ${SCRIPT_DIR}/02_reproduce_qwen.slurm)
echo "  Qwen3 复现: Job ${JOB2a}"
# Gemma3 系列
JOB2b=$(sbatch --parsable --dependency=afterok:${JOB1} ${SCRIPT_DIR}/02_reproduce_gemma.slurm)
echo "  Gemma3 复现: Job ${JOB2b}"

# --- 阶段 3: 跨语言分析 (依赖阶段 2) ---
echo ""
echo "[阶段 3] 跨语言 Feature Sharing..."
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2a}:${JOB2b} ${SCRIPT_DIR}/03_cross_lingual.slurm)
echo "  跨语言分析: Job ${JOB3}"

# --- 阶段 4: Thinking Mode (可与阶段 3 并行) ---
echo ""
echo "[阶段 4] Thinking Mode 对比..."
JOB4=$(sbatch --parsable --dependency=afterok:${JOB2a} ${SCRIPT_DIR}/04_thinking_mode.slurm)
echo "  Thinking Mode: Job ${JOB4}"

# --- 阶段 5: 跨模型 Scaling (依赖阶段 2) ---
echo ""
echo "[阶段 5] 跨模型 Scaling 分析..."
JOB5=$(sbatch --parsable --dependency=afterok:${JOB2a}:${JOB2b} ${SCRIPT_DIR}/05_cross_model.slurm)
echo "  跨模型 Scaling: Job ${JOB5}"

echo ""
echo "============================================"
echo "  所有作业已提交!"
echo ""
echo "  依赖关系:"
echo "  01_hello_world (${JOB1})"
echo "    ├── 02_reproduce_qwen (${JOB2a})"
echo "    │     ├── 03_cross_lingual (${JOB3})"
echo "    │     ├── 04_thinking_mode (${JOB4})"
echo "    │     └── 05_cross_model (${JOB5})"
echo "    └── 02_reproduce_gemma (${JOB2b})"
echo "          ├── 03_cross_lingual (${JOB3})"
echo "          └── 05_cross_model (${JOB5})"
echo ""
echo "  监控: squeue -u \$USER"
echo "  日志: ls ${PROJECT_DIR}/logs/"
echo "============================================"

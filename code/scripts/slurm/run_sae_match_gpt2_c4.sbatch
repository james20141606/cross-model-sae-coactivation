#!/bin/bash
#SBATCH --job-name=ct-match-gpt2-c4
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%A_%a.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --array=0-3
# R18 / AC-review T1: GPT-2 has NO C4 natural-text mutual-NN receipt. The paper's
# "capability threshold" therefore rests on ONE model measured below null on natural text
# (Pythia-70M); GPT-2's sub-threshold status is inferred from the TEMPLATED corpus that
# Finding 1 itself shows mis-estimates sharing in BOTH directions. This array runs the four
# GPT-2 cells on the identical 2000-doc C4 slice / topk=1500 / n_perm=200 configuration as
# the c4_* headline roster, so the template-vs-C4 comparison is exact for GPT-2 too.
# GPT-2's public SAE carries the disclosed TransformerLens-preprocessing caveat either way.
#
# SMOKE=1 -> tiny de-risk run (60 docs, n_perm=20) written to smoke_*.json.
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_ANALYSIS}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN=$(cat $HOME/.cache/huggingface/token 2>/dev/null || true)
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/results/cross_model_sae"
cd "$ROOT/code/analysis"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-0} SMOKE=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

GPT2_R=gpt2-small-res-jb; GPT2_I=blocks.6.hook_resid_pre
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) A=gpt2; RA=$GPT2_R; IA=$GPT2_I
     B=mistralai/Mistral-7B-v0.1; RB=mistral-7b-res-wg; IB=blocks.16.hook_resid_pre
     TAG=gpt2__mistral7b ;;
  1) A=google/gemma-2-2b; RA=gemma-scope-2b-pt-res-canonical; IA=layer_12/width_16k/canonical
     B=gpt2; RB=$GPT2_R; IB=$GPT2_I
     TAG=gemma2-2b__gpt2 ;;
  2) A=google/gemma-2-9b; RA=gemma-scope-9b-pt-res-canonical; IA=layer_20/width_16k/canonical
     B=gpt2; RB=$GPT2_R; IB=$GPT2_I
     TAG=gemma2-9b__gpt2 ;;
  3) A=Qwen/Qwen3-1.7B-Base; RA=qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50; IA=layer14
     B=gpt2; RB=$GPT2_R; IB=$GPT2_I
     TAG=qwen3-1.7b__gpt2 ;;
  *) echo "unknown array task"; exit 2 ;;
esac

if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA="--max_prompts 60 --n_perm 20"
  OUT="$ROOT/results/cross_model_sae/smoke_c4_${TAG}.json"
else
  EXTRA="--n_perm 200"
  OUT="$ROOT/results/cross_model_sae/c4_${TAG}.json"
fi

"$ENV_PY" -u cross_model_sae_match.py \
  --model_a "$A" --sae_a_release "$RA" --sae_a_id "$IA" \
  --model_b "$B" --sae_b_release "$RB" --sae_b_id "$IB" \
  --prompts ../../data/c4_slice.jsonl --lang en --topk 1500 $EXTRA \
  --output "$OUT"
echo "[DONE] $(date -Is) -> $OUT"

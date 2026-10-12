#!/bin/bash
#SBATCH --job-name=ct-probe-xferC4
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%j.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1 --time=01:30:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
# R13 hardening: probe transfer with mutual-NN matching fit on the INDEPENDENT C4 slice
# (kills the same-corpus circularity objection). PAIR env var selects the pair.
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_ANALYSIS}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN=$(cat $HOME/.cache/huggingface/token 2>/dev/null || true)
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/results/cross_model_sae"
cd "$ROOT/code/analysis"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} pair=${PAIR:?set PAIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

case "$PAIR" in
  g9m)  A=google/gemma-2-9b;  RA=gemma-scope-9b-pt-res-canonical; IA=layer_20/width_16k/canonical
        B=mistralai/Mistral-7B-v0.1; RB=mistral-7b-res-wg; IB=blocks.16.hook_resid_pre ;;
  q1g2) A=Qwen/Qwen3-1.7B-Base; RA=qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50; IA=layer14
        B=google/gemma-2-2b; RB=gemma-scope-2b-pt-res-canonical; IB=layer_12/width_16k/canonical ;;
  q8m)  A=Qwen/Qwen3-8B-Base; RA=qwenpt:Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50; IA=layer18
        B=mistralai/Mistral-7B-v0.1; RB=mistral-7b-res-wg; IB=blocks.16.hook_resid_pre ;;
  gpt2m) A=gpt2; RA=gpt2-small-res-jb; IA=blocks.6.hook_resid_pre
        B=mistralai/Mistral-7B-v0.1; RB=mistral-7b-res-wg; IB=blocks.16.hook_resid_pre ;;
  *) echo "unknown PAIR=$PAIR"; exit 2 ;;
esac
TAG=$(echo "$PAIR" | tr / _)
# MATCH_PROMPTS: matching-fit corpus (default = full 2000-doc C4 slice; set to
# ../../data/c4_heldout.jsonl for the held-out replication). OUT_SUFFIX distinguishes outputs.
MATCH_PROMPTS="${MATCH_PROMPTS:-../../data/c4_slice.jsonl}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
"$ENV_PY" -u cross_model_probe_transfer.py \
  --model_a "$A" --sae_a_release "$RA" --sae_a_id "$IA" \
  --model_b "$B" --sae_b_release "$RB" --sae_b_id "$IB" \
  --prompts ../../data/prompts_v3_multi.jsonl --lang en \
  --match_prompts "$MATCH_PROMPTS" \
  --output "../../results/cross_model_sae/probe_transfer_c4match_${TAG}${OUT_SUFFIX}.json"

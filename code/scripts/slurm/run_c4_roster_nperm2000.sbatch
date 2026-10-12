#!/bin/bash
#SBATCH --job-name=ct-c4-2k
#SBATCH --output=<REPO_ROOT>/logs/xmodel_sae/%x-%A_%a.out
#SBATCH --error=<REPO_ROOT>/logs/xmodel_sae/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1 --time=03:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --array=0-16
# Re-run the ENTIRE C4 roster at n_perm=2000 to remove the standing multiplicity objection
# (T6). At the primary n_perm=200 every positive sits at the p floor 1/201=0.004975, which
# does NOT clear Bonferroni over 13 cells (0.05/13=0.003846). At n_perm=2000 the floor is
# 1/2001=0.000500, which clears both the 13-cell roster and the 17-cell count including the
# four GPT-2 cells (0.05/17=0.002941).
#
# Outputs are written to c4n2k_*.json -- a SEPARATE namespace, so the published
# n_perm=200 receipts are never overwritten and the two can be compared cell by cell.
# 80 GiB+ partitions: several cells hold two 7-9B models.
#
# SMOKE=1 -> 200 docs, n_perm=100, to smoke_c4n2k_*.json.
set -uo pipefail
ROOT=${REPO_ROOT}
ENV_PY=${ENV_ANALYSIS}/bin/python
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=${HF_CACHE}/hub
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
mkdir -p "$ROOT/logs/xmodel_sae" "$ROOT/results/cross_model_sae"
cd "$ROOT/code/analysis"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-0} SMOKE=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

GS2=gemma-scope-2b-pt-res-canonical;  IG2=layer_12/width_16k/canonical
GS9=gemma-scope-9b-pt-res-canonical;  IG9=layer_20/width_16k/canonical
MIS=mistral-7b-res-wg;                IMI=blocks.16.hook_resid_pre
Q17=qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50; IQ17=layer14
Q8=qwenpt:Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50;    IQ8=layer18
GPT2R=gpt2-small-res-jb;              IGPT2=blocks.6.hook_resid_pre

# the full published roster: 2 within + 9 cross-family + recipe check + Pythia + 4 GPT-2
case "${SLURM_ARRAY_TASK_ID:-0}" in
   0) A=google/gemma-2-2b;RA=$GS2;IA=$IG2;   B=google/gemma-2-9b;RB=$GS9;IB=$IG9;  TAG=gemma2-2b__gemma2-9b ;;
   1) A=Qwen/Qwen3-8B-Base;RA=$Q8;IA=$IQ8;   B=Qwen/Qwen3-1.7B-Base;RB=$Q17;IB=$IQ17; TAG=qwen3-8b__qwen3-1.7b ;;
   2) A=Qwen/Qwen3-1.7B-Base;RA=$Q17;IA=$IQ17; B=google/gemma-2-2b;RB=$GS2;IB=$IG2; TAG=qwen3-1.7b__gemma2-2b ;;
   3) A=Qwen/Qwen3-8B-Base;RA=$Q8;IA=$IQ8;   B=google/gemma-2-9b;RB=$GS9;IB=$IG9;  TAG=qwen3-8b__gemma2-9b ;;
   4) A=Qwen/Qwen3-8B-Base;RA=$Q8;IA=$IQ8;   B=google/gemma-2-2b;RB=$GS2;IB=$IG2;  TAG=qwen3-8b__gemma2-2b ;;
   5) A=Qwen/Qwen3-1.7B-Base;RA=$Q17;IA=$IQ17; B=google/gemma-2-9b;RB=$GS9;IB=$IG9; TAG=qwen3-1.7b__gemma2-9b ;;
   6) A=google/gemma-2-9b;RA=$GS9;IA=$IG9;   B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=gemma2-9b__mistral7b ;;
   7) A=Qwen/Qwen3-8B-Base;RA=$Q8;IA=$IQ8;   B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=qwen3-8b__mistral7b ;;
   8) A=google/gemma-2-2b;RA=$GS2;IA=$IG2;   B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=gemma2-2b__mistral7b ;;
   9) A=Qwen/Qwen3-1.7B-Base;RA=$Q17;IA=$IQ17; B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=qwen3-1.7b__mistral7b ;;
  10) A=meta-llama/Llama-3.2-1B;RA=chanind/sae-llama-3.2-1b-res;IA=blocks.8.hook_resid_post; B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=llama3.2-1b__mistral7b ;;
  11) A=EleutherAI/pythia-70m-deduped;RA=pythia-70m-deduped-res-sm;IA=blocks.3.hook_resid_post; B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=pythia70m__mistral7b ;;
  12) A=google/gemma-2-2b;RA=$GS2;IA=layer_12/width_65k/canonical; B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=w65k_gemma2-2b__mistral7b ;;
  13) A=gpt2;RA=$GPT2R;IA=$IGPT2;           B=mistralai/Mistral-7B-v0.1;RB=$MIS;IB=$IMI; TAG=gpt2__mistral7b ;;
  14) A=google/gemma-2-2b;RA=$GS2;IA=$IG2;  B=gpt2;RB=$GPT2R;IB=$IGPT2; TAG=gemma2-2b__gpt2 ;;
  15) A=google/gemma-2-9b;RA=$GS9;IA=$IG9;  B=gpt2;RB=$GPT2R;IB=$IGPT2; TAG=gemma2-9b__gpt2 ;;
  16) A=Qwen/Qwen3-1.7B-Base;RA=$Q17;IA=$IQ17; B=gpt2;RB=$GPT2R;IB=$IGPT2; TAG=qwen3-1.7b__gpt2 ;;
   *) echo "unknown task"; exit 2 ;;
esac

if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA="--max_prompts 200 --n_perm 100"
  OUT="$ROOT/results/cross_model_sae/smoke_c4n2k_${TAG}.json"
else
  EXTRA="--n_perm 2000"
  OUT="$ROOT/results/cross_model_sae/c4n2k_${TAG}.json"
fi

"$ENV_PY" -u cross_model_sae_match.py \
  --model_a "$A" --sae_a_release "$RA" --sae_a_id "$IA" \
  --model_b "$B" --sae_b_release "$RB" --sae_b_id "$IB" \
  --prompts ../../data/c4_slice.jsonl --lang en --topk 1500 $EXTRA \
  --output "$OUT"
RC=$?
if [ $RC -ne 0 ] || [ ! -s "$OUT" ]; then
  echo "[FAIL] rc=$RC output=$OUT exists=$([ -s "$OUT" ] && echo yes || echo no)" >&2
  exit 1
fi
echo "[DONE] $(date -Is) -> $OUT"

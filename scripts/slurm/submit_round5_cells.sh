#!/bin/bash
# Round-5 staged cells — submit from the cluster when auth is back:
#   bash scripts/slurm/submit_round5_cells.sh [smoke|full]
# smoke: MAX_PROMPTS=20 N_PERM=20 on one cell of each kind first (debug-gpu discipline).
#
# Cell groups:
#  (1) CROSS-LINGUAL SAE matching (the decisive title-question cells):
#      1a within-model cross-lingual: same model+SAE, lang_a=en vs lang_b in {zh,ja,ko}
#         -> does one model's OWN feature basis align across languages on matched concepts?
#      1b cross-model per-language: gemma2-2b vs gemma2-9b / mistral on zh (vs the en cell we have)
#  (2) HARDEN the positive: n_perm 200->2000 on the two headline pairs
#  (3) C4 template-confound control: rerun headline pairs on data/c4_slice.jsonl
#      (build first: $ENV_PY code/analysis/build_c4_slice.py --n 2000)
set -uo pipefail
cd ${REPO_ROOT}
MODE="${1:-full}"

G2B="MODEL_A=google/gemma-2-2b SAE_A_RELEASE=gemma-scope-2b-pt-res-canonical SAE_A_ID=layer_12/width_16k/canonical"
G9B_B="MODEL_B=google/gemma-2-9b SAE_B_RELEASE=gemma-scope-9b-pt-res-canonical SAE_B_ID=layer_20/width_16k/canonical"
MIS_B="MODEL_B=mistralai/Mistral-7B-v0.1 SAE_B_RELEASE=mistral-7b-res-wg SAE_B_ID=blocks.16.hook_resid_pre"
G2B_B="MODEL_B=google/gemma-2-2b SAE_B_RELEASE=gemma-scope-2b-pt-res-canonical SAE_B_ID=layer_12/width_16k/canonical"

SMOKE=""
[ "$MODE" = smoke ] && SMOKE="MAX_PROMPTS=20 N_PERM=20 TOPK=500"

SUB() { env $1 sbatch scripts/slurm/run_cross_model_sae.sbatch; }

if [ "$MODE" = smoke ]; then
  # one smoke per cell kind
  SUB "$G2B $G2B_B LANG_A=en LANG_B=zh $SMOKE TAG=smoke_xling_gemma2-2b_en_zh"
  exit 0
fi

# (1a) within-model cross-lingual, gemma-2-2b and gemma-2-9b, en vs zh/ja/ko
for L in zh ja ko; do
  SUB "$G2B $G2B_B LANG_A=en LANG_B=$L TAG=xling_gemma2-2b_en_${L}"
  SUB "MODEL_A=google/gemma-2-9b SAE_A_RELEASE=gemma-scope-9b-pt-res-canonical SAE_A_ID=layer_20/width_16k/canonical $G9B_B LANG_A=en LANG_B=$L TAG=xling_gemma2-9b_en_${L}"
done
# (1b) cross-model per-language (zh) for the two headline pairs
SUB "$G2B $G9B_B PROMPT_LANG=zh TAG=zh_gemma2-2b__gemma2-9b"
SUB "$G2B $MIS_B PROMPT_LANG=zh TAG=zh_gemma2-2b__mistral7b"
# (2) n_perm=2000 hardening on headline pairs
SUB "$G2B $G9B_B N_PERM=2000 TAG=hard2000_gemma2-2b__gemma2-9b"
SUB "MODEL_A=google/gemma-2-9b SAE_A_RELEASE=gemma-scope-9b-pt-res-canonical SAE_A_ID=layer_20/width_16k/canonical $MIS_B N_PERM=2000 TAG=hard2000_gemma2-9b__mistral7b"
# (3) C4 control (build data/c4_slice.jsonl first)
if [ -s data/c4_slice.jsonl ]; then
  SUB "$G2B $G9B_B PROMPTS=data/c4_slice.jsonl MAX_PROMPTS=2000 TAG=c4_gemma2-2b__gemma2-9b"
  SUB "MODEL_A=google/gemma-2-9b SAE_A_RELEASE=gemma-scope-9b-pt-res-canonical SAE_A_ID=layer_20/width_16k/canonical $MIS_B PROMPTS=data/c4_slice.jsonl MAX_PROMPTS=2000 TAG=c4_gemma2-9b__mistral7b"
else
  echo "SKIP C4 cells: data/c4_slice.jsonl missing (run build_c4_slice.py first)"
fi

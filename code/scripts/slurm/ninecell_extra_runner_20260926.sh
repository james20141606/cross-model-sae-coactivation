#!/bin/bash
# extra analyze_a runner inside JID 18577359 (srun --overlap); cells given as args, skip-existing
RUN=$HOME/ninecell_20260926; SHM=/dev/shm/${USER}_nc_18577359
export NC_C4=${REPO_ROOT}/data/c4_slice.jsonl NC_RECEIPTS=$RUN/receipts
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd $RUN/code/analysis
for C in "$@"; do
  ${ENV_ANALYSIS}/bin/python -u ninecell_strengthen_20260926.py analyze_a --cell $C \
    --pooled $SHM/pooled --out $RUN/results/ninecell_20260926/A_$C.json --device cuda || echo "[FAIL] $C"
done

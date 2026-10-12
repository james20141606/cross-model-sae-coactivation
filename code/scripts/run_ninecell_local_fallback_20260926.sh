#!/bin/bash
# prereg deviation 1 (b196641): local CPU fallback, 5 cached cells, no R bootstrap
set -u
R=${REPO_ROOT}
PY=$HOME/.venvs/lanfilt/bin/python
export PYTHONPATH=/tmp/ct_sk NC_C4=$HOME/lanfilt_cache/c4_slice.jsonl
OUT=$R/results/ninecell_20260926/local_mps_cache; L=$OUT/logs
CELLS="qwen3-1.7b__gemma2-2b gemma2-2b__mistral7b qwen3-1.7b__mistral7b qwen3-1.7b__gemma2-9b gemma2-9b__mistral7b"
cd $R/code/analysis
(
for C in $CELLS; do
  OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY -u ninecell_strengthen_20260926.py analyze_b \
    --cell $C --pooled /tmp/ct_local/pooled --out $OUT/B_$C.json --platform local_mps_cache > $L/B_$C.log 2>&1
  echo "[B] $C rc=$? $(date -Is)" >> $L/progress.log
done ) &
for C in $CELLS; do
  OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY -u ninecell_strengthen_20260926.py analyze_a \
    --cell $C --pooled /tmp/ct_local/pooled --out $OUT/A_$C.json --device numpy --n_boot 0 --platform local_mps_cache > $L/A_$C.log 2>&1
  echo "[A] $C rc=$? $(date -Is)" >> $L/progress.log
done
wait
nice -n 10 $PY -u ninecell_strengthen_20260926.py summarize --dir $OUT > $L/summarize.log 2>&1
echo "[S] rc=$? $(date -Is)" >> $L/progress.log

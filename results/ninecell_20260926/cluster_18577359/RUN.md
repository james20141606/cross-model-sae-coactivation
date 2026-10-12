# ninecell_20260926 / cluster_18577359 — run record (governs; local_mps_cache is a consistency check)

Prereg `notes/PREREG_ninecell_ladder_20260926.md` (3a0b358). Deviation 1 (local fallback) does not
apply here: this is the registered nine-cell design, R bootstrap included.

- **JID 18577359**
  - Account CLUSTER_ACCOUNT, 1x NVIDIA A100-SXM4-80GB.
  - Started 2026-09-26 17:18 (+08), COMPLETED 00:56:45, exit 0:0.
  - A duplicate, JID 18577418 (general account), was submitted while the queue was blocked on QOSGrpGRES and cancelled before it started.
- **sbatch:** `code/scripts/slurm/run_ninecell_strengthen_20260926.sbatch`. Code and the 18 receipts were rsynced to `~/ninecell_20260926`. Pooled features were kept in /dev/shm and deleted at exit.
- **Script:**
  - Extraction, all B receipts, and A for Q1.7-G2 and Q1.7-G9: commit f403f20 (sha256 1d535dd6...).
  - A for the other seven cells: commit 742715d (sha256 f5ff70f9...). This version only adds skip-if-receipt-exists and an atomic write. No statistic changed.
- **Why 742715d.** GPU utilisation on the single sequential A loop was measured at ~35% (srun nvidia-smi samples 28/30/0/79/42/36%). Two extra runners were started inside the same allocation with `srun --overlap` (`code/scripts/slurm/ninecell_extra_runner_20260926.sh`), working through cells from the end of the list. After that, samples read 84-100%.
- **Duplicate A computations.** Q8-G2 and G9-M were each computed twice, by the main loop and by a runner. Seeds are fixed, so both runs are the same computation; the last write wins.
- **Features:** published `compute_pooled_features`, original google/meta-llama/Qwen/mistralai weights.
  - The six `*.meta.json` files carry sha256 and wall time.
  - Extraction took 79-179 s per model.
- **Host fields:** the `host` fields in the receipts named the cluster node; in this release they read `<node>`.

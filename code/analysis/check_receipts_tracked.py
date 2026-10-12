#!/usr/bin/env python3
"""Every receipt the paper cites must be committed, not merely present on this laptop.

`results/*` is gitignored (outputs are regenerable), so `git add -A` silently skips new
receipts. That has now bitten twice: EleutherAI_pythia-1b-deduped.json was missing, which
made probe_residual_stats.py unable to reproduce its own STATS.json; and the six fine-topic
plus three random-dictionary receipts were cited in the paper while untracked, which would
have made emit_covctl_table.py crash on a clean clone. Both were found by audit, not by the
tooling. This closes that hole.

Checks:
  1. every \\texttt{...json} and results/... path cited in paper/sections/*.tex resolves,
     and every resolved file is git-tracked;
  2. every receipt the analysis gates read is git-tracked: the static GATE_INPUTS list, and
  3. (added 2026-10-04, BootLoops audit F8) every file under results/ that a gate ACTUALLY opens.
     The static list missed the five tokalign_extract/*/extract_receipt.json files that
     check_gemmahook_claims.roster_expected() reads, so a clean clone crashed with
     FileNotFoundError while this checker reported 0 problems. Check 3 runs each gate in
     DYNAMIC_GATES in a subprocess with a sys.addaudithook('open') recorder and requires every
     opened results/ path to be tracked. A hand list cannot drift from the code this way.
     Gates that rewrite tex or recompile (check_rule_coverage, check_plants) are not run here.

    python3 code/analysis/check_receipts_tracked.py      # exit 1 if anything is untracked
"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Receipts that the checkers/emitters load directly. If a gate reads it, it must ship.
GATE_INPUTS = [
    "results/cross_model_sae/splithalf20k_gemma2-9b__mistral7b.json",
    "results/cross_model_sae/splithalf20k_llama3.2-1b__mistral7b.json",
    "results/cross_model_sae/splithalf20k_qwen3-1.7b__gemma2-2b.json",
    "results/cross_model_sae/covctl_k200_c4_qwen3-1.7b__gemma2-2b.json",
    "results/cross_model_sae/covctl_k500_c4_qwen3-1.7b__gemma2-2b.json",
    "results/cross_model_sae/covctl_k200_c4_gemma2-9b__mistral7b.json",
    "results/cross_model_sae/covctl_k500_c4_gemma2-9b__mistral7b.json",
    "results/cross_model_sae/covctl_k200_c4_llama3.2-1b__mistral7b.json",
    "results/cross_model_sae/covctl_k500_c4_llama3.2-1b__mistral7b.json",
    "results/cross_model_sae/INTERP_OVERLAP_NULL.json",
    "results/probe_attribution_controls_n200/STATS.json",
    # A3-EXEC leakage-clean receipts: read by check_tokalign_claims.py and
    # emit_tokalign_table.py, and cited as tokaclean20k_*.json in the paper.
    "results/cross_model_sae/tokaclean20k_qwen3-1.7b__gemma2-2b.json",
    "results/cross_model_sae/tokaclean20k_gemma2-9b__mistral7b.json",
    "results/cross_model_sae/tokaclean20k_llama3.2-1b__mistral7b.json",
    # Nine-cell held-out evaluation of frozen A-selected pairs (JIDs 18176229 / 18177194_1 /
    # 18207953_4), cited as heldout_*.json in app:threats (iv).
    "results/cross_model_sae/heldout_qwen3-1.7b__gemma2-2b.json",
    "results/cross_model_sae/heldout_qwen3-8b__gemma2-2b.json",
    "results/cross_model_sae/heldout_qwen3-1.7b__gemma2-9b.json",
    "results/cross_model_sae/heldout_qwen3-8b__gemma2-9b.json",
    "results/cross_model_sae/heldout_gemma2-2b__mistral7b.json",
    "results/cross_model_sae/heldout_qwen3-8b__mistral7b.json",
    "results/cross_model_sae/heldout_gemma2-9b__mistral7b.json",
    "results/cross_model_sae/heldout_qwen3-1.7b__mistral7b.json",
    "results/cross_model_sae/heldout_llama3.2-1b__mistral7b.json",
    # Lan-filter follow-up (notes/PREREG_lan_filters_20260924.md): read by lanfilt_summary.py
    # and check_lanfilt_claims.py; the unfiltered Table 19 receipts are read for the shares.
    "results/cross_model_sae/lanfiltT_gemma2-9b__mistral7b.json",
    "results/cross_model_sae/lanfiltC_gemma2-9b__mistral7b.json",
    "results/cross_model_sae/lanfiltT_qwen3-1.7b__mistral7b.json",
    "results/cross_model_sae/lanfiltC_qwen3-1.7b__mistral7b.json",
    "results/cross_model_sae/lanfiltT_pythia70m__mistral7b.json",
    "results/cross_model_sae/lanfiltC_pythia70m__mistral7b.json",
    "results/cross_model_sae/lanfiltT_gpt2__mistral7b.json",
    "results/cross_model_sae/lanfiltC_gpt2__mistral7b.json",
    "results/cross_model_sae/lanT_gemma2-9b__mistral7b.json",
    "results/cross_model_sae/lanC_gemma2-9b__mistral7b.json",
    "results/cross_model_sae/lanT_qwen3-1.7b__mistral7b.json",
    "results/cross_model_sae/lanC_qwen3-1.7b__mistral7b.json",
    "results/cross_model_sae/lanT_pythia70m__mistral7b.json",
    "results/cross_model_sae/lanC_pythia70m__mistral7b.json",
    "results/cross_model_sae/lanT_gpt2__mistral7b.json",
    "results/cross_model_sae/lanC_gpt2__mistral7b.json",
    "results/cross_model_sae/LANFILT_DECISIONS.json",
    # Gemma Scope read-point sensitivity (notes/PREREG_gemma_hook_sensitivity_20260924.md): read by
    # gemmahook_summary.py and check_gemmahook_claims.py (local MPS run; cluster 18441889; cluster 18445512).
    "results/cross_model_sae/gemma_hook_sensitivity/GEMMA_HOOK_SENSITIVITY.json",
    "results/cross_model_sae/gemma_hook_sensitivity/cluster/GEMMA_HOOK_SENSITIVITY.json",
    "results/cross_model_sae/gemma_hook_sensitivity/cluster_q8g2/GEMMA_HOOK_SENSITIVITY.json",
    "results/cross_model_sae/gemma_hook_sensitivity/GEMMA_HOOK_COMBINED.json",
    "results/cross_model_sae/gemma_hook_sensitivity/interp_explore_gemma2-9b__mistral7b.json",
    # BOS/tokenizer probe at the pinned revisions + C4 prefix-length distribution (cluster login
    # node, <ENV_ANALYSIS>, tokenizers only, HF offline; 2026-09-25): cited in app:threats (v) and
    # the App. D C4-slice note, read by check_c4_table.py.
    "results/cross_model_sae/bos_probe_ct.json",
    # Token-arm extraction receipts (JID 15816850 family), read by
    # check_gemmahook_claims.roster_expected() for the roster L0 column (App. tab_sae_roster).
    # Gitignored by results/* until 2026-10-04 (audit F8); now tracked with git add -f.
    "results/cross_model_sae/tokalign_extract/gemma2-2b_l12/extract_receipt.json",
    "results/cross_model_sae/tokalign_extract/gemma2-9b_l20/extract_receipt.json",
    "results/cross_model_sae/tokalign_extract/llama3.2-1b_b8/extract_receipt.json",
    "results/cross_model_sae/tokalign_extract/mistral7b_b16/extract_receipt.json",
]
# Gates run under an open() audit hook (check 3). Each must be read-only on the repository.
DYNAMIC_GATES = [
    ["code/analysis/check_c4_table.py"],
    ["code/analysis/check_gemmahook_claims.py"],
    ["code/analysis/check_ladder_table.py"],
    ["code/analysis/check_lanfilt_claims.py"],
    ["code/analysis/check_lexical_claims.py"],
    ["code/analysis/check_lextok_claims.py"],
    ["code/analysis/check_tokalign_claims.py"],
    ["code/analysis/check_body_pages.py"],
    ["code/analysis/emit_covctl_table.py", "--check"],
    # revision R1 (2026-10-07): App. ninecell table + the 85-pin main-number gate read the cluster
    # nine-cell receipts, the local five-cell receipts and the F1 bootstrap diagnosis
    ["code/analysis/emit_ninecell_table.py", "--check"],
    ["code/analysis/check_main_numbers.py"],
    ["code/analysis/lextok_summary.py", "--check"],
    ["code/analysis/lanpair_summary.py", "--check"],
    ["code/analysis/procrustes_summary.py", "--check"],
    ["code/analysis/tokalign_clean_summary.py"],
]
_HOOK = r"""
import os, runpy, sys
_root, _log, _script = sys.argv[1], sys.argv[2], sys.argv[3]
_seen = set()
def _h(ev, args):
    if ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.path.realpath(os.fsdecode(args[0]))
        if p.startswith(os.path.join(_root, "results") + os.sep) and p not in _seen:
            _seen.add(p)
            with open(_log, "a") as f:
                f.write(p + "\n")
sys.addaudithook(_h)
sys.argv = sys.argv[3:]
sys.path.insert(0, os.path.dirname(os.path.abspath(_script)))
runpy.run_path(_script, run_name="__main__")
"""


def dynamic_reads():
    """{relpath under results/: [gates that opened it]} plus gates that crashed."""
    import tempfile
    reads, crashed = {}, []
    root = os.path.realpath(ROOT)
    for cmd in DYNAMIC_GATES:
        script = os.path.join(ROOT, cmd[0])
        if not os.path.exists(script):
            crashed.append("%s (absent)" % cmd[0])
            continue
        with tempfile.NamedTemporaryFile("r", suffix=".log") as log:
            r = subprocess.run([sys.executable, "-c", _HOOK, root, log.name, script] + cmd[1:],
                               cwd=ROOT, capture_output=True, text=True)
            opened = [l.strip() for l in open(log.name) if l.strip()]
        if r.returncode != 0 and ("Error" in r.stderr or "Traceback" in r.stderr):
            crashed.append("%s (rc=%d: %s)" % (" ".join(cmd), r.returncode,
                                                r.stderr.strip().splitlines()[-1][:120]))
        for p in opened:
            reads.setdefault(os.path.relpath(p, root), []).append(os.path.basename(cmd[0]))
    return reads, crashed
# v0 artifacts the paper explicitly says do NOT exist; citing them is not a defect.
KNOWN_ABSENT = {"metadata.json"}


def tracked_set():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return set(out.stdout.split())


def cited_paths():
    tex = ""
    for f in sorted(glob.glob(os.path.join(ROOT, "paper/sections/*.tex"))):
        tex += open(f).read()
    cited = set()
    for m in re.finditer(r"\\texttt\{([^}]*?\.json)\}", tex):
        cited.add(m.group(1).replace("\\_", "_").replace("\\", ""))
    for m in re.finditer(r"results/[A-Za-z0-9_/\\.\*-]+\.json", tex):
        cited.add(m.group(0).replace("\\_", "_").replace("\\", ""))
    return sorted(cited)


def main():
    tracked = tracked_set()
    bad, notes = [], []

    for c in cited_paths():
        if os.path.basename(c) in KNOWN_ABSENT:
            continue
        pattern = c if c.startswith("results/") else os.path.join("results", "**", c)
        matches = glob.glob(os.path.join(ROOT, pattern), recursive=True)
        rel = [os.path.relpath(m, ROOT) for m in matches]
        if not rel:
            bad.append("CITED BUT ABSENT     %s" % c)
            continue
        untracked = [r for r in rel if r not in tracked]
        if untracked:
            for u in untracked:
                bad.append("CITED BUT UNTRACKED  %s  (matched by %s)" % (u, c))
        if "*" in c:
            notes.append("  glob %-42s -> %d file(s), all tracked: %s"
                         % (c, len(rel), not untracked))

    for g in GATE_INPUTS:
        if not os.path.exists(os.path.join(ROOT, g)):
            bad.append("GATE INPUT ABSENT    %s" % g)
        elif g not in tracked:
            bad.append("GATE INPUT UNTRACKED %s" % g)

    if "--static-only" not in sys.argv:
        reads, crashed = dynamic_reads()
        for rel, who in sorted(reads.items()):
            if rel not in tracked:
                bad.append("GATE READ UNTRACKED  %s  (opened by %s)" % (rel, ", ".join(sorted(set(who)))))
        for c in crashed:
            bad.append("GATE CRASHED         %s" % c)
        notes.append("  dynamic: %d gate(s) run, %d results/ file(s) opened"
                     % (len(DYNAMIC_GATES), len(reads)))

    for n in notes:
        print(n)
    for b in bad:
        print(b)
    print("%d cited path(s) + %d gate input(s) checked, %d problem(s)"
          % (len(cited_paths()), len(GATE_INPUTS), len(bad)))
    print("FINDINGS=%d" % len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

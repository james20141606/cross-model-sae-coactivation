#!/usr/bin/env python3
"""Prove every gate can FAIL, using baseline-aware findings-count deltas -- never exit codes.

Why not exit codes: steering reported 10/11 and 12/12 plants "caught" and both numbers
turned out to measure nothing, because one of its gates returns 1 unconditionally on an
unverifiable figure. Once any standing failure exists, "exit==1 after planting" is true
whether or not the plant was detected, and the harness silently becomes a rubber stamp
that always says PASS.

So each gate here prints a machine-readable `FINDINGS=<n>` line, and a plant is judged by
    findings(planted) - findings(baseline) >= 1
which is correct even when the baseline is already non-zero, and which cannot be satisfied
by a pre-existing failure.

Every plant is applied to a real file, then reverted, and the harness asserts the baseline
is restored -- so a plant that corrupts the tree cannot be mistaken for a clean run.

    python3 code/analysis/check_plants.py        # exit 1 if any gate cannot fail
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def findings(cmd):
    """Run a gate and return its FINDINGS count. Absence of the line is fatal, not zero."""
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    m = re.findall(r"^FINDINGS=(\d+)$", r.stdout, re.M)
    if not m:
        raise RuntimeError("gate %r printed no FINDINGS= line (stdout %d bytes, rc=%d)"
                           % (cmd, len(r.stdout), r.returncode))
    return int(m[-1])


def plant_file(path, old, new):
    """Context-manager-ish: returns a restore() closure. Fails loudly if `old` is absent."""
    full = os.path.join(ROOT, path)
    orig = open(full).read()
    if old not in orig:
        raise RuntimeError("plant target %r not present in %s -- plant is stale"
                           % (old[:60], path))
    open(full, "w").write(orig.replace(old, new, 1))
    return lambda: open(full, "w").write(orig)


PY = sys.executable if os.path.basename(sys.executable).startswith("python") else "/usr/bin/python3"

CASES = [
    # (name, gate cmd, file, find, replace, needs_recompile)
    ("c4_table (appendix exact-value row)",
     [PY, "code/analysis/check_c4_table.py"],
     "paper/sections/appendix.tex",
     "Qwen3-1.7B $\\leftrightarrow$ Gemma2-2B & $0.732$ & $0.936$",
     "Gemma2-2B $\\leftrightarrow$ Gemma2-9B & within & $41.1$", False),
    ("receipts_tracked (cited receipt)",
     [PY, "code/analysis/check_receipts_tracked.py"],
     "paper/sections/appendix.tex",
     "\\texttt{rand\\_c4\\_*.json}",
     "\\texttt{rand\\_c4\\_*.json} and \\texttt{no\\_such\\_receipt\\_xyz.json}", False),
    ("covctl emitter (emitted table + compiled PDF)",
     [PY, "code/analysis/emit_covctl_table.py", "--check"],
     "paper/sections/tab_covctl.tex",
     "$0.930$", "$0.111$", True),
    # The defect this gate was written for: the adequacy gate's Delta-R^2 (+0.27, a margin
    # over a badly-predicting shuffled design) read as variance captured. Recompiled, so
    # the PDF scope is exercised through the real pipeline and not just the .tex.
    ("lextok claims (dR2 read as variance captured)",
     [PY, "code/analysis/check_lextok_claims.py"],
     "paper/sections/results.tex",
     "$R^2$ tops out at $0.14$ captures almost none",
     "$R^2$ tops out at $0.27$ captures almost none", True),
    # ... and the same gate's stale-maximum rule, on a number no other gate reads.
    # 2026-09-25 de-numbering moved the max Delta-R^2 out of results.tex; the number now
    # lives only in the app:lexres paragraph, so the plant targets it there.
    ("lextok claims (stale max Delta-R^2)",
     [PY, "code/analysis/check_lextok_claims.py"],
     "paper/sections/appendix.tex",
     "median $\\Delta R^2$ rising to $+0.31$",
     "median $\\Delta R^2$ rising to $+0.27$", True),
]


def recompile():
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                   cwd=os.path.join(ROOT, "paper"), capture_output=True, timeout=300)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                   cwd=os.path.join(ROOT, "paper"), capture_output=True, timeout=300)


def main():
    results = []
    for name, cmd, path, old, new, needs_compile in CASES:
        try:
            base = findings(cmd)
            restore = plant_file(path, old, new)
            try:
                if needs_compile:
                    recompile()
                planted = findings(cmd)
            finally:
                restore()
                if needs_compile:
                    recompile()
            after = findings(cmd)
            caught = planted - base >= 1
            restored = after == base
            results.append((name, base, planted, after, caught, restored))
            print("%-44s baseline=%d planted=%d restored=%d  delta=%+d  %s%s"
                  % (name, base, planted, after, planted - base,
                     "CAUGHT" if caught else "MISSED",
                     "" if restored else "  !! TREE NOT RESTORED"))
        except Exception as e:
            results.append((name, -1, -1, -1, False, False))
            print("%-44s HARNESS ERROR: %s" % (name, e))

    print()
    ok = all(c and r for _, _, _, _, c, r in results)
    for name, b, p, a, c, r in results:
        print("PASS %-44s = %s" % (name, c and r))
    print("\nJudged by findings-count delta, never by exit code: a gate that already fails"
          "\nfor an unrelated reason cannot make a plant look caught.")
    print("VERDICT:", "every gate can fail" if ok else "A GATE CANNOT FAIL -- it is decorative")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

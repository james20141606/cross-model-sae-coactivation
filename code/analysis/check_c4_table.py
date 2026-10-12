#!/usr/bin/env python3
"""Verify every held-out C4 row in tab:c4_verdicts against its committed receipt.

The table was carrying the paper's headline -- nine natural-text-robust cross-family
cells -- with no receipts in the repo at all; they lived only on /scratch. They are
committed now, and this asserts the table against them so the two cannot drift apart
silently, which is how the same defect appeared in three other papers this week.

    python3 code/analysis/check_c4_table.py          # exit 1 on any mismatch
"""
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results/cross_model_sae")
# The exact-value table moved to the appendix when the same evidence became a
# main-text quantitative figure. Keep checking every row, scoped to its label.
TEX = os.path.join(ROOT, "paper/sections/appendix.tex")
APP = os.path.join(ROOT, "paper/sections/appendix.tex")

# display row -> held-out receipt basename.
ROWS = {
    "Qwen3-1.7B $\\leftrightarrow$ Gemma2-2B": "heldout_qwen3-1.7b__gemma2-2b",
    "Qwen3-8B $\\leftrightarrow$ Gemma2-2B": "heldout_qwen3-8b__gemma2-2b",
    "Qwen3-1.7B $\\leftrightarrow$ Gemma2-9B": "heldout_qwen3-1.7b__gemma2-9b",
    "Qwen3-8B $\\leftrightarrow$ Gemma2-9B": "heldout_qwen3-8b__gemma2-9b",
    "Qwen3-8B $\\leftrightarrow$ Mistral-7B": "heldout_qwen3-8b__mistral7b",
    "Gemma2-9B $\\leftrightarrow$ Mistral-7B": "heldout_gemma2-9b__mistral7b",
    "Gemma2-2B $\\leftrightarrow$ Mistral-7B": "heldout_gemma2-2b__mistral7b",
    "Qwen3-1.7B $\\leftrightarrow$ Mistral-7B": "heldout_qwen3-1.7b__mistral7b",
    "Llama-3.2-1B $\\leftrightarrow$ Mistral-7B": "heldout_llama3.2-1b__mistral7b",
}
# tab:gpt2_c4 (appendix): the four GPT-2 natural-text cells. These are the receipts behind
# the "uninformative, not a second negative" claim, so they are gated too.
GPT2_ROWS = {
    "GPT-2 $\\leftrightarrow$ Mistral-7B": "c4_gpt2__mistral7b",
    "Gemma2-2B $\\leftrightarrow$ GPT-2": "c4_gemma2-2b__gpt2",
    "Gemma2-9B $\\leftrightarrow$ GPT-2": "c4_gemma2-9b__gpt2",
    "Qwen3-1.7B $\\leftrightarrow$ GPT-2": "c4_qwen3-1.7b__gpt2",
}


def z(tag):
    return json.load(open(os.path.join(RES, tag + ".json")))["mutual_z"]


def main():
    tex = open(TEX).read()
    bad = []
    tables = [b.split("\\end{table}", 1)[0] for b in tex.split("\\begin{table}")
              if "\\label{tab:c4_verdicts}" in b.split("\\end{table}", 1)[0]]
    if len(tables) != 1:
        bad.append("tab:c4_verdicts not found exactly once in appendix.tex")
    tex = tables[0] if len(tables) == 1 else ""
    for row, tag in ROWS.items():
        m = re.search(re.escape(row) +
                      r"\s*&\s*\$(\d+\.\d{3})\$\s*&\s*\$(\d+\.\d{3})\$", tex)
        if not m:
            bad.append("%s: row not found in tab:c4_verdicts (appendix.tex)" % row)
            continue
        d = json.load(open(os.path.join(RES, tag + ".json")))
        pe, pr = float(m.group(1)), float(m.group(2))
        ae = float(d["held_out"]["excess"])
        ar = float(d["ratio_heldout_over_insample"])
        if pe != round(ae, 3) or pr != round(ar, 3):
            bad.append("%s: prints %.3f/%.3f, receipt says %.6f/%.6f"
                       % (row, pe, pr, ae, ar))
        if d["n_perm"] != 2000 or d["held_out"]["p"] != 1 / 2001:
            bad.append("%s: held-out n_perm/p is not 2000 and 1/2001" % row)

    # tab:gpt2_c4 lives in the appendix
    # tab:gpt2_c4 rows are "<cell> & <obs> & <null> & <mutual z> & <greedy z> \\".
    # Match the row at the start of a line and take the 3rd &-field, rather than the first
    # $...$ that happens to follow the cell name -- the same names appear in other tables.
    app = open(APP).read()
    # scope to the tab:gpt2_c4 table: tab:sae_matching has 5-column rows with the same
    # cell names (templated corpus), and would otherwise be matched instead.
    blocks = [b for b in app.split("\\begin{table}") if "\\label{tab:gpt2_c4}" in b]
    if len(blocks) != 1:
        bad.append("tab:gpt2_c4 table block not found exactly once in appendix.tex")
        blocks = [""]
    app_rows = {}
    for line in blocks[0].splitlines():
        if "&" not in line or "\\\\" not in line:
            continue
        head = line.split("&")[0].strip()
        app_rows.setdefault(head, []).append(line)
    for row, tag in GPT2_ROWS.items():
        cands = app_rows.get(row, [])
        cands = [c for c in cands if len(c.split("&")) == 5]
        if len(cands) != 1:
            bad.append("%s: expected 1 five-column row in appendix.tex, found %d"
                       % (row, len(cands)))
            continue
        f = cands[0].split("&")[3].strip()
        m = re.fullmatch(r"\$(-?\d+\.\d\d)\$", f)
        if not m:
            bad.append("%s: mutual-z field %r not parseable" % (row, f))
            continue
        printed, actual = float(m.group(1)), z(tag)
        if abs(printed - round(actual, 2)) > 1e-9:
            bad.append("%s: prints %.2f, receipt says %.4f" % (row, printed, actual))
        # the claim is that these are UNDERPOWERED, not that they are clean negatives
        nullmu = json.load(open(os.path.join(RES, tag + ".json")))["null_mutual_nn"]["mean"]
        if nullmu > 10:
            bad.append("%s: null mean %.1f is no longer the degenerate ~2 the text claims"
                       % (row, nullmu))

    # the killed flagship must stay killed
    p70 = json.load(open(os.path.join(RES, "c4_pythia70m__mistral7b.json")))
    if p70["mutual_z"] >= 0 or p70["mutual_p"] < 0.5:
        bad.append("pythia70m<->mistral7b is no longer null: z=%.2f p=%.3f"
                   % (p70["mutual_z"], p70["mutual_p"]))

    # Finding 1 (results.tex) prints the held-out/re-selected ratio range in words-and-numbers.
    # The de-numbering pass once turned "74--94%" into "at least three-quarters" although the
    # Llama cell's ratio is 0.7393; nothing caught it. Pin the printed range to the receipts:
    # the low end must be round(min*100) and must not exceed the minimum ratio itself.
    ratios = [float(json.load(open(os.path.join(RES, t + ".json")))["ratio_heldout_over_insample"])
              for t in ROWS.values()]
    res_tex = open(os.path.join(ROOT, "paper/sections/results.tex")).read()
    m = re.search(r"held-out excess is\s+\$(\d+)\$--\$(\d+)\\%\$\s+of the eval-half", res_tex)
    if not m:
        bad.append("results.tex Finding 1: held-out/re-selected ratio range '$lo$--$hi\\%%$ of the "
                   "eval-half' not found (min receipt ratio %.4f)" % min(ratios))
    else:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo != round(100 * min(ratios)) or hi != round(100 * max(ratios)):
            bad.append("results.tex Finding 1 prints %d--%d%%, receipts give %.4f--%.4f"
                       % (lo, hi, min(ratios), max(ratios)))
        if lo / 100 > min(ratios) + 0.005:
            bad.append("results.tex Finding 1 lower bound %d%% exceeds min ratio %.4f" % (lo, min(ratios)))

    # BOS probe receipt (bos_probe_ct.py): threats (v) and the App. D C4-slice note.
    bp = json.load(open(os.path.join(RES, "bos_probe_ct.json")))
    tk = bp["tokenizers"]
    want = {"google/gemma-2-2b": True, "google/gemma-2-9b": True, "mistralai/Mistral-7B-v0.1": True,
            "meta-llama/Llama-3.2-1B": True, "Qwen/Qwen3-1.7B-Base": False, "Qwen/Qwen3-8B-Base": False}
    for hf, pre in want.items():
        if hf not in tk or tk[hf]["prepends_bos"] is not pre:
            bad.append("bos_probe_ct.json: %s prepends_bos != %s (threats (v))" % (hf, pre))
    rev_tex = open(os.path.join(ROOT, "paper/sections/tab_hyperparams.tex")).read()
    for hf, d in tk.items():
        if d["revision"] not in rev_tex:
            bad.append("bos_probe_ct.json: %s revision %s not the one pinned in tab:model-revisions"
                       % (hf, d["revision"][:8]))
    c = bp["c4_files"]["c4_slice.jsonl"]["chars"]
    n300 = bp["c4_files"]["c4_slice.jsonl"]["n_at_300"]
    tmin = min(d["T_d_c4_slice"]["min"] for d in tk.values())
    tmax = max(d["T_d_c4_slice"]["max"] for d in tk.values())
    mlo = min(d["T_d_c4_slice"]["mean"] for d in tk.values())
    mhi = max(d["T_d_c4_slice"]["mean"] for d in tk.values())
    note = ("$%d$--$%d$ characters (mean $%d$; $%s$ of $2{,}000$ at the cap), or $%d$--$%d$ tokens,\n"
            "with means of $%d$--$%d$" % (c["min"], c["max"], round(c["mean"]),
                                          "{:,}".format(n300).replace(",", "{,}"),
                                          tmin, tmax, round(mlo), round(mhi)))
    if note not in app:
        bad.append("App. D C4-slice note does not print the receipt's lengths: expected %r" % note)
    if "about $1/%d$ here" % round(statistics.median([d["T_d_c4_slice"]["median"] for d in tk.values()])) not in app:
        bad.append("App. D C4-slice note: BOS weight '1/T_d about 1/<median T_d>' does not match receipt")

    for b in bad:
        print("MISMATCH  " + b)
    n = len(ROWS) + len(GPT2_ROWS) + 1
    print("%d rows checked, %d mismatches" % (n, len(bad)))
    print("FINDINGS=%d" % len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

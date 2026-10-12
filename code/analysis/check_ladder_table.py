#!/usr/bin/env python3
"""Gate: every cell of the main-text control ladder (tab:ladder, results.tex) and the
obs / null-mean columns of tab:c4_counts (appendix) against the committed receipts.

WHY THIS FILE EXISTS. The 2026-09-24 final review mutation-tested the existing gates:
check_c4_table covers tab:c4_verdicts but not tab:c4_counts (811 -> 812 passed), and no
gate read tab:ladder row by row, although it is the one main-text table. Every expected
string below is derived from a receipt; nothing is typed here.

    /usr/bin/python3 code/analysis/check_ladder_table.py              # exit 1 on mismatch
    /usr/bin/python3 code/analysis/check_ladder_table.py --selftest   # plants must fire
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/cross_model_sae"
NINECELL = ROOT / "results/ninecell_20260926/cluster_18577359"
RESULTS = ROOT / "paper/sections/results.tex"
APPENDIX = ROOT / "paper/sections/appendix.tex"
REPS = ["qwen3-1.7b__gemma2-2b", "gemma2-9b__mistral7b", "llama3.2-1b__mistral7b"]


def j(name):
    return json.loads((RES / name).read_text())


def ladder_expected():
    """row label (as typeset, $ stripped) -> expected 'strong/median/weak' string."""
    rows = {}
    c4 = [j(f"c4_{s}.json") for s in REPS]
    rows["C4 screen"] = "/".join(
        f"{d['observed']['n_mutual_nn']} ({d['null_mutual_nn']['mean']:.1f})" for d in c4)
    rd = [j(f"rand_c4_{s}.json") for s in REPS]
    rows["Random directions"] = "/".join(
        f"{d['observed']['n_mutual_nn']} ({d['null_mutual_nn']['mean']:.1f})" for d in rd)
    ho = [j(f"heldout_{s}.json") for s in REPS]
    rows["Held-out frozen pairs"] = "/".join(f"{d['held_out']['excess']:.3f}" for d in ho)
    rows["ratio to re-selected pairs"] = "/".join(
        f"{d['ratio_heldout_over_insample']:.3f}" for d in ho)
    # Revision R1 (2026-10-07, item C3): the topic rows print the registered post-submission run,
    # whose size-matched calibration averages 20 random partitions (point.top{k}.R). The
    # submitted rows used the one-partition covctl_k{k}_c4_* receipts, still shown in tab:covctl
    # (pinned by emit_covctl_table --check).
    for k in (200, 500):
        nc = [json.loads((NINECELL / f"A_{s}.json").read_text())["point"][f"top{k}"] for s in REPS]
        assert all(d["calibration"]["n_rep"] == 20 for d in nc)
        rows[f"Topic strata, k{{=}}{k}"] = "/".join(f"{d['R']:.3f}" for d in nc)
    rows["Lexical residualization, mutual NN"] = "/".join(
        f"{j(f'lextok20k_{s}.json')['arms']['4096']['R_lex']:.3f}" for s in REPS)
    rows["pairing rule (reg.)"] = "/".join(
        f"{j(f'lanpair_lextok20k_{s}.json')['arms']['4096']['R_lex']:.3f}" for s in REPS)
    rows["Procrustes pairing"] = "/".join(
        f"{j(f'proclex_lextok20k_{s}.json')['arms']['4096']['R_lex']:.3f}" for s in REPS)
    rows["Token-aligned, within-document order"] = "/".join(
        f"{j(f'tokaclean20k_{s}.json')['s1']['obs_mean_pair_r']:.3f}" for s in REPS)
    rows["Current-word type, first design"] = "/".join(
        f"{j(f'tokalign20k_{s}.json')['s2']['R_tok']:.3f}" for s in REPS)
    rows["Current-word type, leakage-clean"] = "/".join(
        f"{j(f'tokaclean20k_{s}.json')['s2_clean']['R_tok_clean']:.3f}" for s in REPS)
    return rows


C4_COUNTS = {  # tab:c4_counts row label -> receipt
    "Gemma2-2B $\\leftrightarrow$ Gemma2-9B": "c4_gemma2-2b__gemma2-9b.json",
    "Qwen3-8B $\\leftrightarrow$ Qwen3-1.7B": "c4_qwen3-8b__qwen3-1.7b.json",
    "Qwen3-1.7B $\\leftrightarrow$ Gemma2-2B": "c4_qwen3-1.7b__gemma2-2b.json",
    "Qwen3-8B $\\leftrightarrow$ Gemma2-9B": "c4_qwen3-8b__gemma2-9b.json",
    "Qwen3-8B $\\leftrightarrow$ Gemma2-2B": "c4_qwen3-8b__gemma2-2b.json",
    "Qwen3-1.7B $\\leftrightarrow$ Gemma2-9B": "c4_qwen3-1.7b__gemma2-9b.json",
    "Gemma2-9B $\\leftrightarrow$ Mistral-7B": "c4_gemma2-9b__mistral7b.json",
    "Qwen3-8B $\\leftrightarrow$ Mistral-7B": "c4_qwen3-8b__mistral7b.json",
    "Gemma2-2B $\\leftrightarrow$ Mistral-7B": "c4_gemma2-2b__mistral7b.json",
    "Qwen3-1.7B $\\leftrightarrow$ Mistral-7B": "c4_qwen3-1.7b__mistral7b.json",
    "Llama-3.2-1B $\\leftrightarrow$ Mistral-7B": "c4_llama3.2-1b__mistral7b.json",
    "Gemma2-2B(w$65$k) $\\leftrightarrow$ Mistral-7B": "c4_w65k_gemma2-2b__mistral7b.json",
    "Pythia-70M $\\leftrightarrow$ Mistral-7B": "c4_pythia70m__mistral7b.json",
}


def block(text, label):
    i = text.index("\\label{" + label + "}")
    return text[i:text.index("\\end{tabular}", i)]


def scan(results_tex, appendix_tex):
    bad = []
    lad = block(results_tex, "tab:ladder").replace("$", "")
    lines = [ln for ln in lad.splitlines() if "&" in ln]
    for key, want in ladder_expected().items():
        row = next((ln for ln in lines if key.replace("$", "") in ln), None)
        if row is None:
            bad.append((f"tab:ladder row '{key}'", "row not found"))
        elif not re.search(r"(?<![\d.])" + re.escape(want) + r"(?![\d.])", row):
            bad.append((f"tab:ladder row '{key}'", want))
    cc = block(appendix_tex, "tab:c4_counts")
    for label, rec in C4_COUNTS.items():
        d = j(rec)
        row = next((ln for ln in cc.splitlines() if ln.strip().startswith(label + " ")
                    or ln.strip().startswith(label + "&")), None)
        if row is None:
            bad.append((f"tab:c4_counts row {rec}", "row not found"))
            continue
        cells = [c.strip().replace("$", "") for c in row.split("&")]
        want_obs = str(d["observed"]["n_mutual_nn"])
        want_null = f"{d['null_mutual_nn']['mean']:.1f}{{\\pm}}{d['null_mutual_nn']['sd']:.1f}"
        if cells[2] != want_obs:
            bad.append((f"tab:c4_counts obs {rec}", want_obs))
        if cells[3] != want_null:
            bad.append((f"tab:c4_counts null {rec}", want_null))
    return bad


def main():
    bad = scan(RESULTS.read_text(), APPENDIX.read_text())
    for what, v in bad:
        print(f"MISMATCH {what}: receipt value {v}")
    print("check_ladder_table:", "FAIL" if bad else "PASS",
          f"({len(ladder_expected())} tab:ladder rows, {len(C4_COUNTS)} tab:c4_counts rows)")
    return 1 if bad else 0


def selftest():
    r, a = RESULTS.read_text(), APPENDIX.read_text()
    checks = [("clean pass", not scan(r, a))]
    checks.append(("ladder value plant (0.927->0.928)",
                   bool(scan(r.replace("$0.927$/$0.982$", "$0.928$/$0.982$", 1), a))))
    # R1: the superseded one-partition k=500 row must not pass (1.033 -> 1.060)
    checks.append(("ladder stale one-partition plant (1.033->1.060)",
                   bool(scan(r.replace("$0.986$/$1.033$/$0.974$", "$0.986$/$1.060$/$0.974$", 1), a))))
    checks.append(("ladder swap plant",
                   bool(scan(r.replace("$0.933$/$0.942$", "$0.942$/$0.933$", 1), a))))
    i = a.index("\\label{tab:c4_counts}")
    a2 = a[:i] + a[i:].replace("& $811$ &", "& $812$ &", 1)
    checks.append(("c4_counts obs plant (811->812)", bool(scan(r, a2))))
    a3 = a[:i] + a[i:].replace("$371.8{\\pm}10.3$", "$371.9{\\pm}10.3$", 1)
    checks.append(("c4_counts null plant (371.8->371.9)", bool(scan(r, a3))))
    for name, got in checks:
        print(f"  {name:38s} {'OK' if got else 'FAILED'}")
    ok = all(g for _, g in checks)
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())

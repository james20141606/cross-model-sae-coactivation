#!/usr/bin/env python3
"""Recompute the main-text control ladder (Table 1, tab:ladder) from the bundled receipts and
compare each row's value triple with the compiled paper.

Standard library only. Reads the receipts through the same derivation the paper's gate uses
(code/analysis/check_ladder_table.py: ladder_expected), so this is a reproduction, not an
independent re-derivation. Prints the table, and, if paper/main.pdf and `pdftotext` (poppler) are
available, checks that each recomputed strong/median/weak string occurs somewhere in the PDF text
(not located at its row). The p column is printed but NOT compared. Exit 1 on any mismatch.

    python3 tools/reproduce_main_table.py
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/analysis"))
import check_ladder_table as L  # noqa: E402

RES = ROOT / "results/cross_model_sae"
ORDER = [
    ("C4 screen", "C4 screen, 2,000 docs", "M (null mean)"),
    ("Random directions", "Random directions (size, L0 matched)", "M (null mean)"),
    ("Held-out frozen pairs", "Held-out frozen pairs, 1,000+1,000", "Delta_E"),
    ("ratio to re-selected pairs", "  ratio to re-selected pairs", "r"),
    ("Topic strata, k{=}200", "Topic strata, k=200", "R"),
    ("Topic strata, k{=}500", "Topic strata, k=500", "R"),
    ("Lexical residualization, mutual NN", "Lexical residualization, mutual NN (reg.)", "R_lex"),
    ("pairing rule (reg.)", "  Lan et al. pairing rule (reg.)", "R_lex"),
    ("Procrustes pairing", "  Procrustes pairing, disjoint half (reg.)", "R_lex"),
    ("Token-aligned, within-document order", "Token-aligned, within-document order (reg.)", "S1_clean"),
    ("Current-word type, first design", "Current-word type, first design (leaky)", "R_tok"),
    ("Current-word type, leakage-clean", "Current-word type, leakage-clean (reg.)", "R_tok_clean"),
]


def p_column():
    """Permutation p per rung, from n_perm and the receipts' own p (all cells share one value)."""
    def one(files, get):
        vals = {round(get(json.loads((RES / f).read_text())), 12) for f in files}
        return vals
    cells = L.REPS
    out = {}
    out["C4 screen"] = one([f"c4_{c}.json" for c in cells], lambda d: d["mutual_p"])
    out["Random directions"] = one([f"rand_c4_{c}.json" for c in cells], lambda d: d["mutual_p"])
    out["Held-out frozen pairs"] = one([f"heldout_{c}.json" for c in cells], lambda d: d["held_out"]["p"])
    for k in (200, 500):
        out[f"Topic strata, k{{=}}{k}"] = one([f"covctl_k{k}_c4_{c}.json" for c in cells],
                                            lambda d: d["bow_matched"]["mutual_p"])
    for key, pre in (("Lexical residualization, mutual NN", "lextok20k"), ("pairing rule (reg.)", "lanpair_lextok20k"),
                     ("Procrustes pairing", "proclex_lextok20k")):
        out[key] = one([f"{pre}_{c}.json" for c in cells],
                       lambda d: d["arms"]["4096"]["pair_survival_real"]["p"])
    out["Token-aligned, within-document order"] = one([f"tokaclean20k_{c}.json" for c in cells],
                                                     lambda d: d["s1"]["null_n1"]["p"])
    out["Current-word type, leakage-clean"] = one([f"tokaclean20k_{c}.json" for c in cells],
                                                 lambda d: d["s2_clean"]["null_n3"]["p"])
    return out


def fmt_p(vals):
    if not vals:
        return "---"
    if len(vals) > 1:
        return "/".join(str(v) for v in sorted(vals))
    v = next(iter(vals))
    if v >= 0.999:
        return "1.0"
    return f"1/{round(1 / v)}"


def pdf_text():
    pdf = ROOT / "paper/main.pdf"
    exe = shutil.which("pdftotext")
    if not (pdf.exists() and exe):
        return None
    t = subprocess.run([exe, str(pdf), "-"], capture_output=True, text=True).stdout
    return re.sub(r"\s+", "", t.replace("−", "-"))


def main():
    rows = L.ladder_expected()
    ps = p_column()
    txt = pdf_text()
    bad = 0
    print(f"{'Rung':46s} {'statistic':14s} {'strong/median/weak':44s} {'p':8s} PDF")
    for key, label, stat in ORDER:
        val = rows[key]
        p = fmt_p(ps.get(key, set()))
        if txt is None:
            mark = "(no pdftotext/PDF: not compared)"
        else:
            ok = re.sub(r"\s+", "", val) in txt
            mark = "match" if ok else "MISMATCH"
            bad += not ok
        print(f"{label:46s} {stat:14s} {val:44s} {p:8s} {mark}")
    print(f"\nstrong/median/weak = Qwen3-1.7B<->Gemma-2-2B / Gemma-2-9B<->Mistral-7B / "
          f"Llama-3.2-1B<->Mistral-7B; lexical rows at k=4096.")
    if txt is not None:
        print(f"PDF comparison: {len(ORDER) - bad}/{len(ORDER)} rows found verbatim in paper/main.pdf")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

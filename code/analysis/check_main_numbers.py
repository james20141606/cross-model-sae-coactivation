#!/usr/bin/env python3
"""Gate: context-anchored pins for the main-text sentence/table-row fragments that the
2026-10-05 project mutation sweep (<internal-notes>
report.md) found NO existing checker protects (11/60 caught -- the worst preset in the
project), plus the items listed in notes/CAMERA_READY_CHECKER_TODO.md sections A-D.

WHY A NEW FILE. check_ladder_table.py already pins tab:ladder's strong/median/weak value
triples and tab:c4_counts' obs/null columns -- that is why those cells are *not* repeated
here. Everything below is a gap check_ladder_table and the other existing checkers do not
read: the tab:ladder p-column, the prose sentences around the tables (abstract, intro
caption, method, experiments, limitations, results), and a few structural invariants
(Finding N ordering, the eq:excess/eq:rho/eq:count formulas, the three-cell REPS naming
order in the tab:ladder caption).

DESIGN. Every expected number is computed from a results/ receipt (or, where the number is
a citation to an external paper's own table, or a chosen protocol constant such as "95%
bootstrap CI" or "0.30 collapse bar", flagged in a comment as a registered constant rather
than a receipt-derived one). Each pin is a *fragment*: enough literal surrounding text, with
all whitespace collapsed to single spaces, that a one-digit change, a sign flip, or a swap
with another number in the same sentence/row breaks the fragment match. This file does not
import any other checker (independence -- see <internal-notes>'s audit finding F7 on
reproduce_main_table.py importing check_ladder_table).

    /usr/bin/python3 code/analysis/check_main_numbers.py              # exit 1 on mismatch
    /usr/bin/python3 code/analysis/check_main_numbers.py --selftest   # mutation + null-edit controls
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEC = ROOT / "paper/sections"
RES = ROOT / "results/cross_model_sae"

REPS = ["qwen3-1.7b__gemma2-2b", "gemma2-9b__mistral7b", "llama3.2-1b__mistral7b"]
HELDOUT_CELLS = [  # the 9 non-pre-registered held-out cells (results.tex "all nine cells")
    "gemma2-2b__mistral7b", "gemma2-9b__mistral7b", "llama3.2-1b__mistral7b",
    "qwen3-1.7b__gemma2-2b", "qwen3-1.7b__gemma2-9b", "qwen3-1.7b__mistral7b",
    "qwen3-8b__gemma2-2b", "qwen3-8b__gemma2-9b", "qwen3-8b__mistral7b",
]

SECTION_FILES = {
    "abstract": SEC / "abstract.tex",
    "intro": SEC / "intro.tex",
    "method": SEC / "method.tex",
    "experiments": SEC / "experiments.tex",
    "results": SEC / "results.tex",
    "limitations": SEC / "limitations.tex",
    "related": SEC / "related.tex",
    "discussion": SEC / "discussion.tex",
    # revision R1 (2026-10-07): App. ninecell + its generated table, the fine-topic paragraph
    "appendix": SEC / "appendix.tex",
    "tab_ninecell": SEC / "tab_ninecell.tex",
}
NINECELL = ROOT / "results/ninecell_20260926/cluster_18577359"   # registered cluster run (after submission)
NINECELL_LOCAL = ROOT / "results/ninecell_20260926/local_mps_cache"
F1DIAG = ROOT / "results/f1_bootstrap_diag_20261004/F1_BOOTSTRAP_DIAG.json"
NINE_ORDER = ["qwen3-1.7b__gemma2-2b", "qwen3-1.7b__gemma2-9b", "qwen3-1.7b__mistral7b",
              "qwen3-8b__gemma2-2b", "qwen3-8b__gemma2-9b", "qwen3-8b__mistral7b",
              "gemma2-2b__mistral7b", "gemma2-9b__mistral7b", "llama3.2-1b__mistral7b"]
NAME_TEX = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-8b": "Qwen3-8B", "gemma2-2b": "Gemma-2-2B",
            "gemma2-9b": "Gemma-2-9B", "mistral7b": "Mistral-7B", "llama3.2-1b": "Llama-3.2-1B"}
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def j(name):
    return json.loads((RES / name).read_text())


def norm(s):
    """Collapse all whitespace runs to single spaces so pins survive re-wrapping,
    but not any digit/word change inside the fragment."""
    return re.sub(r"\s+", " ", s).strip()


def comma(n):
    """LaTeX thousands separator: 1500 -> '1{,}500'. Scoped to just the number so it
    can be substituted into an f-string without any risk of a nearby prose comma
    (', i.e.' etc.) being caught by a careless whole-string .replace(',', '{,}')."""
    return f"{n:,}".replace(",", "{,}")


def load_sections():
    return {k: norm(p.read_text()) for k, p in SECTION_FILES.items()}


# ---------------------------------------------------------------------------
# Receipt-derived scalars shared by several pins
# ---------------------------------------------------------------------------

def derive():
    d = {}
    c4 = {s: j(f"c4_{s}.json") for s in REPS}
    rand = {s: j(f"rand_c4_{s}.json") for s in REPS}
    heldout = {s: j(f"heldout_{s}.json") for s in HELDOUT_CELLS}
    splithalf = {s: j(f"splithalf20k_{s}.json") for s in REPS}
    lextok = {s: j(f"lextok20k_{s}.json") for s in REPS}
    lanpair = {s: j(f"lanpair_lextok20k_{s}.json") for s in REPS}
    proclex = {s: j(f"proclex_lextok20k_{s}.json") for s in REPS}
    tokaclean = {s: j(f"tokaclean20k_{s}.json") for s in REPS}
    covctl200 = {s: j(f"covctl_k200_c4_{s}.json") for s in REPS}
    covctl500 = {s: j(f"covctl_k500_c4_{s}.json") for s in REPS}
    templated_q = j("qwen3-1.7b__gemma2-2b.json")  # non-C4 templated-corpus receipt
    c4_g2_m7 = j("c4_gemma2-2b__mistral7b.json")
    c4_w65k = j("c4_w65k_gemma2-2b__mistral7b.json")
    steer50 = j("steer50_gemma2-9b__mistral7b.json")
    steer_b2 = j("steer_beta2_gemma2-9b__mistral7b.json")
    steer_b8 = j("steer_gemma2-9b__mistral7b.json")

    n_c4_cells = len(glob.glob(str(RES / "c4_*.json")))  # 17-cell roster
    d["n_c4_cells"] = n_c4_cells
    assert n_c4_cells == 17, f"c4_*.json count drifted to {n_c4_cells}; roster-size pins stale"

    n_perm_c4 = templated_q["n_perm"]           # 200 -> floor 1/201
    n_perm_heldout = heldout[HELDOUT_CELLS[0]]["n_perm"]   # 2000 -> floor 1/2001
    assert all(heldout[c]["n_perm"] == n_perm_heldout for c in HELDOUT_CELLS)
    assert all(covctl200[s]["n_perm"] == n_perm_heldout for s in REPS)
    assert all(covctl500[s]["n_perm"] == n_perm_heldout for s in REPS)
    assert all(tokaclean[s]["n_perm"] == n_perm_heldout for s in REPS)
    d["p_floor_c4"] = f"1/{n_perm_c4 + 1}"        # "1/201"
    d["p_floor_heldout"] = f"1/{n_perm_heldout + 1}"  # "1/2001"
    d["n_perm_heldout"] = n_perm_heldout

    d["n_heldout_cells"] = len(HELDOUT_CELLS)  # "nine"
    assert d["n_heldout_cells"] == 9

    # abstract/results: dictionary-width swap z=22.1 -> 14.1
    d["z_g2m7_16k"] = f"{c4_g2_m7['mutual_z']:.1f}"
    d["z_g2m7_65k"] = f"{c4_w65k['mutual_z']:.1f}"

    # Finding 1: weak on templates z=3.59 p=1/201, strong on C4 z=42.7
    d["z_templated_qg"] = f"{templated_q['mutual_z']:.2f}"
    d["z_c4_qg"] = f"{c4['qwen3-1.7b__gemma2-2b']['mutual_z']:.1f}"

    # abstract: 92--94% lexical-residualization retention (mutual-NN row, k=4096 governing)
    r_lex = [lextok[s]["arms"]["4096"]["R_lex"] for s in REPS]
    d["lex_pct_lo"] = round(100 * min(r_lex))
    d["lex_pct_hi"] = round(100 * max(r_lex))

    # abstract/results "0.88": min verdict-bearing R across every panel-(b) row
    # (topic k=200, topic k=500, lexical mutual-NN, Lan-rule, Procrustes, token-clean;
    # the leaky first-design R_tok is explicitly excluded by the caption itself)
    # R1: Table 1's topic rows print the 20-partition run (ninecell A receipts), so the floor
    # is taken over those; the one-partition covctl values are listed in tab:covctl only.
    nine_a = {c: json.loads((NINECELL / f"A_{c}.json").read_text()) for c in NINE_ORDER}
    nine_b = {c: json.loads((NINECELL / f"B_{c}.json").read_text()) for c in NINE_ORDER}
    r_all = []
    for s in REPS:
        r_all.append(nine_a[s]["point"]["top200"]["R"])
        r_all.append(nine_a[s]["point"]["top500"]["R"])
        r_all.append(lextok[s]["arms"]["4096"]["R_lex"])
        r_all.append(lanpair[s]["arms"]["4096"]["R_lex"])
        r_all.append(proclex[s]["arms"]["4096"]["R_lex"])
        r_all.append(tokaclean[s]["s2_clean"]["R_tok_clean"])
    import math
    d["r_floor"] = math.floor(100 * min(r_all)) / 100.0  # "at least 0.88"

    # results.tex:42 "74--94%" held-out excess vs re-selected excess, all 9 cells
    ratios = [heldout[c]["ratio_heldout_over_insample"] for c in HELDOUT_CELLS]
    d["heldout_pct_lo"] = round(100 * min(ratios))
    d["heldout_pct_hi"] = round(100 * max(ratios))

    # method/limitations: split-half selection optimism on the 3 ladder cells
    sh_ratios = [splithalf[s]["summary"]["mean_ratio_heldout_over_insample"] for s in REPS]
    d["sh_ratio_lo"] = f"{min(sh_ratios):.2f}"
    d["sh_ratio_hi"] = f"{max(sh_ratios):.2f}"
    opt_pct = [100 * (1 - r) for r in sh_ratios]
    d["opt_pct_lo"] = round(min(opt_pct))
    d["opt_pct_hi"] = round(max(opt_pct))
    # method.tex "On five cells, seeded random halves change the median held-out excess by at most
    # 3% (separately extracted features)": derived from the five local B receipts' rho
    # (CAMERA_READY_CHECKER_TODO M2), not from splithalf20k (a different quantity, same value).
    loc_rho = [json.loads(p.read_text())["random_splits"]["rho"]
               for p in sorted(NINECELL_LOCAL.glob("B_*.json"))]
    assert len(loc_rho) == 5, len(loc_rho)
    d["n_local_cells"] = WORD[len(loc_rho)]
    d["local_rho_pct"] = math.ceil(100 * max(abs(1 - r) for r in loc_rho))
    d["n_docs_splithalf"] = splithalf[REPS[0]]["n_docs"]
    assert all(splithalf[s]["n_docs"] == d["n_docs_splithalf"] for s in REPS)

    # method.tex: C4 slice size = select + eval halves of the held-out follow-up
    ho0 = heldout[HELDOUT_CELLS[3]]  # qwen3-1.7b__gemma2-2b, representative
    d["c4_slice_docs"] = ho0["n_sel_docs"] + ho0["n_eval_docs"]
    d["heldout_half_docs"] = ho0["n_sel_docs"]
    assert ho0["n_sel_docs"] == ho0["n_eval_docs"]

    # method.tex Prop 1(a): E[M] = Ka*Kb/(Ka+Kb-1) approx 750 at Ka=Kb=1500
    Ka, Kb = c4["qwen3-1.7b__gemma2-2b"]["Ka"], c4["qwen3-1.7b__gemma2-2b"]["Kb"]
    assert Ka == Kb == 1500, f"K drifted to Ka={Ka} Kb={Kb}; eq:rho/Prop-1 pins stale"
    d["K"] = Ka
    d["EM_approx"] = round(Ka * Kb / (Ka + Kb - 1))

    # results.tex Finding 3: steering overlap counts
    d["steer50_overlap"] = f"{steer50['matched_mean_overlap']:.2f}"
    d["steer50_null"] = f"{steer50['null_mean']:.2f}"
    d["steer50_ntop"] = steer50["n_top"]
    d["steer50_npairs"] = steer50["n_pairs"]
    d["steer50_beta"] = int(steer50["beta"])
    d["steer_b2_npairs"] = steer_b2["n_pairs"]
    d["steer_b2_beta"] = int(steer_b2["beta"])
    d["steer_b8_npairs"] = steer_b8["n_pairs"]
    d["steer_b8_beta"] = int(steer_b8["beta"])
    assert d["steer_b2_npairs"] == d["steer_b8_npairs"] == 20

    # roster-derived: 8 models / 6 families, Pythia-70M .. Mistral-7B match null
    roster = (SEC / "tab_sae_roster.tex").read_text()
    rows = re.findall(r"^([A-Za-z].*?) & ", roster, re.M)
    rows = [r for r in rows if r not in ("Model",)]
    d["n_roster_rows"] = len(rows)
    fam = lambda name: re.match(r"[A-Za-z]+(-[0-9]+)?", name).group(0).split("-")[0]
    d["n_roster_families"] = len({fam(r) for r in rows})
    m = re.search(r"Pythia-(\d+M)", roster)
    d["pythia_size"] = m.group(1)
    m = re.search(r"Mistral-(\d+B)-v0\.1", roster)
    d["mistral_size"] = m.group(1)
    gemma_sizes = sorted({int(s) for s in re.findall(r"Gemma-2-(\d+)B\b", roster)})
    d["gemma9b_size"] = f"{gemma_sizes[-1]}B"  # the roster's largest Gemma-2, pairs with pythia_size "70M"
    assert d["gemma9b_size"] == "9B", d["gemma9b_size"]

    # results.tex:8 "leave at least $0.96$ of the excess" (length-decile nulls, six cells):
    # five cells from the separately-extracted-feature re-run (NINECELL_SUMMARY.json) plus the
    # three ladder cells' *uncalibrated* length-decile arm inside the covctl receipts
    # (length_matched.mutual_excess / baseline_random.mutual_excess) -- appendix.tex:697-703
    # ("Length deciles leave R_len=0.972--1.003 ... leaves 1.002, 0.993 and 0.964").
    ninecell = json.loads((ROOT / "results/ninecell_20260926/local_mps_cache/NINECELL_SUMMARY.json").read_text())
    r_len_5cell = [v["arms"]["len"]["R"] for v in ninecell["decision"]["QA_cells"].values()]
    r_len_3ladder = []
    for s in REPS:
        c = covctl200[s]
        r_len_3ladder.append(c["length_matched"]["mutual_excess"] / c["baseline_random"]["mutual_excess"])
    d["len_decile_min"] = math.floor(100 * min(r_len_5cell + r_len_3ladder)) / 100.0
    d["ladder_len_arm"] = [f"{x:.3f}" for x in r_len_3ladder]
    assert d["len_decile_min"] == 0.96, d["len_decile_min"]

    # results.tex:110 "median per-feature $R^2$ tops out at $0.14$": max real median-R2 over
    # model_a/model_b and the three ladder cells at the governing k=4096 lexical-nuisance width.
    r2_vals = []
    for s in REPS:
        adeq4096 = lextok[s]["arms"]["4096"]["adequacy"]
        r2_vals.append(adeq4096["model_a"]["median_r2_real"])
        r2_vals.append(adeq4096["model_b"]["median_r2_real"])
    d["lextok_r2_max"] = math.floor(100 * max(r2_vals)) / 100.0
    assert d["lextok_r2_max"] == 0.14, d["lextok_r2_max"]

    # experiments.tex:11 / results.tex:189,193,196: residual scale-up counts. Receipt-derived from
    # paper/sections/appendix.tex's tab:models and tab:residual_scaleup (prose + table rows) --
    # appendix.tex is NOT in the mutation sweep's scope=main candidate file list (same status as
    # tab_sae_roster.tex above: an unmutated source of truth this checker cross-reads rather than
    # a receipt JSON, because no standalone JSON receipt for these counts was found under results/).
    appendix = (SEC / "appendix.tex").read_text()
    m = re.search(r"the \$(\d+)\$ antonym pairs are hand-authored", appendix)
    d["n_antonym_pairs"] = int(m.group(1))
    assert d["n_antonym_pairs"] == 20, d["n_antonym_pairs"]
    m = re.search(r"\$(\d+)\$-pair scale-up set spans nine", appendix)
    d["n_residual_pairs"] = int(m.group(1))
    assert d["n_residual_pairs"] == 206, d["n_residual_pairs"]
    resid_tab = appendix.split(r"\label{tab:residual_scaleup}", 1)[1].split(r"\end{table}", 1)[0]
    model_rows = re.findall(r"[A-Za-z][\w.\-]*\s*&\s*(?:EN|ML)\s*&\s*\$[+-]", resid_tab)
    d["n_residual_models"] = len(model_rows)
    assert d["n_residual_models"] == 17, d["n_residual_models"]

    # ---------------- revision R1 (2026-10-07) ----------------
    f3 = lambda x: f"{x:.3f}"
    ceil3 = lambda x: math.ceil(1000 * x - 1e-9) / 1000.0
    # Table 1 topic rows and caption (20-partition calibration)
    for k in (200, 500):
        d[f"topic{k}"] = "/".join(f"${f3(nine_a[s]['point'][f'top{k}']['R'])}$" for s in REPS)
        assert all(nine_a[s]["point"][f"top{k}"]["p"] == 1 / (n_perm_heldout + 1) for s in REPS)
    nrep = {nine_a[s]["point"][f"top{k}"]["calibration"]["n_rep"] for s in REPS for k in (200, 500)}
    assert len(nrep) == 1
    d["n_cal_rep"] = nrep.pop()
    topic_all = [nine_a[s]["point"][f"top{k}"]["R"] for s in REPS for k in (200, 500)]
    d["topic_lo"], d["topic_hi"] = f3(min(topic_all)), f3(max(topic_all))
    med = nine_a[REPS[1]]["point"]["top500"]
    one = [med["excess"] / r["excess"] for r in med["calibration"]["reps"]]
    d["k500_med_one_lo"], d["k500_med_one_hi"] = f3(min(one)), f3(max(one))
    # T12: k=500 block structure (identical strata in the three ladder cells)
    bs = {tuple(sorted(nine_a[s]["block_sizes"]["top500"])) for s in REPS}
    assert len(bs) == 1
    bs = list(bs.pop())
    assert len(bs) == 500 and sum(bs) == d["c4_slice_docs"]
    d["k500_big"] = comma(max(bs))
    d["k500_single"] = sum(1 for b in bs if b == 1)
    d["k500_small"] = WORD[len(bs) - 1 - d["k500_single"]]
    assert all(1 < b < 10 for b in sorted(bs)[d["k500_single"]:-1])
    # abstract: R_tok^clean range at Table 1's 3 dp; lexical ratio range behind "92--94%"
    rtc = [tokaclean[s]["s2_clean"]["R_tok_clean"] for s in REPS]
    d["rtc_lo"], d["rtc_hi"] = f3(min(rtc)), f3(max(rtc))
    d["lex_r_lo"], d["lex_r_hi"] = f3(min(r_lex)), f3(max(r_lex))
    d["n_templated_prompts"] = templated_q["N_prompts"]
    d["proclex_triple"] = "/".join(f"${f3(proclex[s]['arms']['4096']['R_lex'])}$" for s in REPS)
    d["s1clean_triple"] = "/".join(f"${f3(tokaclean[s]['s1']['obs_mean_pair_r'])}$" for s in REPS)
    assert all(abs(proclex[s]["arms"]["4096"]["pair_survival_real"]["p"] - 1 / (n_perm_heldout + 1)) < 1e-12 for s in REPS)
    assert all(abs(tokaclean[s]["s1"]["null_n1"]["p"] - 1 / (n_perm_heldout + 1)) < 1e-12 for s in REPS)
    # App. ninecell, cluster run: A1 (length/coarse-topic nulls), B1 (random splits), G0
    rl = [nine_a[c]["point"]["len"]["R"] for c in NINE_ORDER]
    rt = [nine_a[c]["point"]["top10"]["R"] for c in NINE_ORDER]
    assert all(nine_a[c]["point"][a]["p"] == 1 / (n_perm_heldout + 1) for c in NINE_ORDER for a in ("len", "top10"))
    d["nc_rlen"] = (f3(min(rl)), f3(max(rl)))
    d["nc_rtop"] = (f3(min(rt)), f3(max(rt)))
    q8 = [c for c in NINE_ORDER if c.startswith("qwen3-8b")]
    d["q8_rlen"] = (f3(min(nine_a[c]["point"]["len"]["R"] for c in q8)), f3(max(nine_a[c]["point"]["len"]["R"] for c in q8)))
    d["q8_rtop"] = (f3(min(nine_a[c]["point"]["top10"]["R"] for c in q8)), f3(max(nine_a[c]["point"]["top10"]["R"] for c in q8)))
    d["nc_n_q8"] = WORD[len(q8)]
    rs = {c: nine_b[c]["random_splits"] for c in NINE_ORDER}
    d["nc_n_tests"] = sum(len(rs[c]["splits"]) for c in NINE_ORDER)
    assert all(abs(rs[c]["max_p"] - 1 / (n_perm_heldout + 1)) < 1e-15 for c in NINE_ORDER)
    rho = [rs[c]["rho"] for c in NINE_ORDER]
    d["nc_rho"] = (f3(min(rho)), f3(max(rho)))
    d["nc_rho_pct"] = math.ceil(100 * max(abs(1 - r) for r in rho))
    cons = {c: nine_b[c]["consecutive"]["held_out"]["excess"] for c in NINE_ORDER}
    worst = max(NINE_ORDER, key=lambda c: 1 - rs[c]["min_excess"] / cons[c])
    a_, b_ = worst.split("__")
    d["nc_worst_pct"] = f"{100 * (1 - rs[worst]['min_excess'] / cons[worst]):.1f}"
    d["nc_worst_name"] = f"{NAME_TEX[a_]}$\\leftrightarrow${NAME_TEX[b_]}"
    d["nc_worst_vals"] = (f3(rs[worst]["min_excess"]), f3(cons[worst]))
    d["nc_min_split"] = f3(min(rs[c]["min_excess"] for c in NINE_ORDER))
    d["nc_g0_pairs"] = WORD[max(abs(nine_a[c]["G0a"]["dM"]) for c in NINE_ORDER)]
    d["nc_g0_exc"] = f"{ceil3(max(abs(nine_b[c]['G0b']['d']) for c in NINE_ORDER)):.3f}"
    loc = sorted(p.name[2:-5] for p in NINECELL_LOCAL.glob("A_*.json"))
    la = {c: json.loads((NINECELL_LOCAL / f"A_{c}.json").read_text()) for c in loc}
    lb = {c: json.loads((NINECELL_LOCAL / f"B_{c}.json").read_text()) for c in loc}
    d["nc_n_local"] = WORD[len(loc)]
    d["nc_agree"] = tuple(f"{ceil3(x):.3f}" for x in (
        max(abs(nine_a[c]["point"]["len"]["R"] - la[c]["point"]["len"]["R"]) for c in loc),
        max(abs(nine_a[c]["point"]["top10"]["R"] - la[c]["point"]["top10"]["R"]) for c in loc),
        max(abs(cons[c] - lb[c]["consecutive"]["held_out"]["excess"]) for c in loc)))
    # tab_ninecell rows, derived here independently of emit_ninecell_table.py
    lab = {"qwen3-1.7b": "Q1.7", "qwen3-8b": "Q8", "gemma2-2b": "G2", "gemma2-9b": "G9",
           "mistral7b": "M", "llama3.2-1b": "L1"}
    rows = []
    for c in NINE_ORDER:
        a_, b_ = c.split("__")
        p_ = nine_a[c]["point"]
        rows.append(f"{lab[a_]}--{lab[b_]} & ${nine_a[c]['obs_M']}$ & ${f3(p_['len']['R'])}$ & "
                    f"${f3(p_['top10']['R'])}$ & ${f3(cons[c])}$ & ${f3(rs[c]['min_excess'])}$--"
                    f"${f3(rs[c]['max_excess'])}$ & ${f3(rs[c]['rho'])}$ & ${f3(rs[c]['median_r'])}$ \\\\")
    d["nc_rows"] = rows
    # registered F1 diagnosis (C0)
    f1 = json.loads(F1DIAG.read_text())
    sm, cl = f1["summary"], f1["cells"]["cluster"]
    assert sm["QC_reading"].startswith("C0")
    assert sm["random_splits"]["n_tests"] == d["nc_n_tests"] and sm["random_splits"]["all_p_1_over_2001"]
    d4 = list(sm["D4_base_excess_ratio"].values())
    d["f1_d4"] = (f"{min(d4):.2f}", f"{max(d4):.2f}")
    d3 = [r["D3_mc_over_boot_sd"] for c in cl.values() for r in c["A"]["rows"] if r["name"] in ("R_len", "R_top10")]
    assert len(d3) == 18
    d["f1_d3"] = (f"{min(d3):.2f}", f"{max(d3):.2f}")
    sh = [abs(c["B"]["rows"][0]["shift_in_sd"]) for c in cl.values()]
    assert all(c["B"]["rows"][0]["shift_in_sd"] < 0 for c in cl.values())
    d["f1_shift"] = (f"{min(sh):.1f}", f"{max(sh):.1f}")
    d["f1_excl"] = (len(sm["intervals_excluding_point"]), sm["n_intervals"])
    pl = f1["D5_planted"]
    pct = lambda x: f"{100 * x:.1f}".rstrip("0").rstrip(".")
    d["f1_cov"] = (pct(pl["gaussian"]["coverage"]), pct(pl["sparse_q0.02"]["coverage"]), pct(pl["sparse_q0.10"]["coverage"]))
    nds = {v["n_data"] for v in pl.values()}
    assert len(nds) == 1
    d["f1_ndata"] = nds.pop()

    return d


# ---------------------------------------------------------------------------
# Pins: (id, section, fragment-building function taking derived dict d)
# ---------------------------------------------------------------------------

def build_pins(d):
    pins = []

    def pin(id_, section, frag):
        pins.append((id_, section, norm(frag)))

    # ---- abstract.tex ----
    pin("A-nine-2001-17", "abstract",
        r"non-pre-registered follow-up, all nine cross-family cells (Gemma-2, Mistral, Qwen3, Llama-3) "
        f"hold on disjoint C4 text ($p={d['p_floor_heldout']}$ each, Bonferroni over the {d['n_c4_cells']}-cell roster).")
    pin("A-0.88", "abstract",
        f"every control leaves at least ${d['r_floor']:.2f}$ of the excess its reference arm leaves.")
    pin("A-92-94", "abstract",
        f"leaves ${d['lex_pct_lo']}$--${d['lex_pct_hi']}\\%$ of the excess: the shared-vocabulary explanation")
    pin("A-22.1-14.1", "abstract",
        f"moves its cell from $z={d['z_g2m7_16k']}$ to ${d['z_g2m7_65k']}$), or shared computation")
    pin("A-70M-7B", "abstract",  # R1 (C5): the cell is named, as in the body
        f"and a Pythia-{d['pythia_size']}$\\leftrightarrow$Mistral-{d['mistral_size']} match null on natural text).")
    pin("A-1500-chance", "abstract",  # R1 seed-2 sweep gap
        f"two sets of ${comma(d['K'])}$ features reciprocally match about half of each other under idealized chance")
    pin("I-0.88-intro", "intro",  # R1 seed-2 sweep gap
        f"each leave a retention ratio of at least ${d['r_floor']:.2f}$ against its reference arm.")
    pin("M-heldout-halves-1000", "method",  # R1 seed-2 sweep gap
        f"Split the corpus into disjoint halves $D_S,D_E$ (${comma(d['heldout_half_docs'])}$ C4 documents each).")
    pin("R-qwen-sizes-scope", "results",  # R1 seed-2 sweep gap (registered model names, literal)
        r"and Qwen3-1.7B/8B (Qwen-Scope SAEs, \citealp{qwenscope2026};")
    pin("R-ladder-procrustes-p", "results",  # R1 seed-2 sweep gap (row incl. p column)
        f"Procrustes pairing, disjoint half (reg.) & $R_{{\\text{{lex}}}}$ & {d['proclex_triple']} & ${d['p_floor_heldout']}$ \\\\")
    pin("R-ladder-s1clean-p", "results",  # R1 seed-2 sweep gap
        f"(mean $r$) & {d['s1clean_triple']} & ${d['p_floor_heldout']}$ \\\\")
    pin("E-206-prompt-corpus", "experiments",  # R1 seed-0 re-sweep gap
        f"on a shared ${d['n_templated_prompts']}$-prompt English concept corpus and on C4")
    pin("I-1500-chance", "intro",  # R1 seed-0 re-sweep gap
        f"two sets of ${comma(d['K'])}$ features reciprocally match about half of each other")
    pin("R-fig2a-1000-docs", "results",  # R1 seed-0 re-sweep gap (Fig 2 caption, line moved by the R1 edit)
        f"(a)~Pairs selected on ${comma(d['heldout_half_docs'])}$ C4 documents and frozen")
    # 2026-10-10 Figure 1 redraw: the control ladder (old Fig 2b) became Figure 1 (intro.tex), so its
    # R=1 definition pin moved from results to intro; "(shaded: lexical pairing rules)" is gone with
    # the shading. Same literal check otherwise.
    pin("I-fig1-R1-means", "intro",  # was R-fig2b-R1-means (R1 seed-0 re-sweep gap, definition, literal)
        r"$R=1$ means the control removes nothing beyond its reference arm, and $R=0$ that the covariate,"
        r" as the control captures it, explains all of the sharing.")
    pin("I-fig1-row-scope", "intro",  # 2026-10-10: which rows cover nine cells vs three
        r"Top two rows: all nine cross-family cells (\Cref{tab:ninecell}); other rows: three"
        r" representative cells (\Cref{tab:ladder}).")
    pin("M-half-of-1500", "method",  # R1 seed-1 re-sweep gap
        f"By (a), about half of the ${comma(d['K'])}$ screened features per model are reciprocally matched under")
    pin("M-prop2a-eq", "method",  # R1 seed-1 re-sweep gap (Prop 2(a) equation, literal)
        r"$\mathbb{E}[T_E(\widehat{\mathcal{P}})\mid D_S]=|\widehat{\mathcal{P}}|^{-1}"
        r" \sum_{(i,j)\in\widehat{\mathcal{P}}}\mathbb{E}[\rho^E_{ij}]$, as for a pair set fixed in advance;")
    pin("A-Rtokclean-range", "abstract",  # R1 (C5): Table 1's 3-dp values
        f"keeps $R_{{\\text{{tok}}}}^{{\\text{{clean}}}}{{=}}{d['rtc_lo']}$--${d['rtc_hi']}$ of its token-level excess")

    # ---- intro.tex (figure caption) ----
    # 2026-10-10: I-1000-1000-nine and I-fig-markers-23 retired -- they pinned sentences of the
    # schematic Figure 1 caption, which was replaced by the data-drawn control ladder (the sentences
    # no longer exist; the 1,000+1,000 held-out design stays pinned in method/results:
    # M-heldout-halves-1000, R-fig2a-1000-docs). The new caption is pinned by I-fig1-* above.

    # ---- experiments.tex ----
    pin("E-2000-17-2001-0.05", "experiments",
        f"disjoint-half follow-up uses ${comma(d['n_perm_heldout'])}"
        f"$ draws and all nine tests clear Bonferroni over the ${d['n_c4_cells']}$ C4 cells "
        f"(${d['p_floor_heldout']}<0.05/{d['n_c4_cells']}$). Other verdicts use registered")
    pin("E-9.1e-ci", "experiments",
        r"Intervals are $95\%$ bootstrap CIs ($10{,}000$ resamples; model/data seeds $0$, bootstrap seed $1$).")
    pin("E-8bbase-model-list", "experiments",  # closes experiments.tex:5 seed-1 gap (registered model-name constant)
        r"Qwen3-1.7B/8B-Base \citep{yang2025qwen3}, Llama-3.2-1B \citep{grattafiori2024llama3}")
    pin("E-206-17-residualscaleup", "experiments",  # closes experiments.tex:11 seed-1 gap
        f"on ${d['n_antonym_pairs']}$ aligned antonym pairs per language in en/zh/ja/ko, expanded to"
        f" ${d['n_residual_pairs']}$ concept pairs across nine concept families and ${d['n_residual_models']}"
        "$ models for the residual scale-up.")

    # ---- method.tex ----
    pin("M-eqrho", "method",
        f"We keep the $K{{=}}{comma(d['K'])}"
        r"$ highest-variance features per model, standardize each over $D$ to $z^A_i(d)$, and define"
        r" \begin{equation}\label{eq:rho} \rho_{ij} \;=\; \tfrac{1}{N}\textstyle\sum_{n=1}^{N} z^A_i(d_n)\, z^B_j(d_n),"
        r" \qquad i\le K_A,\; j\le K_B, \end{equation}")
    pin("M-eqcount-draws", "method",
        r"From draws $M^{(1)},\dots,M^{(n_{\text{perm}})}$ we report")
    pin("M-eqexcess", "method",
        r"\begin{equation}\label{eq:excess} \Delta = M-\overline{M^{\pi}},\qquad z=\Delta/\operatorname{sd}(M^{\pi}),\qquad"
        r" p=\frac{1+\#\{b: M^{(b)}\ge M\}}{n_{\text{perm}}+1}. \end{equation}")
    pin("M-prop1a-premise", "method",  # covers method.tex:35's swap_sent 1->1{,}500, missed by
        r"(a) If, for every $(i,j)$, the $K_A{+}K_B{-}1$ entries in row $i$ and column $j$ of"
        r" $\rho$ are exchangeable and almost surely distinct (e.g.\ i.i.d.\ continuous), then")
    pin("M-prop1a", "method",
        f"\\mathbb{{E}}[M]=K_AK_B/(K_A{{+}}K_B{{-}}1)$, i.e.\\ $\\approx{d['EM_approx']}$ at "
        f"$K_A{{=}}K_B{{=}}{comma(d['K'])}$.")
    pin("M-eqTE", "method",  # closes method.tex:54 seed-1 gap (held-out frozen-pairs statistic, literal equation)
        r"frozen and scored on $D_E$ by $T_E(\mathcal{P})=|\mathcal{P}|^{-1}\sum_{(i,j)\in\mathcal{P}}\rho^{E}_{ij}$,")
    pin("M-heldout-2000perm", "method",  # closes method.tex:55 seed-1 gap
        f"with excess $\\Delta_E$ and $p$ from ${comma(d['n_perm_heldout'])}$ permutations of $D_E$ alone. The ratio")
    pin("M-c4slice", "method",
        f"our C4 slice is the first ${comma(d['c4_slice_docs'])}"
        "$ eligible streamed documents, split into consecutive halves.")
    pin("M-splithalf-20k-0.97-0.99", "method",
        f"a seeded random split of ${comma(d['n_docs_splithalf'])}"
        f"$ documents keeps ${d['sh_ratio_lo']}$--${d['sh_ratio_hi']}$ of in-sample excess held out")
    pin("M-selection-optimism-pct", "method",  # R1: source fixed to the local B receipts' rho
        f"On {d['n_local_cells']} cells, seeded random halves change the median held-out excess by at most "
        f"${d['local_rho_pct']}\\%$ (separately extracted features,")
    pin("M-q8-after-submission", "method",
        r"the Qwen3-8B cells were untested at submission (a registered run after submission covers all nine,")

    # ---- results.tex ----
    pin("R-lendecile-0.96", "results",  # closes results.tex:8 seed-1 gap; R1 adds the nine-cell pointer
        f"length-decile nulls, run on six of the nine cells, leave at least ${d['len_decile_min']:.2f}"
        r"$ of the excess (\Cref{app:lencov,app:ninecell}; all nine after submission).")
    pin("R-fig-meanpm2sd", "results",
        r"(marker) against its own permutation null (grey, mean${\pm}2$\,SD) for trained SAEs and")
    pin("R-1000-heldout-p", "results",
        f"over two different pair sets (\\Cref{{prop:heldout}}); every held-out $p={d['p_floor_heldout']}$.")
    pin("R-2000-doc-c4-screen", "results",
        f"The ${comma(d['c4_slice_docs'])}" + "$-document C4 screen")
    pin("R-bonferroni-17-c4cells", "results",  # closes results.tex:41, missed at seed 0 of the re-run
        f"scoring them on the second half confirms all nine at $p={d['p_floor_heldout']}$, "
        f"which clears Bonferroni over all ${d['n_c4_cells']}$ C4 cells considered;")
    pin("R-74-94", "results",
        f"held-out excess is ${d['heldout_pct_lo']}$--${d['heldout_pct_hi']}\\%$ of the eval-half re-selected excess")
    pin("R-finding1-3.59-201-42.7", "results",
        f"weak on templates ($z={d['z_templated_qg']}$, $p={d['p_floor_c4']}$) and strong on C4 ($z={d['z_c4_qg']}$), the")
    pin("R-70M-9B-natural", "results",  # closes results.tex:31 seed-1 gap
        f"among Gemma-2, Mistral-7B and Pythia (models from ${d['pythia_size'][:-1]}$M to ${d['gemma9b_size'][:-1]}"
        "$B) clears its reciprocal null")
    pin("R-22.1-14.1", "results",
        f"swapping only the Gemma-2-2B dictionary $16$k$\\to$$65$k moves its cell from $z={d['z_g2m7_16k']}$ to ${d['z_g2m7_65k']}$")
    pin("R-reps-caption-order", "results",
        r"strong/median/weak (Qwen3-1.7B$\leftrightarrow$Gemma-\mbox{2-2B}, Gemma-2-9B$\leftrightarrow$Mistral-7B,"
        r" Llama-3.2-1B$\leftrightarrow$Mistral-7B).")
    pin("R-governing-k4096", "results",
        r"Ratios follow \eqref{eq:ratio}; lexical rows use the governing $k{=}4096$.")
    pin("R-ladder-c4screen-row", "results",
        f"C4 screen, ${comma(d['c4_slice_docs'])}"
        f"$ docs & $M$ (null mean) & $811$ ($371.8$)/$819$ ($553.4$)/$551$ ($434.9$) & ${d['p_floor_c4']}$ \\\\")
    pin("R-ladder-randdir-p", "results",
        r"Random directions (size, $L_0$ matched) & $M$ (null mean) & $152$ ($604.7$)/$212$ ($607.0$)/$223$ ($584.5$) & $1.0$ \\")
    pin("R-ladder-heldout-row", "results",
        f"Held-out frozen pairs, $1{{,}}000{{+}}1{{,}}000$ & $\\Delta_E$ & $0.732$/$0.602$/$0.519$ & ${d['p_floor_heldout']}$ \\\\")
    pin("R-ladder-topic200-p", "results",  # R1 (C3): 20-partition values, receipt-derived
        f"Topic strata, $k{{=}}200$ & $R$ & {d['topic200']} & ${d['p_floor_heldout']}$ \\\\")
    pin("R-ladder-topic500-p", "results",
        f"Topic strata, $k{{=}}500$ & $R$ & {d['topic500']} & ${d['p_floor_heldout']}$ \\\\")
    pin("R-ladder-caption-ncal", "results",
        f"the size-matched calibration averages ${d['n_cal_rep']}$ random partitions in the registered nine-cell run"
        r" (\Cref{app:ninecell}); the submitted values used one partition (\Cref{tab:covctl}).")
    pin("R-ladder-lexnn-p", "results",
        f"Lexical residualization, mutual NN (reg.) & $R_{{\\text{{lex}}}}$ & $0.933$/$0.942$/$0.918$ & ${d['p_floor_heldout']}$ \\\\")
    pin("R-ladder-lanrule-p", "results",
        f"pairing rule (reg.) & $R_{{\\text{{lex}}}}$ & $0.900$/$0.905$/$0.910$ & ${d['p_floor_heldout']}$ \\\\")
    pin("R-ladder-tokclean-p", "results",
        f"Current-word type, leakage-clean (reg.) & $R_{{\\text{{tok}}}}^{{\\text{{clean}}}}$ & $0.925$/$0.986$/$0.977$ & ${d['p_floor_heldout']}$ \\\\")
    pin("R-earlier-2000-doc-attempt", "results",
        f"An earlier attempt on ${comma(d['c4_slice_docs'])}" + "$ documents could not adjudicate:")
    pin("R-gateR2-margin", "results",  # closes results.tex:107 seed-1 gap; R1 (C5) adds 92--94% in the body
        f"keep nearly all of their excess: residualization leaves ${d['lex_pct_lo']}$--${d['lex_pct_hi']}\\%$ of it "
        f"($R_{{\\text{{lex}}}}={d['lex_r_lo']}$--${d['lex_r_hi']}$, \\Cref{{tab:ladder}}). "
        r"The gate's $\Delta R^2$ is a margin over a shuffled design,")
    pin("R-medianR2-0.14", "results",  # closes results.tex:110 seed-1 gap (two mutations, one fragment)
        f"a lexical model whose median per-feature $R^2$ tops out at ${d['lextok_r2_max']:.2f}"
        "$ captures almost none of what the two models share.")
    pin("R-lan-tables-5-6", "results",
        r"the $1$--$1$ fraction must fall inside the interval implied by their Tables~5--6, and")
    pin("R-0.30-collapse-bar", "results",
        r"Their rule retains slightly less than ours in every cell, far above the registered $0.30$ collapse bar")
    pin("R-steer-top30", "results",  # closes results.tex:156 seed-1 gap
        f"and compare the identity of the top-${d['steer50_ntop']}$ promoted token strings.")
    pin("R-finding3-steer50", "results",
        f"Matched pairs share more strings than a derangement null (${d['steer50_overlap']}$ vs.\\ ${d['steer50_null']}$ "
        f"of ${d['steer50_ntop']}$ at ${d['steer50_npairs']}$ pairs and $\\beta{{=}}{d['steer50_beta']}$);")
    pin("R-finding3-steer20-beta", "results",
        "the excess holds at $" + str(d['steer_b2_npairs']) + "$ pairs for $\\beta\\in\\{"
        + str(d['steer_b2_beta']) + "," + str(d['steer_b8_beta']) + "\\}$ and replicates on")
    pin("R-finding-sequence", "results", "__FINDING_SEQUENCE__")  # handled specially in scan()
    pin("R-antonym-n20", "results",  # closes results.tex:189 seed-1 gap
        f"appears to separate the classes at $n={d['n_antonym_pairs']}$ antonym pairs, with Pythia as the sole")
    pin("R-residualscaleup-193", "results",  # closes results.tex:193 seed-1 gap
        f"With ${d['n_residual_pairs']}$ concept pairs, ${d['n_residual_models']}$ models and three further English-only")
    pin("R-residualscaleup-196", "results",  # closes results.tex:196 seed-1 gap
        f"and GPT-2's residual is the largest of all ${d['n_residual_models']}$ models. We report all three tests")

    # ---- limitations.tex ----
    pin("L-frozenpair-20000", "limitations",  # closes limitations.tex:8 seed-1 gap
        f"The frozen-pair lexical arm selects and scores on the same ${comma(d['n_docs_splithalf'])}$ documents.")
    pin("L-roster-8-6", "limitations",
        f"eight models across six families" if (d['n_roster_rows'], d['n_roster_families']) == (8, 6) else
        f"__ROSTER_DRIFT_{d['n_roster_rows']}_{d['n_roster_families']}__")
    pin("L-selection-optimism-range", "limitations",
        f"A split-half check finds ${d['opt_pct_lo']}$--${d['opt_pct_hi']}\\%$ selection optimism in pooled excess on three cells,")

    # ---- related.tex ----
    pin("RW-claude3-sonnet", "related",  # closes related.tex:53 seed-1 gap (citation fact, literal)
        r"\citet{templeton2024scaling} report SAE features that fire across languages in Claude 3 Sonnet,")
    pin("RW-claude35-haiku", "related",  # closes related.tex:54 seed-1 gap (citation fact, literal)
        r"and \citet{lindsey2025biology} extend this to circuits in Claude 3.5 Haiku; both are closed-weight.")

    pin("RW-groger-source-strength", "related",  # R1 (C6): arXiv 2602.14486 abstract wording
        r"after permutation calibration the convergence seen by global spectral measures largely disappears while"
        r" local-neighborhood similarity retains significant agreement.")

    # ---- appendix.tex, App. ninecell (revision R1, C1/C2/C4) ----
    pin("N-g0", "appendix",
        f"In every cell it reproduces the published screen count to within {d['nc_g0_pairs']} pairs and the "
        f"consecutive-split held-out excess to within ${d['nc_g0_exc']}$ (registered reproduction gate:")
    pin("N-A1-ranges", "appendix",
        f"Length deciles leave $R_{{\\text{{len}}}}={d['nc_rlen'][0]}$--${d['nc_rlen'][1]}$ and coarse topic "
        f"($k{{=}}10$) leaves $R_{{\\text{{top}}}}={d['nc_rtop'][0]}$--${d['nc_rtop'][1]}$, every arm at "
        f"$p={d['p_floor_heldout']}$ (registered branch A1).")
    pin("N-A1-q8", "appendix",
        f"This includes the {d['nc_n_q8']} Qwen3-8B cells, which had neither null in the submitted version "
        f"($R_{{\\text{{len}}}}={d['q8_rlen'][0]}$--${d['q8_rlen'][1]}$, "
        f"$R_{{\\text{{top}}}}={d['q8_rtop'][0]}$--${d['q8_rtop'][1]}$).")
    pin("N-B1-tests", "appendix",
        f"all ${d['nc_n_tests']}$ held-out tests give $p={d['p_floor_heldout']}$ (branch B1).")
    pin("N-B1-median-single", "appendix",  # C4: "median" kept, single-split spread stated
        f"The median held-out excess over the ten splits is ${d['nc_rho'][0]}$--${d['nc_rho'][1]}$ times that of "
        r"the consecutive split ($\rho$ in \Cref{tab:ninecell}), so the median is within "
        f"${d['nc_rho_pct']}\\%$ of the consecutive split; single splits differ by up to ${d['nc_worst_pct']}\\%$ "
        f"({d['nc_worst_name']}, ${d['nc_worst_vals'][0]}$ vs.\\ ${d['nc_worst_vals'][1]}$), and the smallest "
        f"single-split held-out excess is ${d['nc_min_split']}$.")
    pin("N-local-agree", "appendix",
        f"On the {d['nc_n_local']} cells that both runs cover, the two agree to within ${d['nc_agree'][0]}$ in "
        f"$R_{{\\text{{len}}}}$, ${d['nc_agree'][1]}$ in $R_{{\\text{{top}}}}$ and ${d['nc_agree'][2]}$ in $\\Delta_E$.")
    pin("N-f1-d4", "appendix",
        f"across resamples the mean unrestricted excess is ${d['f1_d4'][0]}$--${d['f1_d4'][1]}$ times the observed one.")
    pin("N-f1-d3", "appendix",
        f"is ${d['f1_d3'][0]}$--${d['f1_d3'][1]}$ times the bootstrap SD, so these intervals largely measure Monte Carlo noise.")
    pin("N-f1-d5", "appendix",
        f"covers the truth in ${d['f1_cov'][0]}\\%$ of Gaussian datasets but in only ${d['f1_cov'][1]}\\%$ and "
        f"${d['f1_cov'][2]}\\%$ of sparse, heavy-tailed ones (${d['f1_ndata']}$ datasets each).")
    pin("N-f1-shift-excl", "appendix",
        f"the $\\Delta_E$ bootstrap medians sit ${d['f1_shift'][0]}$--${d['f1_shift'][1]}$ SD below the point "
        f"estimate, and ${d['f1_excl'][0]}$ of the ${d['f1_excl'][1]}$ intervals exclude their own point estimate.")
    pin("N-f1-C0", "appendix",
        "so the registered reading is C0 (instrument failed) and no interval is reported.")
    pin("N-ladder-len-arm", "appendix",  # CAMERA_READY P4: uncalibrated length arm of the covctl receipts
        f"that leaves ${d['ladder_len_arm'][0]}$, ${d['ladder_len_arm'][1]}$ and ${d['ladder_len_arm'][2]}$ of the excess")
    for i, row in enumerate(d["nc_rows"]):
        pin(f"N-tab-row-{i}", "tab_ninecell", row)
    pin("N-topic-single-draw", "appendix",  # R1 (C3) companion in the fine-topic paragraph
        f"with the calibration averaged over ${d['n_cal_rep']}$ partitions (registered run, after submission; "
        f"\\Cref{{tab:ladder}}) they are ${d['topic_lo']}$--${d['topic_hi']}$, and one partition alone moves the "
        f"$k{{=}}500$ median cell between ${d['k500_med_one_lo']}$ and ${d['k500_med_one_hi']}$.")
    pin("N-T12-blocks", "appendix",
        f"on this slice KMeans returns one cluster of ${d['k500_big']}$ documents, ${d['k500_single']}$ single "
        f"documents and {d['k500_small']} small blocks")

    # ---- discussion.tex ----
    pin("D-finding2-xref", "discussion",  # closes discussion.tex:7 seed-1 gap (cross-reference, literal)
        r"(greedy-max excess saturates even for GPT-2$\leftrightarrow$Mistral-7B, Finding~2), and read each")

    return pins


FINDING_RE = re.compile(r"\\paragraph\{Finding (\d+)")

# Revision R1 (2026-10-07): withdrawn readings must stay withdrawn. Each (id, section, fragment)
# must NOT occur (whitespace-normalised). C1: the document-bootstrap intervals and the C1 branch
# (results/f1_bootstrap_diag_20261004/, reading C0); C6: the strengthened Groger paraphrase;
# C5: the abstract's size-range wording; C3: the one-partition k=500 triple in Table 1.
FORBIDDEN = [
    ("X-C1-branch", "appendix", r"for $\Delta_E$, C1)"),
    ("X-lean-low", "appendix", "these intervals lean low"),
    ("X-deltaE-width", "appendix", r"intervals for $\Delta_E$ of width"),
    ("X-interval-col", "tab_ninecell", "[95"),
    ("X-interval-col-appendix", "appendix", r"$\Delta_E$ [95\%]"),
    ("X-old-table-input", "appendix", r"\input{../results/ninecell_20260926/local_mps_cache/tab_ninecell.tex}"),
    ("X-median-dropped", "appendix", "so on these cells the consecutive split overstates the held-out excess by at most"),
    ("X-groger-only", "related", "only local-neighborhood agreement survives"),
    ("X-abstract-size-range", "abstract", "$70$M-to-$7$B"),
    ("X-ladder-onepartition-k500", "results", "$0.983$/$1.060$/$1.012$"),
]


def scan(sections, d):
    bad = []
    for id_, section, frag in build_pins(d):
        text = sections[section]
        if frag == "__FINDING_SEQUENCE__":
            # Read from the passed-in (possibly mutated, e.g. under --selftest) text, not
            # from disk -- a disk re-read here would make this branch untestable by selftest
            # and, worse, blind if a caller ever passes sections built some other way.
            nums = [int(n) for n in FINDING_RE.findall(text)]
            if nums != [1, 2, 3, 4, 5]:
                bad.append((id_, f"Finding paragraphs out of sequence: {nums}"))
            continue
        if frag.startswith("__ROSTER_DRIFT"):
            bad.append((id_, f"roster row/family count drifted: {frag}"))
            continue
        if frag not in text:
            bad.append((id_, frag))
    for id_, section, frag in FORBIDDEN:
        if norm(frag) in sections[section]:
            bad.append((id_, "withdrawn text is back: " + norm(frag)))
    return bad


def main():
    sections = load_sections()
    d = derive()
    bad = scan(sections, d)
    for id_, frag in bad:
        print(f"MISMATCH {id_}: expected fragment not found: {frag[:160]}")
    n = len(build_pins(d))
    print("check_main_numbers:", "FAIL" if bad else "PASS", f"({n} pins)")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# --selftest: mutate a few of the sweep's actual missed mutations and assert
# the checker now fails; plus a null edit that must still pass.
# ---------------------------------------------------------------------------

MUTATIONS = [  # (section, (old_substring, new_substring)) -- literal reproductions of
               # rows from <internal-notes>
    ("abstract", ("$p=1/2001$ each", "$p=1/2000$ each")),                  # last_digit
    ("abstract", ("leaves $92$--$94\\%$", "leaves $92$--$49\\%$")),         # digit_swap
    ("abstract", ("from $z=22.1$ to $14.1$", "from $z=22.2$ to $14.1$")),   # last_digit
    ("abstract", ("Pythia-70M$\\leftrightarrow$Mistral-7B match", "Pythia-70M$\\leftrightarrow$Mistral-8B match")),  # R1 C5
    ("experiments", ("uses $2{,}000$ draws", "uses $2{,}001$ draws")),      # last_digit
    ("experiments", ("$1/2001<0.05/17$", "$1/2002<0.05/17$")),             # last_digit
    ("method", (r"\tfrac{1}{N}\textstyle\sum_{n=1}^{N} z^A_i(d_n)",        # method.tex:13 swap_sent
                 r"\tfrac{1{,}500}{N}\textstyle\sum_{n=1}^{N} z^A_i(d_n)")),
    ("method", (r"From draws $M^{(1)},\dots,M^{(n_{\text{perm}})}$",       # method.tex:26 last_digit
                 r"From draws $M^{(0)},\dots,M^{(n_{\text{perm}})}$")),
    ("method", (r"p=\frac{1+\#\{b: M^{(b)}\ge M\}}{n_{\text{perm}}+1}.",   # method.tex:29 last_digit (numerator)
                 r"p=\frac{0+\#\{b: M^{(b)}\ge M\}}{n_{\text{perm}}+1}.")),
    ("method", (r"\{b: M^{(b)}\ge M\}}{n_{\text{perm}}+1}.",               # method.tex:29 last_digit (denominator)
                 r"\{b: M^{(b)}\ge M\}}{n_{\text{perm}}+0}.")),
    ("method", (r"the $K_A{+}K_B{-}1$ entries in row $i$ and column $j$",  # method.tex:35 swap_sent
                 r"the $K_A{+}K_B{-}1{,}500$ entries in row $i$ and column $j$")),
    ("method", ("first $2{,}000$ eligible", "first $0{,}200$ eligible")),  # method.tex:71 digit_swap
    ("method", ("a seeded random split of $20{,}000$ documents",          # method.tex:73 last_digit
                 "a seeded random split of $20{,}001$ documents")),
    ("results", ("every held-out $p=1/2001$", "every held-out $p=0/2001$")),  # results.tex:10 k_of_n
    ("results", ("confirms all nine at $p=1/2001$, which clears Bonferroni",    # results.tex:41 digit_swap
                  "confirms all nine at $p=1/2010$, which clears Bonferroni")),
    ("results", (r"mean${\pm}2$\,SD) for trained SAEs",                   # results.tex:14 swap_sent
                  r"mean${\pm}1$\,SD) for trained SAEs")),
    ("results", ("The $2{,}000$-document C4 screen", "The $2{,}001$-document C4 screen")),  # results.tex:32 last_digit
    ("results", ("strong on C4 ($z=42.7$)", "strong on C4 ($z=1$)")),      # results.tex:44 swap_sent
    ("results", ("moves its cell from $z=22.1$ to $14.1$", "moves its cell from $z=21.2$ to $14.1$")),  # digit_swap
    ("results", ("lexical rows use the governing $k{=}4096$", "lexical rows use the governing $k{=}0496$")),  # digit_swap
    ("results", ("$1.26$ vs.\\ $0.01$ of $30$ at $50$", "$2.16$ vs.\\ $0.01$ of $30$ at $50$")),  # digit_swap
    ("results", ("the excess holds at $20$ pairs for $\\beta\\in\\{2,8\\}$", "the excess holds at $21$ pairs for $\\beta\\in\\{2,8\\}$")),  # last_digit
    ("results", (r"\paragraph{Finding 2: the loose criteria are confounded.}",   # results.tex:143 last_digit
                  r"\paragraph{Finding 3: the loose criteria are confounded.}")),  # -> duplicate 3, breaks sequence
    ("results", (r"\paragraph{Finding 4 (templated prompts only): cross-lingual",  # results.tex:163 last_digit
                  r"\paragraph{Finding 3 (templated prompts only): cross-lingual")),  # -> duplicate 3
    ("results", (r"\paragraph{Finding 5: zero-shot probe transfer is in-sample only.}",  # results.tex:175 last_digit
                  r"\paragraph{Finding 4: zero-shot probe transfer is in-sample only.}")),  # -> duplicate 4
    ("limitations", ("finds $1$--$3\\%$ selection", "finds $2$--$3\\%$ selection")),  # last_digit
    # -- seed-1 re-run (<internal-notes>) --
    ("discussion", ("Finding~2), and read each", "Finding~3), and read each")),  # last_digit
    ("experiments", ("Qwen3-1.7B/8B-Base", "Qwen3-1.7B/2B-Base")),               # swap_sent
    ("experiments", ("and $17$ models for the residual scale-up.",              # last_digit
                      "and $18$ models for the residual scale-up.")),
    ("experiments", ("expanded to $206$ concept pairs", "expanded to $207$ concept pairs")),  # last_digit
    # 2026-10-10: the two old-caption mutations ("(2)~Reciprocal best matches", "(3)~Non-pre-registered")
    # retired with their sentences; replaced by mutations on the new Figure 1 caption:
    ("intro", ("$R=1$ means the control removes nothing", "$R=0$ means the control removes nothing")),  # last_digit
    ("intro", ("Top two rows: all nine cross-family cells", "Top two rows: all eight cross-family cells")),  # k_of_n
    ("limitations", ("the same $20{,}000$ documents. A split-half",             # last_digit
                      "the same $20{,}001$ documents. A split-half")),
    ("method", (r"$T_E(\mathcal{P})=|\mathcal{P}|^{-1}\sum",                    # last_digit
                 r"$T_E(\mathcal{P})=|\mathcal{P}|^{-0}\sum")),
    ("method", ("from $2{,}000$ permutations of $D_E$ alone.",                  # swap_sent
                 "from $1$ permutations of $D_E$ alone.")),
    ("related", ("Claude 3 Sonnet, and \\citet{lindsey2025biology}",            # last_digit
                  "Claude 4 Sonnet, and \\citet{lindsey2025biology}")),
    ("related", ("circuits in Claude 3.5 Haiku; both", "circuits in Claude 3.6 Haiku; both")),  # last_digit
    ("results", ("leave at least $0.96$ of the excess", "leave at least $0.97$ of the excess")),  # last_digit
    ("results", ("models from $70$M to $9$B)", "models from $70$M to $70$B)")),  # swap_sent
    ("results", ("The gate's $\\Delta R^2$ is a margin", "The gate's $\\Delta R^3$ is a margin")),  # last_digit
    ("results", ("median per-feature $R^2$ tops out at $0.14$",                 # last_digit (equation)
                  "median per-feature $R^1$ tops out at $0.14$")),
    ("results", ("tops out at $0.14$ captures", "tops out at $0.13$ captures")),  # last_digit (result)
    ("results", ("compare the identity of the top-$30$ promoted", "compare the identity of the top-$03$ promoted")),  # digit_swap
    ("results", ("separate the classes at $n=20$ antonym", "separate the classes at $n=02$ antonym")),  # digit_swap
    ("results", ("With $206$ concept pairs, $17$ models", "With $206$ concept pairs, $16$ models")),  # last_digit
    ("results", ("is the largest of all $17$ models. We report", "is the largest of all $71$ models. We report")),  # digit_swap
    # -- revision R1 (2026-10-07): every changed number/sentence, plus re-insertion of withdrawn text --
    ("abstract", ("{=}0.925$--$0.986$ of its", "{=}0.92$--$0.99$ of its")),                     # C5 revert
    ("abstract", ("and a Pythia-70M$\\leftrightarrow$Mistral-7B match null", "and a $70$M-to-$7$B match null")),  # C5 revert
    ("results", ("$0.927$/$0.982$/$0.979$", "$0.930$/$0.986$/$0.996$")),                         # C3 revert k=200
    ("results", ("$0.986$/$1.033$/$0.974$", "$0.983$/$1.060$/$1.012$")),                         # C3 revert k=500
    ("results", ("calibration averages $20$ random partitions", "calibration averages $2$ random partitions")),
    ("results", ("leaves $92$--$94\\%$ of it", "leaves $92$--$95\\%$ of it")),                    # C5 body
    ("results", ("$R_{\\text{lex}}=0.918$--$0.942$", "$R_{\\text{lex}}=0.918$--$0.924$")),
    ("method", ("by at most $3\\%$ (separately", "by at most $2\\%$ (separately")),
    ("method", ("cells were untested at submission", "cells are untested at submission")),
    ("related", ("measures largely disappears while", "measures disappears while")),               # C6 hedge dropped
    ("related", ("local-neighborhood similarity retains significant agreement.",
                 "only local-neighborhood agreement survives.")),                                  # C6 revert
    ("appendix", ("to within two pairs and the", "to within three pairs and the")),
    ("appendix", ("$R_{\\text{len}}=0.968$--$1.003$", "$R_{\\text{len}}=0.969$--$1.003$")),
    ("appendix", ("$R_{\\text{top}}=0.986$--$0.997$, every arm", "$R_{\\text{top}}=0.979$--$0.994$, every arm")),
    ("appendix", ("($R_{\\text{len}}=0.988$--$0.992$", "($R_{\\text{len}}=0.988$--$0.993$")),
    ("appendix", ("all $90$ held-out tests", "all $50$ held-out tests")),
    ("appendix", ("so the median is within $3\\%$", "so the median is within $2\\%$")),          # C4
    ("appendix", ("differ by up to $6.4\\%$", "differ by up to $6.5\\%$")),                      # C4
    ("appendix", ("single-split held-out excess is $0.494$.", "single-split held-out excess is $0.496$.")),
    ("appendix", ("agree to within $0.002$ in", "agree to within $0.001$ in")),
    ("appendix", ("is $1.05$--$1.46$ times", "is $1.05$--$1.45$ times")),
    ("appendix", ("is $0.56$--$1.48$ times the bootstrap SD", "is $0.65$--$1.48$ times the bootstrap SD")),
    ("appendix", ("in only $65\\%$ and $70.5\\%$", "in only $65\\%$ and $71\\%$")),
    ("appendix", ("$15$ of the $42$ intervals", "$15$ of the $24$ intervals")),
    ("appendix", ("sit $1.2$--$2.8$ SD below", "sit $1.2$--$2.6$ SD below")),
    ("appendix", ("the registered reading is C0 (instrument failed)", "the registered reading is C1 (instrument failed)")),
    ("appendix", ("(branch B1).", "(branch B1; and, for $\\Delta_E$, C1).")),                      # C1 re-inserted
    ("appendix", ("No verdict rested on these intervals;", "Yet these intervals lean low; no verdict rested on these intervals;")),
    ("appendix", ("they are $0.927$--$1.033$,", "they are $0.927$--$1.060$,")),
    ("appendix", ("one cluster of $1{,}490$ documents", "one cluster of $1{,}409$ documents")),       # T12
    ("appendix", ("$493$ single documents", "$439$ single documents")),
    ("appendix", ("that leaves $1.002$, $0.993$ and $0.964$", "that leaves $1.002$, $0.939$ and $0.964$")),
    ("tab_ninecell", ("Q1.7--G2 & $811$ & $1.003$", "Q1.7--G2 & $811$ & $1.030$")),
    # -- R1 seed-1 re-sweep (/tmp/ct_mutsweep_r1c_seed1/report.md) --
    ("method", ("about half of the $1{,}500$ screened", "about half of the $5{,}100$ screened")),
    ("method", (r"|\widehat{\mathcal{P}}|^{-1} \sum_{(i,j)\in\widehat{\mathcal{P}}}\mathbb{E}",
                r"|\widehat{\mathcal{P}}|^{+1} \sum_{(i,j)\in\widehat{\mathcal{P}}}\mathbb{E}")),
    # -- R1 seed-0 re-sweep after the text moved (/tmp/ct_mutsweep_r1b_seed0/report.md) --
    ("experiments", ("a shared $206$-prompt English", "a shared $207$-prompt English")),
    ("intro", ("two sets of $1{,}500$ features reciprocally", "two sets of $5{,}100$ features reciprocally")),
    ("results", ("(a)~Pairs selected on $1{,}000$ C4", "(a)~Pairs selected on $0{,}100$ C4")),
    # ("results", "$R=1$ means ...") moved to section "intro" on 2026-10-10 with the Figure 1 redraw (above).
    # -- R1 seed-2 sweep (/tmp/ct_mutsweep_r1_seed2/report.md): the six misses, reproduced --
    ("abstract", ("two sets of $1{,}500$ features", "two sets of $5{,}100$ features")),
    ("intro", ("of at least $0.88$ against its reference arm", "of at least $0.89$ against its reference arm")),
    ("method", ("$D_S,D_E$ ($1{,}000$ C4 documents each)", "$D_S,D_E$ ($0{,}100$ C4 documents each)")),
    ("results", ("and Qwen3-1.7B/8B (Qwen-Scope", "and Qwen3-1.7B/1B (Qwen-Scope")),
    ("results", ("$0.928$/$0.920$/$0.885$ & $1/2001$", "$0.928$/$0.920$/$0.885$ & $2/2001$")),
    ("results", ("$0.522$/$0.398$/$0.335$ & $1/2001$", "$0.522$/$0.398$/$0.335$ & $1/2000$")),
    ("tab_ninecell", ("& $0.566$ & $0.530$--$0.565$", "& $0.566$ [0.548, 0.567] & $0.530$--$0.565$")),  # interval back
]


def selftest():
    sections = load_sections()
    d = derive()
    base_bad = scan(sections, d)
    results = [("clean HEAD passes", not base_bad)]
    for section, (old, new) in MUTATIONS:
        old_n, new_n = norm(old), norm(new)
        text = sections[section]
        if old_n not in text:
            results.append((f"{section}: {old_n[:50]!r} setup", False))
            continue
        if old_n == new_n:
            continue  # placeholder row, not a real case
        mutated = sections.copy()
        mutated[section] = text.replace(old_n, new_n, 1)
        bad = scan(mutated, d)
        caught = bool(bad)
        results.append((f"{section}: {old_n[:40]!r} -> {new_n[:40]!r}", caught))

    # null-edit control: a whitespace-only change plus an edit to a number that is NOT
    # covered by any pin here (a models/revisions citation key) must still pass.
    null_text = dict(sections)
    null_text["results"] = null_text["results"].replace(
        "mean-pooled features", "mean - pooled   features")
    null_bad = scan(null_text, d)
    results.append(("null edit (whitespace-only, unrelated text) still passes", not null_bad))

    ok = True
    for name, got in results:
        print(f"  {name:70s} {'OK' if got else 'FAILED'}")
        ok = ok and got
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())

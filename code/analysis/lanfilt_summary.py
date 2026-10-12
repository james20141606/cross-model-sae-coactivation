#!/usr/bin/env python3
"""Read the eight lanfilt{T,C}_*__mistral7b.json receipts, apply the pre-registered decision
rules (notes/PREREG_lan_filters_20260924.md §5-6) mechanically, write
results/cross_model_sae/LANFILT_DECISIONS.json and emit paper/sections/tab_lanfilt.tex.

Unfiltered comparators (the "63-94%" shares) are read from the Table 19 receipts
lan{T,C}_*__mistral7b.json, not from the local re-extraction; G0 bounds the difference.

    /usr/bin/python3 code/analysis/lanfilt_summary.py
"""
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/cross_model_sae"
TAB = ROOT / "paper/sections/tab_lanfilt.tex"
PAIRS = [("gemma2-9b", "Gemma-2-9B$\\leftrightarrow$Mistral (survives)"),
         ("qwen3-1.7b", "Qwen3-1.7B$\\leftrightarrow$Mistral (survives)"),
         ("pythia70m", "Pythia-70M$\\leftrightarrow$Mistral (artifact)"),
         ("gpt2", "GPT-2$\\leftrightarrow$Mistral (null)")]
NONNULL = ["gemma2-9b", "qwen3-1.7b", "pythia70m"]
CORP = [("T", "templates"), ("C", "C4")]


def load():
    cells = {}
    for p, _ in PAIRS:
        for L, _ in CORP:
            cells[(p, L)] = json.loads((RES / f"lanfilt{L}_{p}__mistral7b.json").read_text())
    return cells


def unfiltered_share(p, L):
    corpus = "templates" if L == "T" else "c4"
    x = json.loads((RES / f"lan{L}_{p}__mistral7b.json").read_text())[corpus]
    return (x["rsa_permuted_pairing"] - x["rsa_random_pairing"]) / (x["rsa"] - x["rsa_random_pairing"])


def excess(f):
    """exc_f if perm estimable, else the pre-registered fallback obs - rand."""
    if f.get("perm_estimable"):
        return f["rsa"] - f["rsa_perm"], "exc"
    return f["rsa"] - f["rsa_rand"], "obs-rand"


def compare(f1, f2):
    d = excess(f1)[0] - excess(f2)[0]
    yard = 2 * math.sqrt(f1["rsa_rand_sd"] ** 2 + f2["rsa_rand_sd"] ** 2)
    band = max(0.05, yard)
    return {"diff": d, "yardstick": yard, "band": band, "separated": abs(d) >= band,
            "kinds": [excess(f1)[1], excess(f2)[1]]}


def decide(cells):
    P = {k: v["filtered"]["P"] for k, v in cells.items()}
    g0 = {f"{p}_{L}": {"pass": v["G0"]["pass"], "d_rsa": v["G0"]["delta_rsa"],
                       "d_svcca": v["G0"]["delta_svcca"]} for (p, L), v in cells.items()}
    n_fail = sum(not x["pass"] for x in g0.values())
    # D1
    per = {}
    persist = removed = 0
    for p in NONNULL:
        for L, _ in CORP:
            f = P[(p, L)]
            if not f["estimable"]:
                per[f"{p}_{L}"] = "not testable"; continue
            if not f["perm_estimable"]:
                per[f"{p}_{L}"] = "removed (perm not estimable)"; removed += 1; continue
            gap = f["rsa_perm"] - f["rsa_rand"]
            if gap >= 0.02 and f["rsa_share"] >= 0.10:
                per[f"{p}_{L}"] = "persists"; persist += 1
            elif gap < 0.02:
                per[f"{p}_{L}"] = "removed (gap<0.02)"; removed += 1
            else:
                per[f"{p}_{L}"] = "neither (gap>=0.02, share<0.10)"
    d1 = "persists" if persist >= 4 else "removed" if removed >= 4 else "mixed"
    # D2
    g, py = P[("gemma2-9b", "T")], P[("pythia70m", "T")]
    if not (g["estimable"] and py["estimable"]):
        d2 = {"decision": "not testable"}
    else:
        c = compare(g, py)
        dec = "persists" if not c["separated"] else ("removed" if c["diff"] > 0 else "reversed")
        d2 = {"decision": dec, **c}
    # D3
    q, pc, gc = P[("qwen3-1.7b", "C")], P[("pythia70m", "C")], P[("gemma2-9b", "C")]
    if not (q["estimable"] and pc["estimable"]):
        d3 = {"decision": "not testable"}
    else:
        c = compare(q, pc)
        dec = "removed" if (c["separated"] and c["diff"] > 0) else "persists"
        d3 = {"decision": dec, **c,
              "gemma_vs_qwen": compare(gc, q), "gemma_vs_pythia": compare(gc, pc)}
    return {"prereg": "notes/PREREG_lan_filters_20260924.md (commit 4cbd289)",
            "G0": g0, "G0_n_fail": n_fail,
            "G0_max_abs_delta": max(max(abs(x["d_rsa"]), abs(x["d_svcca"])) for x in g0.values()),
            "D1": {"decision": d1, "per_cell": per, "n_persist": persist, "n_removed": removed},
            "D2": d2, "D3": d3,
            "unfiltered_rsa_share_receipts": {f"{p}_{L}": unfiltered_share(p, L)
                                              for p in NONNULL for L, _ in CORP}}


def f3(x):
    s = f"{x:.3f}"
    if s == "-0.000":
        s = "0.000"                                    # no negative zero
    return f"${s}$".replace("$-", "${-}")


def emit(cells):
    P = {k: v["filtered"]["P"] for k, v in cells.items()}
    rows = []
    for p, label in PAIRS:
        for i, (L, cname) in enumerate(CORP):
            f = P[(p, L)]
            name = label if i == 0 else ""
            kept = f"${f['K_f']}$"
            if not f["estimable"]:
                rows.append(f"{name} & {cname} & {kept} & \\multicolumn{{7}}{{c}}{{no pair kept}} \\\\")
                continue
            perm = f3(f["rsa_perm"]) if f["perm_estimable"] else \
                f"n.e.\\ (${round(f['perm_estimable_frac'] * 200)}/200$)"
            exc, kind = excess(f)
            exc_s = f3(exc) + ("" if kind == "exc" else "$^{\\dagger}$")
            share = f"${100 * f['rsa_share']:.1f}\\%$" if f["perm_estimable"] else "---"
            rows.append(f"{name} & {cname} & {kept} & {f3(f['rsa'])} & {f3(f['rsa_rand'])} & {perm} & "
                        f"{exc_s} & {share} & {f3(f['svcca'])} & {f3(f['svcca_rand'])} \\\\")
        if p != PAIRS[-1][0]:
            rows.append("\\addlinespace")
    body = "\n".join(rows)
    tex = f"""% AUTO-GENERATED by code/analysis/lanfilt_summary.py -- DO NOT HAND-EDIT.
% Numbers formatted from results/cross_model_sae/lanfilt{{T,C}}_*__mistral7b.json (filtered["P"]).
\\begin{{table}}[h]
\\centering
\\caption{{\\textbf{{The Lan et al.\\ protocol with their filters.}} Primary pre-registered variant
(non-concept and shared-keyword filter on whitespace-stripped top-5 token strings, their
code's one-to-one step, $r{{>}}0.1$). kept $=$ pairs left of the $512$ greedy pairs. RSA of
the kept pairs' decoder rows (obs), their shuffle baseline within the kept pairs (rand, $200$
draws), and our permuted-pairing null pushed through all three filters (perm, $200$ draws;
n.e.\\ $=$ fewer than half the draws keep the $21$ pairs SVCCA needs, with the number that do).
exc.\\ $=$ obs $-$ perm; $^{{\\dagger}}$obs $-$ rand where perm is n.e.\\ (the pre-registered
fallback). share $=$ (perm $-$ rand)/(obs $-$ rand), the quantity that is $63$--$94\\%$ on the
templated corpus without the filters (\\Cref{{tab:lan}}). SVCCA is near its ceiling on so few
rows, so its random baseline rises as kept shrinks; it is shown, not read. Receipts:
\\texttt{{lanfilt\\{{T,C\\}}\\_*.json}}; pre-registration:
\\texttt{{notes/PREREG\\_lan\\_filters\\_20260924.md}}.}}
\\label{{tab:lanfilt}}
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llr rrrrr rr}}
\\toprule
& & & \\multicolumn{{5}}{{c}}{{RSA}} & \\multicolumn{{2}}{{c}}{{SVCCA}} \\\\
\\cmidrule(lr){{4-8}}\\cmidrule(lr){{9-10}}
Pair (our verdict) & corpus & kept & obs & rand & perm & exc. & share & obs & rand \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
"""
    TAB.write_text(tex)


def main():
    cells = load()
    dec = decide(cells)
    (RES / "LANFILT_DECISIONS.json").write_text(json.dumps(dec, indent=1))
    emit(cells)
    print(json.dumps({k: dec[k] for k in ("G0_n_fail", "G0_max_abs_delta")}, indent=1))
    print("D1", dec["D1"]); print("D2", dec["D2"]); print("D3", {k: v for k, v in dec["D3"].items()
                                                                if not k.startswith("gemma")})
    print("D3 gemma", dec["D3"].get("gemma_vs_qwen"), dec["D3"].get("gemma_vs_pythia"))
    print("unfiltered shares", {k: round(v, 3) for k, v in dec["unfiltered_rsa_share_receipts"].items()})
    print(f"wrote {TAB.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())

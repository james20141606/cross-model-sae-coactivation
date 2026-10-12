#!/usr/bin/env python3
"""Combine the read-point sensitivity receipts and emit paper/sections/tab_gemmahook.tex
(prereg notes/PREREG_gemma_hook_sensitivity_20260924.md).

Inputs (all written by gemma_hook_sensitivity.py match):
  results/cross_model_sae/gemma_hook_sensitivity/GEMMA_HOOK_SENSITIVITY.json        local MPS run
      (unsloth Gemma re-uploads; the registered primary environment)
  results/cross_model_sae/gemma_hook_sensitivity/cluster/GEMMA_HOOK_SENSITIVITY.json       cluster 18441889
  results/cross_model_sae/gemma_hook_sensitivity/cluster_q8g2/GEMMA_HOOK_SENSITIVITY.json  cluster 18445512
      (original google/gemma-2-* weights), OPTIONAL, merged in that order. Added after the local Qwen3-8B-Base download
      failed; it is the source for the two Qwen3-8B cells only (deviation, disclosed). For every
      other cell the local run stays the registered primary and the cluster run is reported as a replication.

Column N = the published receipts (c4_*.json, heldout_*.json; read point hidden_states[N]);
column N+1 = the re-run at the training hook. Nothing numeric is typed here.
Writes GEMMA_HOOK_COMBINED.json (per-cell source, gates, decision, prereg outcome on the blind
cells) next to the inputs.

    /usr/bin/python3 code/analysis/gemmahook_summary.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "results/cross_model_sae/gemma_hook_sensitivity"
REC_LOCAL = DIR / "GEMMA_HOOK_SENSITIVITY.json"
# cluster receipts, in order of precedence: 18441889 (G9 cells; its gemma2-2b extract failed, the 65k
# params were not in the offline cache) and 18445512 (2026-09-25 rerun for the gemma2-2b cells).
REC_CLUSTER = [DIR / "cluster" / "GEMMA_HOOK_SENSITIVITY.json",
             DIR / "cluster_q8g2" / "GEMMA_HOOK_SENSITIVITY.json"]
OUT = DIR / "GEMMA_HOOK_COMBINED.json"
PUB = ROOT / "results/cross_model_sae"
TAB = ROOT / "paper/sections/tab_gemmahook.tex"

NAME = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-8b": "Qwen3-8B", "gemma2-2b": "Gemma2-2B",
        "gemma2-9b": "Gemma2-9B", "mistral7b": "Mistral-7B"}
HELD = ["qwen3-1.7b__gemma2-2b", "qwen3-8b__gemma2-2b", "qwen3-1.7b__gemma2-9b",
        "qwen3-8b__gemma2-9b", "gemma2-9b__mistral7b", "gemma2-2b__mistral7b"]
SCREEN_ONLY = ["gemma2-2b__gemma2-9b", "w65k_gemma2-2b__mistral7b"]
CLUSTER_PRIMARY = {"qwen3-8b__gemma2-2b", "qwen3-8b__gemma2-9b"}   # deviation: local run impossible
NON_BLIND = {"gemma2-9b__mistral7b"}


def label(tag):
    w65 = tag.startswith("w65k_")
    a, b = tag.replace("w65k_", "").split("__")
    return (NAME[a] + ("(w$65$k)" if w65 else "")) + " $\\leftrightarrow$ " + NAME[b]


def f1(x):
    return f"${x:.1f}$"


def f3(x):
    return f"${x:.3f}$"


def load(p):
    return json.load(open(p)) if p.exists() else None


def load_cluster():
    """Merge the cluster receipts; for every key the first receipt that has it wins."""
    parts = [(p, load(p)) for p in REC_CLUSTER]
    parts = [(p, R) for p, R in parts if R is not None]
    if not parts:
        return None
    M = {"G0": {}, "decision": {}, "H": {}, "screen": {}, "receipt_of": {}}
    for p, R in parts:
        for k in ("G0", "decision", "H", "screen"):
            for t, v in R.get(k, {}).items():
                M[k].setdefault(t, v)
        for t in R.get("decision", {}):
            M["receipt_of"].setdefault(t, str(p.relative_to(ROOT)))
    return M


def combine():
    L, T = load(REC_LOCAL), load_cluster()
    runs = {"local": L, "cluster": T}
    cells = {}
    for tag in HELD:
        src = "cluster" if tag in CLUSTER_PRIMARY else "local"
        R = runs[src]
        c = dict(source=src, non_blind=tag in NON_BLIND, ran=False)
        if R is not None and tag in R.get("decision", {}):
            d = R["decision"][tag]
            c.update(ran=True, G0=R["G0"].get(tag), decision=d,
                     H={k: v["pass_"] for k, v in R["H"].items()})
            if src == "cluster":
                c["receipt"] = R["receipt_of"][tag]
        rep = "cluster" if src == "local" else "local"
        if runs[rep] is not None and tag in runs[rep].get("decision", {}):
            c["replication"] = dict(source=rep, decision=runs[rep]["decision"][tag],
                                    G0=runs[rep]["G0"].get(tag))
            if rep == "cluster":
                c["replication"]["receipt"] = T["receipt_of"][tag]
        cells[tag] = c
    screen = {}
    for tag in SCREEN_ONLY:
        s = (L or {}).get("screen", {}).get(tag, {}).get("1")
        screen[tag] = dict(source="local", z_N=json.load(open(PUB / f"c4_{tag}.json"))["mutual_z"],
                           z_N1=s["z"] if s else None, G0=(L or {}).get("G0", {}).get(tag))
    blind = [c["decision"] for t, c in cells.items()
             if c["ran"] and not c["non_blind"] and not c["decision"]["blocked_by_gate"]]
    outcome = None
    if blind:
        if not all(d["retained"] for d in blind):
            outcome = "C"
        elif all(d["in_band"] for d in blind):
            outcome = "A"
        else:
            outcome = "B"
    not_run = [t for t, c in cells.items() if not c["ran"]]
    blocked = [t for t, c in cells.items() if c["ran"] and c["decision"]["blocked_by_gate"]]
    evaluated = [t for t, c in cells.items()
                 if c["ran"] and not c["non_blind"] and not c["decision"]["blocked_by_gate"]]
    H = {}
    for name, R in (("local", L), ("cluster", T)):
        if R:
            for k, v in R["H"].items():
                H.setdefault(k, {})[name] = v
    return dict(prereg="notes/PREREG_gemma_hook_sensitivity_20260924.md",
                inputs={"local": str(REC_LOCAL.relative_to(ROOT)) if L else None,
                        "cluster": [str(p.relative_to(ROOT)) for p in REC_CLUSTER if p.exists()]},
                cells=cells, screen_only=screen, H=H,
                blind_cells_evaluated=evaluated, blind_cells_blocked_by_gate=blocked,
                cells_not_run=not_run, outcome_on_blind_cells=outcome)


def g0_str(g):
    if g is None:
        return "--"
    return "pass" if g["pass"] else f"fail ($\\Delta M{{=}}{g['dM']}$)"


def verdict(c):
    if not c["ran"]:
        return "not run"
    d = c["decision"]
    if d["blocked_by_gate"]:
        return "not interpreted$^{\\dagger}$"
    v = "retained" if d["retained"] else "\\textbf{not retained}"
    return v + ("$^{*}$" if c["non_blind"] else "")


def rows(C):
    out = []
    for tag in HELD:
        c = C["cells"][tag]
        pz = json.load(open(PUB / f"c4_{tag}.json"))["mutual_z"]
        ph = json.load(open(PUB / f"heldout_{tag}.json"))
        env = "cluster" if c["source"] == "cluster" else "local"
        if not c["ran"]:
            env = "--"
            out.append(f"{label(tag)} & {env} & -- & {f1(pz)} / -- & {f3(ph['held_out']['excess'])} / --"
                       f" & {f3(ph['ratio_heldout_over_insample'])} / -- & not run \\\\")
            continue
        d = c["decision"]
        out.append(f"{label(tag)} & {env} & {g0_str(c['G0'])} & {f1(d['z_N'])} / {f1(d['z_N1'])}"
                   f" & {f3(d['excess_N'])} / {f3(d['excess_N1'])}"
                   f" & {f3(d['r_N'])} / {f3(d['r_N1'])} & {verdict(c)} \\\\")
    out.append("\\midrule")
    for tag in SCREEN_ONLY:
        s = C["screen_only"][tag]
        z1 = f1(s["z_N1"]) if s["z_N1"] is not None else "--"
        out.append(f"{label(tag)} & local & {g0_str(s['G0'])} & {f1(s['z_N'])} / {z1} & -- & -- & screen only \\\\")
    return out


def emit(C):
    body = "\n".join(rows(C))
    TAB.write_text(
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{\\textbf{Gemma Scope cells read at the dictionaries' training hook} (pre-registered;\n"
        "\\Cref{app:gemmahook}). Each entry is $N$ / $N{+}1$: $N$ is the published value\n"
        "(\\texttt{hidden\\_states[$N$]}, the input to block $N$; \\Cref{tab:c4_counts,tab:c4_verdicts}),\n"
        "$N{+}1$ reads \\texttt{hidden\\_states[$N{+}1$]}, the \\texttt{resid\\_post} hook each Gemma Scope\n"
        "SAE was trained on, all else fixed. $z$: C4-screen mutual-NN $z$ ($200$ permutations); held-out\n"
        "excess and retained fraction as in \\Cref{tab:c4_verdicts} ($2{,}000$ permutations). G0: the\n"
        "re-run at $N$ reproduces the published receipt (reciprocal count within $2$, held-out excess\n"
        "within $0.005$). Verdict at $N{+}1$ by the registered rule (held-out $p=1/2001$, excess $>0$,\n"
        "screen $p\\le1/201$). Env: local = Apple MPS with ungated Gemma re-uploads; cluster = original\n"
        "weights. $^{*}$Seen before registration; not counted. $^{\\dagger}$Failed G0; not interpreted.\n"
        "Receipts: \\texttt{results/cross\\_model\\_sae/gemma\\_hook\\_sensitivity/}.}\n"
        "\\label{tab:gemmahook}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n"
        "\\begin{tabular}{llllllr}\n\\toprule\n"
        "Cell & env & G0 & $z$ & held-out excess & retained & verdict at $N{+}1$ \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


if __name__ == "__main__":
    C = combine()
    json.dump(C, open(OUT, "w"), indent=1)
    emit(C)
    print(json.dumps(dict(outcome=C["outcome_on_blind_cells"], evaluated=C["blind_cells_evaluated"],
                          blocked=C["blind_cells_blocked_by_gate"], not_run=C["cells_not_run"]), indent=1))
    print(f"wrote {TAB} and {OUT}")

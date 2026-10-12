#!/usr/bin/env python3
"""Pin every token-aligned number the paper prints to the tokalign20k receipts.
Exit 1 on mismatch; --selftest plants wrong values and must catch them.

Fixed after external review (RESCOPE, 2026-08-16): the previous scan() accepted the
abstract substring "r{=}0.34" as satisfying ALL three S1 checks (any `v` passed once
that literal was present). Now each of the three exact S1 values must appear verbatim
in the scanned scope, and the scope includes appendix.tex and conclusion.tex, which
also carry the numbers.

Hardened after adversarial audit (2026-08-19), closing five demonstrated passes:

  M1/M3  the abstract's per-cell S1 values were unpinned AS AN ORDERED TUPLE in the
         abstract itself: mutating one value, or transposing two, passed because
         copies elsewhere satisfied the presence checks. PER_FILE now requires each
         file that prints a receipt triple to carry it in receipt order.
  M2     substring containment: "0.3421" was accepted as containing "0.342". All
         matching is now boundary-aware ((?<![\\d.]) value (?![\\d.])).
  M4     the abstract's R_tok_clean range "0.93--0.99" was unpinned. The endpoints
         are now derived from the receipts (the 2-dp rounding of the min/max of the
         3-dp printed values -- the paper's own convention) and pinned. Revision R1
         (2026-10-07): the abstract prints 3 dp (0.925--0.986), derived the same way.
  M5     the generated tab_tokalign.tex was outside FILES entirely: any mutation of
         its cells passed. It is now scanned, and each row's S1 (3 dp) and R_tok
         (2 dp) cells are pinned per-cell to the receipts by row label.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CELLS = ["qwen3-1.7b__gemma2-2b", "gemma2-9b__mistral7b", "llama3.2-1b__mistral7b"]
# row labels as the emitter writes them (emit_tokalign_table.py), keyed to CELLS order
TAB_LABELS = [
    "Qwen3-1.7B$\\leftrightarrow$Gemma-2-2B",
    "Gemma-2-9B$\\leftrightarrow$Mistral-7B",
    "Llama-3.2-1B$\\leftrightarrow$Mistral-7B",
]
TAB = "paper/sections/tab_tokalign.tex"
FILES = [
    "paper/sections/results.tex",
    "paper/sections/abstract.tex",
    "paper/sections/limitations.tex",
    "paper/sections/appendix.tex",
    "paper/sections/conclusion.tex",
    TAB,
]


def receipt_vals():
    r1, rt = [], []
    for c in CELLS:
        d = json.loads((ROOT / f"results/cross_model_sae/tokalign20k_{c}.json").read_text())
        assert not d.get("smoke"), f"{c}: refusing to pin claims to a smoke receipt"
        r1.append(f"{d['s1']['obs_mean_pair_r']:.3f}")
        rt.append(f"{d['s2']['R_tok']:.3f}")
    return r1, rt


def clean_receipt_vals():
    """A3-EXEC leakage-clean receipts: clean S1 (4 d.p., split-B) and R_tok_clean."""
    r1c, rtc, rtc_raw, rtcn = [], [], [], []
    for c in CELLS:
        d = json.loads(
            (ROOT / f"results/cross_model_sae/tokaclean20k_{c}.json").read_text())
        assert not d.get("smoke"), f"{c}: refusing to pin claims to a smoke clean receipt"
        assert all(d["a3_requirements"].values()), \
            f"{c}: clean receipt missing an A3 requirement marker"
        r1c.append(f"{d['s1']['obs_mean_pair_r']:.4f}")
        rtc.append(f"{d['s2_clean']['R_tok_clean']:.3f}")
        rtc_raw.append(float(d['s2_clean']['R_tok_clean']))
        # naive (not support-matched) clean ratio: unpinned until the 2026-09-24 final
        # review found A3-stale 0.904/0.961 in the appendix passing this gate.
        rtcn.append(f"{d['s2_clean']['R_tok_clean_naive']:.3f}")
    return r1c, rtc, rtc_raw, rtcn


def splithalf_receipt_vals():
    """Pooled split-half receipts (app:threats (iv)): held-out/in-sample ratio per cell,
    both directions averaged, and the held-out p floor. Pinned so the appendix cannot
    drift from splithalf20k_*.json the way the A3 numbers once drifted from theirs."""
    ratios, raw, pmax = [], [], 0.0
    for c in CELLS:
        d = json.loads((ROOT / f"results/cross_model_sae/splithalf20k_{c}.json").read_text())
        assert d["split"]["n_a"] == d["split"]["n_b"] == 10000 and d["n_perm"] == 2000, \
            f"{c}: split-half receipt is not the registered 10k/10k, n_perm=2000 design"
        r = float(d["summary"]["mean_ratio_heldout_over_insample"])
        ratios.append(f"{r:.2f}"); raw.append(r)
        pmax = max(pmax, float(d["summary"]["min_heldout_p"]))
    return ratios, raw, pmax


def derived():
    """Every string this gate pins, derived from the receipts -- never typed here."""
    r1, rt = receipt_vals()
    r1c, rtc, rtc_raw, rtcn = clean_receipt_vals()
    sh, sh_raw, sh_pmax = splithalf_receipt_vals()
    rt2, cs1_3 = [], []
    for c in CELLS:
        d = json.loads((ROOT / f"results/cross_model_sae/tokalign20k_{c}.json").read_text())
        rt2.append(f"{d['s2']['R_tok']:.2f}")
        dc = json.loads((ROOT / f"results/cross_model_sae/tokaclean20k_{c}.json").read_text())
        cs1_3.append(f"{dc['s1']['obs_mean_pair_r']:.3f}")
    # The abstract's range endpoints: the paper rounds the 3-dp PRINTED R_tok_clean
    # values to 2 dp (0.925 -> 0.93, 0.986 -> 0.99). Derive with the same convention
    # so the pin follows the receipts, and any receipt drift breaks the match.
    # Round the RAW receipt values once (0.9247 -> 0.92), never the printed 3-dp
    # values (0.925 -> 0.93): the 2026-09-03 max-effort review caught that double
    # rounding in the abstract/limitations/conclusion range.
    # Revision R1 (2026-10-07, abstract-gate item C5): the abstract now prints the range at the
    # 3 dp of Table 1 (0.925--0.986), so the body carries the same tokens; still one rounding of
    # the RAW receipt values, never a re-rounding of printed ones.
    lo = f"{min(rtc_raw):.3f}"
    hi = f"{max(rtc_raw):.3f}"
    return {
        "s1": r1, "rt": rt, "rt2": rt2, "cs1_4": r1c, "cs1_3": cs1_3, "rtc": rtc,
        "s1_triple": "/".join(r1), "rt_triple": "/".join(rt),
        "cs1_3_triple": "/".join(cs1_3), "cs1_4_triple": "/".join(r1c),
        "rtc_triple": "/".join(rtc), "rtc_range": f"{lo}--{hi}",
        "rtcn_triple": "/".join(rtcn),
        "sh_triple": "/".join(sh), "sh_range": f"{min(sh_raw):.2f}--{max(sh_raw):.2f}",
        "sh_pfloor": ("1/2001" if abs(sh_pmax - 1 / 2001) < 1e-12
                       else f"{sh_pmax:.8g}"),
    }


def bounded(text, val):
    """Boundary-aware containment: '0.3421' must NOT satisfy a check for '0.342'."""
    return re.search(r"(?<![\d.])" + re.escape(val) + r"(?![\d.])", text) is not None


# Each file that prints a receipt triple must carry it IN RECEIPT ORDER; a value
# present elsewhere in the paper must not stand in for a mutated or transposed copy.
PER_FILE = {
    "paper/sections/abstract.tex": [
        ("abstract clean-S1 ordered triple", "cs1_3_triple"),
        ("abstract R_tok_clean range", "rtc_range"),
    ],
    "paper/sections/results.tex": [
        ("results clean-S1 ordered triple", "cs1_3_triple"),
        ("results R_tok ordered triple", "rt_triple"),
        ("results R_tok_clean ordered triple", "rtc_triple"),
    ],
    "paper/sections/appendix.tex": [
        ("appendix first-design S1 ordered triple", "s1_triple"),
        ("appendix clean-S1 ordered triple (4dp)", "cs1_4_triple"),
        ("appendix R_tok_clean ordered triple", "rtc_triple"),
        ("appendix R_tok_clean naive ordered triple", "rtcn_triple"),
        ("appendix pooled split-half ratio ordered triple", "sh_triple"),
        ("appendix pooled split-half held-out p floor", "sh_pfloor"),
    ],
}


def check_tab(tab_plain, D):
    """Pin each row's S1 (3 dp) and R_tok (2 dp) cells to the receipts, by row label."""
    bad = []
    lines = tab_plain.splitlines()
    for i, label in enumerate(TAB_LABELS):
        key = label.replace("$", "")
        row = next((ln for ln in lines if key in ln), None)
        if row is None:
            bad.append((f"tab_tokalign row {CELLS[i]}", "row not found"))
            continue
        if not bounded(row, D["s1"][i]):
            bad.append((f"tab_tokalign S1 cell {CELLS[i]}", D["s1"][i]))
        if not bounded(row, D["rt2"][i]):
            bad.append((f"tab_tokalign R_tok cell {CELLS[i]}", D["rt2"][i]))
        # A4 columns: unpinned until the 2026-08-21 final review demonstrated an
        # A3-stale R_tok_clean cell passing. Pin both to the tokaclean receipts.
        if not bounded(row, D["cs1_3"][i]):
            bad.append((f"tab_tokalign S1_clean cell {CELLS[i]}", D["cs1_3"][i]))
        if not bounded(row, D["rtc"][i]):
            bad.append((f"tab_tokalign R_tok_clean cell {CELLS[i]}", D["rtc"][i]))
    return bad


def scan(files):
    """files: {relpath: raw tex}. Global boundary-aware presence checks over the whole
    scope, plus per-file ordered-triple checks, plus per-cell table pins."""
    D = derived()
    plain_all = "".join(files[f] for f in FILES).replace("$", "")
    bad = []
    for v in D["s1"]:
        if not bounded(plain_all, v):
            bad.append(("S1 r", v))
    for what, key in (("R_tok triple", "rt_triple"),
                      ("clean S1 triple", "cs1_4_triple"),
                      ("R_tok_clean triple", "rtc_triple")):
        if not bounded(plain_all, D[key]):
            bad.append((what, D[key]))
    for f, reqs in PER_FILE.items():
        plain_f = files[f].replace("$", "")
        for what, key in reqs:
            if not bounded(plain_f, D[key]):
                bad.append((what, D[key]))
    bad += check_tab(files[TAB].replace("$", ""), D)
    return bad


def load_files():
    return {f: (ROOT / f).read_text() for f in FILES}


def main():
    bad = scan(load_files())
    for what, v in bad:
        print(f"MISMATCH tokalign {what}: receipt value {v} not in paper")
    print("check_tokalign_claims:", "FAIL" if bad else "PASS",
          f"({len(CELLS)} cells pinned to receipts across {len(FILES)} tex files, "
          f"ordered per-file triples + per-cell table pins)")
    return 1 if bad else 0


def selftest():
    files = load_files()
    D = derived()
    ABS = "paper/sections/abstract.tex"
    RESF = "paper/sections/results.tex"

    def mutate(f, old, new):
        m = dict(files)
        if old not in m[f]:          # triples are typeset as $a$/$b$/$c$; scan strips $
            m[f] = m[f].replace("$", "")
        assert old in m[f], f"selftest plant target {old!r} not present in {f}"
        m[f] = m[f].replace(old, new)
        return m

    checks = [("clean pass", "must be clean", not scan(files))]
    # legacy plants: corrupt a value/triple EVERYWHERE -> global checks must fire
    every = {f: t.replace(D["s1"][0], "9.999") for f, t in files.items()}
    checks.append(("global S1 plant", "must fire",
                   any(w == "S1 r" for w, _ in scan(every))))
    every = {f: t.replace(D["rt"][1], "9.999") for f, t in files.items()}
    checks.append(("global R_tok plant", "must fire",
                   any("R_tok" in w for w, _ in scan(every))))
    every = {f: t.replace(D["rtc"][1], "9.999") for f, t in files.items()}
    checks.append(("global R_tok_clean plant", "must fire",
                   any("R_tok_clean" in w for w, _ in scan(every))))
    every = {f: t.replace(D["cs1_4"][0], "9.9999") for f, t in files.items()}
    checks.append(("global clean-S1 plant", "must fire",
                   any("clean S1" in w for w, _ in scan(every))))
    # M1: one value mutated in the abstract only (copies elsewhere intact)
    m = mutate(ABS, D["cs1_3_triple"],
               D["cs1_3_triple"].replace(D["cs1_3"][0], "0.523", 1))
    checks.append(("M1 abstract one-value plant", "must fire",
                   any("abstract clean-S1" in w for w, _ in scan(m))))
    # M3: transposition in the abstract only
    t = "/".join([D["cs1_3"][1], D["cs1_3"][0], D["cs1_3"][2]])
    m = mutate(ABS, D["cs1_3_triple"], t)
    checks.append(("M3 abstract transpose plant", "must fire",
                   any("abstract clean-S1" in w for w, _ in scan(m))))
    # M2: digit-extension must no longer satisfy containment (first-design S1 now
    # lives in the table, so plant there)
    m = mutate(TAB, D["s1"][2] + "$", D["s1"][2] + "1$")
    checks.append(("M2 substring plant", "must fire", bool(scan(m))))
    # Regression for the 2026-08-21 final-review blind spot: an A3-stale
    # R_tok_clean cell (0.925 -> 0.926) in the table ONLY must now be caught.
    m = mutate(TAB, D["rtc"][0], "0.926")
    checks.append(("A3-stale R_tok_clean table plant", "must fire",
                   any("tab_tokalign R_tok_clean" in w for w, _ in scan(m))))
    # And the new S1_clean column is pinned per-cell.
    m = mutate(TAB, D["cs1_3"][0], D["s1"][0])
    checks.append(("S1_clean table plant", "must fire",
                   any("tab_tokalign S1_clean" in w for w, _ in scan(m))))
    # M4: range endpoint mutated in the abstract (raw tex carries "$--$", so mutate
    # the lo endpoint value itself, boundary-aware, rather than the stripped form)
    lo = D["rtc_range"].split("--")[0]
    m = dict(files)
    m[ABS], nsub = re.subn(r"(?<![\d.])" + re.escape(lo) + r"(?![\d.])",
                           "0.919", m[ABS], count=1)
    assert nsub == 1, f"selftest plant target {lo!r} not present in {ABS}"
    checks.append(("M4 range plant", "must fire",
                   any("range" in w for w, _ in scan(m))))
    # M5: table S1 cell mutated / R_tok cells swapped between rows
    m = mutate(TAB, f"${D['s1'][0]}$", "$0.999$")
    checks.append(("M5 tab S1 plant", "must fire",
                   any("tab_tokalign S1" in w for w, _ in scan(m))))
    m = dict(files)
    m[TAB] = m[TAB].replace(f"${D['rt2'][0]}$", "$SWAP$").replace(
        f"${D['rt2'][1]}$", f"${D['rt2'][0]}$").replace("$SWAP$", f"${D['rt2'][1]}$")
    checks.append(("M5 tab R_tok swap plant", "must fire",
                   any("tab_tokalign R_tok" in w for w, _ in scan(m))))

    # 2026-09-24: A3-stale naive clean ratio in the appendix only must be caught.
    m = mutate("paper/sections/appendix.tex", D["rtcn_triple"],
               D["rtcn_triple"].replace(D["rtcn_triple"].split("/")[0], "0.904", 1))
    checks.append(("A3-stale naive clean ratio plant", "must fire",
                   any("naive" in w for w, _ in scan(m))))
    for name, want, got in checks:
        print(f"  {name:32s} {want:14s} {'OK' if got else 'FAILED'}")
    good = all(got for _, _, got in checks)
    print("selftest:", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())

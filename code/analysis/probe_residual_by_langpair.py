#!/usr/bin/env python3
"""Test the en-anchoring hypothesis on the existing n=206 artifacts (CPU only).

The paper's mechanism claim for the residual reversal: 'en-anchored prompt pairs (every
en-zh/en-ja/en-ko pair reuses the same English vector) plus shared token/numeral/format
structure within a concept family produce concept-aligned logit-lens correlation even in
a model that understands no language but English.'

Test: recompute the depth+family-corrected within-family residual SEPARATELY for
  * en-anchored language pairs: en-zh, en-ja, en-ko
  * non-English pairs:          zh-ja, zh-ko, ja-ko
If the EN-only models' residual is driven by en-anchoring, their en-pair residual should
exceed their non-en-pair residual. Shared-script/numeral structure (e.g. zh-ja kanji
numerals) may still support non-en residual; report both, honestly.

Reads raw_logit_lens from results/probe_attribution_controls_n200/*.json. No GPU.
Writes results/probe_attribution_controls_n200/LANGPAIR_SPLIT.json.
"""
import json, glob, os, sys
import numpy as np

LANGS = ["en", "zh", "ja", "ko"]
EN_PAIRS = [("en", "zh"), ("en", "ja"), ("en", "ko")]
NONEN_PAIRS = [("zh", "ja"), ("zh", "ko"), ("ja", "ko")]
_EN_ONLY_PATTERNS = ("pythia", "gpt2", "gpt-neo", "opt-")
EN_ONLY = lambda m: any(p in m.lower() for p in _EN_ONLY_PATTERNS)


def fast_pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    sx = np.sqrt((x * x).sum()); sy = np.sqrt((y * y).sum())
    return float((x * y).sum() / (sx * sy)) if sx > 1e-9 and sy > 1e-9 else 0.0


def derange(n, rng):
    if n < 2:
        return np.arange(n)
    for _ in range(50):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    return np.roll(np.arange(n), 1)


def derange_wf(fam, rng):
    n = len(fam); perm = np.arange(n)
    for f in np.unique(fam):
        pos = np.where(fam == f)[0]
        if len(pos) >= 2:
            perm[pos] = pos[derange(len(pos), rng)]
    return perm


def main():
    ctrl = sys.argv[1] if len(sys.argv) > 1 else "results/probe_attribution_controls_n200"
    prompts = sys.argv[2] if len(sys.argv) > 2 else "data/prompts_v3_multi.jsonl"
    n_der = int(os.environ.get("N_DER", 200))
    rng = np.random.default_rng(0)

    fam_map = {}
    for line in open(prompts):
        r = json.loads(line)
        fam_map[int(r["pair_id"])] = r.get("family", "na")

    rows = []
    for fp in sorted(glob.glob(f"{ctrl}/*.json")):
        bn = os.path.basename(fp)
        if bn in ("STATS.json", "LANGPAIR_SPLIT.json") or "_smoke" in bn or "pythia-70m" in bn:
            continue
        d = json.load(open(fp))
        raw = d["raw_logit_lens"]; mid = d["mid_layer_band"]
        M = {}
        for pid_s, langd in raw.items():
            pid = int(pid_s)
            if all(lg in langd for lg in LANGS):
                M[pid] = {lg: np.array([langd[lg].get(str(L), 0.0) for L in mid], float)
                          for lg in LANGS}
        pids = sorted(M)
        fam = np.array([fam_map.get(p, "na") for p in pids])

        def resid_for(pairs):
            vals = []
            for (la, lb) in pairs:
                A = np.stack([M[p][la] for p in pids])
                B = np.stack([M[p][lb] for p in pids])
                xs = A.ravel()
                real = fast_pearson(xs, B.ravel())
                mm = np.mean([fast_pearson(xs, B[derange_wf(fam, rng)].ravel())
                              for _ in range(n_der)])
                vals.append(real - mm)
            return float(np.mean(vals))

        r_en = resid_for(EN_PAIRS); r_non = resid_for(NONEN_PAIRS)
        rows.append(dict(model=d["model"], en_only=EN_ONLY(d["model"]),
                         resid_wf_en_pairs=r_en, resid_wf_nonen_pairs=r_non,
                         diff_en_minus_nonen=r_en - r_non))
        print(f"{d['model']:34s} {'EN' if EN_ONLY(d['model']) else 'ML'} "
              f"en-pairs={r_en:+.4f}  non-en={r_non:+.4f}  diff={r_en-r_non:+.4f}", flush=True)

    en = [r for r in rows if r["en_only"]]; ml = [r for r in rows if not r["en_only"]]
    summ = dict(
        n_models=len(rows), n_derangements=n_der,
        mean_en_pairs_ENonly=float(np.mean([r["resid_wf_en_pairs"] for r in en])),
        mean_nonen_pairs_ENonly=float(np.mean([r["resid_wf_nonen_pairs"] for r in en])),
        mean_en_pairs_ML=float(np.mean([r["resid_wf_en_pairs"] for r in ml])),
        mean_nonen_pairs_ML=float(np.mean([r["resid_wf_nonen_pairs"] for r in ml])),
        per_model=rows)
    print("\nEN-only models : en-pairs %.4f vs non-en %.4f" %
          (summ["mean_en_pairs_ENonly"], summ["mean_nonen_pairs_ENonly"]))
    print("Multilingual   : en-pairs %.4f vs non-en %.4f" %
          (summ["mean_en_pairs_ML"], summ["mean_nonen_pairs_ML"]))
    json.dump(summ, open(f"{ctrl}/LANGPAIR_SPLIT.json", "w"), indent=2)
    print(f"[wrote] {ctrl}/LANGPAIR_SPLIT.json")


if __name__ == "__main__":
    main()

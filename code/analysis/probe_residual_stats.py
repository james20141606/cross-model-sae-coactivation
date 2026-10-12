#!/usr/bin/env python3
"""Statistical defensibility of the depth-corrected cross-lingual semantic residual.

Reads the raw logit-lens matrices saved by probe_attribution_runner.py at n>=200
concept pairs (results/probe_attribution_controls_n200/*.json -> field raw_logit_lens)
and computes, with NO further GPU work.

THE RESIDUAL.  residual = real - mismatched-pair, where both are the pooled Pearson r
over (concept x mid-band-layer) observations, averaged over the 6 language pairs (the
SAME construct as the committed n=20 result). mismatched-pair re-aligns lang A's concept
p to a DIFFERENT lang-B concept and so removes whatever the metric scores when concepts
do NOT match -- i.e. the part driven purely by the shared per-layer DEPTH profile.

Two mismatched controls (because the scaled set spans 9 families with DIFFERENT depth
profiles, a purely cross-family mismatch would also destroy family structure and inflate
the residual -- even for an English-only model):
  * global mismatched   : derange concept identity across ALL pairs (depth-corrected only).
  * within-family mism.  : derange concept identity ONLY within the same family, holding
                           depth AND family structure constant. residual_wf isolates
                           concept identity beyond family -> the conservative HEADLINE.

Per model: residual_global and residual_wf, each with a bootstrap 95% CI (1e4 resamples
OVER PROMPT PAIRS) and a one-sided bootstrap p for residual>0.

Multilingual-vs-EN-only separation of residual_wf (the headline):
  (2a) shared-resample bootstrap of GAP = mean_multi - mean_EN (all models share the
       prompt set, so one resample is applied to every model) -> point, 95% CI, p(gap<=0).
  (2b) model-level EXACT permutation (C(n_models,n_EN) assignments) -> HEADLINE p.
  (2c) prompt-pair-level permutation (1e4) on the per-pair within-family residual
       (high power but clustered/anti-conservative; secondary).

EN-only := 'pythia' in model name (Pile, English). Else multilingual.

Usage: probe_residual_stats.py [ctrl_dir] [prompts.jsonl]
"""
import json, glob, os, sys
import numpy as np
from itertools import combinations

LANGS = ["en", "zh", "ja", "ko"]
# English-only (predominantly-English pretraining corpora): Pythia (Pile),
# GPT-2 (WebText), GPT-Neo (Pile), OPT (mostly-English CommonCrawl/Pile mix).
# Everything else in the sweep (Qwen, Gemma, BLOOM, mGPT, Llama-3.2) is
# explicitly multilingual. Substring match on the HF model id.
_EN_ONLY_PATTERNS = ("pythia", "gpt2", "gpt-neo", "opt-")
EN_ONLY = lambda m: any(p in m.lower() for p in _EN_ONLY_PATTERNS)


def fast_pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    sx = np.sqrt((x * x).sum()); sy = np.sqrt((y * y).sum())
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return float((x * y).sum() / (sx * sy))


def load_family_map(prompts_fp):
    fam = {}
    with open(prompts_fp) as f:
        for line in f:
            r = json.loads(line)
            fam[int(r["pair_id"])] = r.get("family", "na")
    return fam


def load_model(fp):
    d = json.load(open(fp))
    raw = d["raw_logit_lens"]; mid_band = d["mid_layer_band"]
    M = {}
    for pid_s, langd in raw.items():
        pid = int(pid_s); row = {}; ok = True
        for lg in LANGS:
            if lg not in langd:
                ok = False; break
            row[lg] = np.array([langd[lg].get(str(L), 0.0) for L in mid_band], float)
        if ok:
            M[pid] = row
    return d["model"], M, len(mid_band)


def build_mats(M, pids):
    out = {}
    for i, la in enumerate(LANGS):
        for lb in LANGS[i + 1:]:
            A = np.stack([M[p][la] for p in pids])
            B = np.stack([M[p][lb] for p in pids])
            out[(la, lb)] = (A, B)
    return out


def derange_global(n, rng):
    if n < 2:
        return np.arange(n)
    for _ in range(50):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    return np.roll(np.arange(n), 1)


def derange_within_family(fam_arr, rng):
    """Permutation that maps each position to another position of the SAME family."""
    n = len(fam_arr); perm = np.arange(n)
    for fam in np.unique(fam_arr):
        pos = np.where(fam_arr == fam)[0]
        if len(pos) < 2:
            continue
        perm[pos] = pos[derange_global(len(pos), rng)]
    return perm


def pooled_metrics(mats, idx, fam_arr, n_der, rng):
    """mean-over-langpairs (real, mm_global, mm_wf) on resampled rows `idx`."""
    fam_sel = fam_arr[idx]
    reals, mmg, mmw = [], [], []
    for (A, B) in mats.values():
        a = A[idx]; b = B[idx]; xs = a.ravel()
        reals.append(fast_pearson(xs, b.ravel()))
        g, w = [], []
        for _ in range(n_der):
            g.append(fast_pearson(xs, b[derange_global(len(idx), rng)].ravel()))
            w.append(fast_pearson(xs, b[derange_within_family(fam_sel, rng)].ravel()))
        mmg.append(np.mean(g)); mmw.append(np.mean(w))
    return float(np.mean(reals)), float(np.mean(mmg)), float(np.mean(mmw))


def per_pair_residual(mats, fam_arr):
    """per (langpair, concept) within-family residual:
       corr(A_p,B_p) - mean_{q in same family, q!=p} corr(A_p,B_q). Pooled over langpairs."""
    res = []
    for (A, B) in mats.values():
        Az = A - A.mean(1, keepdims=True); Bz = B - B.mean(1, keepdims=True)
        an = np.sqrt((Az * Az).sum(1)); bn = np.sqrt((Bz * Bz).sum(1))
        good = (an > 1e-9) & (bn > 1e-9)
        Az = Az / np.where(an[:, None] > 1e-9, an[:, None], 1.0)
        Bz = Bz / np.where(bn[:, None] > 1e-9, bn[:, None], 1.0)
        C = Az @ Bz.T; n = C.shape[0]
        aligned = np.diag(C).copy()
        rp = np.full(n, np.nan)
        for p in range(n):
            same = np.where(fam_arr == fam_arr[p])[0]
            same = same[same != p]
            if len(same) == 0:
                continue
            rp[p] = aligned[p] - C[p, same].mean()
        m = good & ~np.isnan(rp)
        res.append(rp[m])
    return np.concatenate(res) if res else np.array([])


def main():
    ctrl = sys.argv[1] if len(sys.argv) > 1 else "results/probe_attribution_controls_n200"
    prompts = sys.argv[2] if len(sys.argv) > 2 else "data/prompts_v3_multi.jsonl"
    n_boot = int(os.environ.get("N_BOOT", 10000))
    n_perm = int(os.environ.get("N_PERM", 10000))
    seed = 0

    fam_map = load_family_map(prompts)
    # STATS.json is this script's own output; LANGPAIR_SPLIT.json is written later by
    # probe_residual_by_langpair.py into the same directory. Neither carries
    # raw_logit_lens, so both must be excluded or the glob crashes on re-run.
    files = sorted(f for f in glob.glob(f"{ctrl}/*.json")
                   if "_smoke" not in f
                   and os.path.basename(f) not in ("STATS.json", "LANGPAIR_SPLIT.json")
                   and "pythia-70m" not in f)  # 70m = smoke only
    models = {}
    for fp in files:
        name, M, B = load_model(fp)
        models[name] = M
        print(f"[load] {name:34s} pairs={len(M)} midband={B}", flush=True)

    common = sorted(set.intersection(*[set(M.keys()) for M in models.values()]))
    n_pairs = len(common)
    fam_arr = np.array([fam_map.get(p, "na") for p in common])
    print(f"[common] {n_pairs} concept pairs x {len(models)} models; "
          f"families={len(np.unique(fam_arr))}\n", flush=True)

    mats = {name: build_mats(M, common) for name, M in models.items()}
    rng = np.random.default_rng(seed)
    arange = np.arange(n_pairs)

    rows = []
    for name in models:
        real_pt, mmg_pt, mmw_pt = pooled_metrics(mats[name], arange, fam_arr, 200, rng)
        rg_pt, rw_pt = real_pt - mmg_pt, real_pt - mmw_pt
        bg = np.empty(n_boot); bw = np.empty(n_boot)
        for t in range(n_boot):
            idx = rng.integers(0, n_pairs, n_pairs)
            r, mg, mw = pooled_metrics(mats[name], idx, fam_arr, 2, rng)
            bg[t] = r - mg; bw[t] = r - mw
        glo, ghi = np.percentile(bg, [2.5, 97.5])
        wlo, whi = np.percentile(bw, [2.5, 97.5])
        pp = per_pair_residual(mats[name], fam_arr)
        rows.append(dict(
            model=name, multilingual=not EN_ONLY(name),
            real=real_pt, mm_global=mmg_pt, mm_within_family=mmw_pt,
            residual_global=rg_pt, residual_wf=rw_pt,
            rg_ci=[float(glo), float(ghi)], rw_ci=[float(wlo), float(whi)],
            rw_p_le0=float(np.mean(bw <= 0)), rg_p_le0=float(np.mean(bg <= 0)),
            rw_boot_se=float(bw.std()),
            per_pair_wf_mean=float(pp.mean()), per_pair_n=int(pp.size)))
        print(f"[boot] {name:30s} real={real_pt:+.3f} mmWF={mmw_pt:+.3f} "
              f"residWF={rw_pt:+.4f} CI[{wlo:+.4f},{whi:+.4f}]  "
              f"(global resid={rg_pt:+.4f})", flush=True)

    rows.sort(key=lambda r: -r["residual_wf"])
    multi = [r for r in rows if r["multilingual"]]
    mono = [r for r in rows if not r["multilingual"]]
    nm = [r["model"] for r in multi]; ne = [r["model"] for r in mono]

    # (2a) shared-resample bootstrap of the GAP on residual_wf -----------------
    gap_pt = np.mean([r["residual_wf"] for r in multi]) - np.mean([r["residual_wf"] for r in mono])
    gb = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.integers(0, n_pairs, n_pairs)
        rv = {}
        for name in models:
            r, mg, mw = pooled_metrics(mats[name], idx, fam_arr, 2, rng)
            rv[name] = r - mw
        gb[t] = np.mean([rv[n] for n in nm]) - np.mean([rv[n] for n in ne])
    glo, ghi = np.percentile(gb, [2.5, 97.5]); gap_p = float(np.mean(gb <= 0))

    # (2b) model-level EXACT permutation on residual_wf -----------------------
    names = [r["model"] for r in rows]
    vals = np.array([r["residual_wf"] for r in rows])
    n_en = len(mono); obs = gap_pt
    combos = list(combinations(range(len(names)), n_en))
    perm_gaps = np.array([(vals[[i for i in range(len(names)) if i not in c]].mean()
                           - vals[list(c)].mean()) for c in combos])
    ge = int((perm_gaps >= obs - 1e-12).sum())
    perm_p_model = ge / len(combos)
    # The one-sided test above asks only about a POSITIVE multilingual advantage (the
    # pre-registered direction). Because the observed gap is negative, that p-value alone
    # cannot support a claim of "no gap in EITHER direction"; report the two-sided test too.
    perm_p_model_two_sided = float((np.abs(perm_gaps) >= abs(obs) - 1e-12).sum() / len(combos))

    # (2c) pair-level permutation on per-pair within-family residual ----------
    ppm = np.concatenate([per_pair_residual(mats[n], fam_arr) for n in nm])
    ppe = np.concatenate([per_pair_residual(mats[n], fam_arr) for n in ne])
    obs_pair = ppm.mean() - ppe.mean()
    pool = np.concatenate([ppm, ppe]); k = ppm.size; ge2 = 0; ge2_abs = 0
    for _ in range(n_perm):
        pr = rng.permutation(pool.size)
        g = pool[pr[:k]].mean() - pool[pr[k:]].mean()
        if g >= obs_pair - 1e-15:
            ge2 += 1
        if abs(g) >= abs(obs_pair) - 1e-15:
            ge2_abs += 1
    perm_p_pair = (ge2 + 1) / (n_perm + 1)
    perm_p_pair_two_sided = (ge2_abs + 1) / (n_perm + 1)

    # ---- report --------------------------------------------------------------
    print(f"\n========= DEPTH+FAMILY-CORRECTED RESIDUAL @ n={n_pairs} =========")
    print(f"{'model':30s} {'real':>6s} {'mmWF':>6s} {'residWF':>8s} {'95% CI (within-family)':>22s} {'p<=0':>6s} type")
    for r in rows:
        sig = "*" if r["rw_ci"][0] > 0 else (" " if r["rw_ci"][1] < 0 else ".")
        print(f"{r['model']:30s} {r['real']:+6.3f} {r['mm_within_family']:+6.3f} "
              f"{r['residual_wf']:+8.4f} [{r['rw_ci'][0]:+.4f},{r['rw_ci'][1]:+.4f}]{sig} "
              f"{r['rw_p_le0']:6.4f} {'multi' if r['multilingual'] else 'EN-only'}")
    print("  (* = within-family bootstrap 95% CI excludes 0; for reference, global residual:)")
    for r in rows:
        sig = "*" if r["rg_ci"][0] > 0 else " "
        print(f"      {r['model']:26s} residGLOBAL={r['residual_global']:+.4f} "
              f"CI[{r['rg_ci'][0]:+.4f},{r['rg_ci'][1]:+.4f}]{sig}")

    print("\n---- multilingual vs EN-only (on within-family residual) ----")
    print(f"  mean residWF multilingual = {np.mean([r['residual_wf'] for r in multi]):+.4f}  ({', '.join(nm)})")
    print(f"  mean residWF EN-only      = {np.mean([r['residual_wf'] for r in mono]):+.4f}  ({', '.join(ne)})")
    print(f"  GAP = {gap_pt:+.4f}")
    print(f"  (2a) shared-resample bootstrap gap 95% CI=[{glo:+.4f},{ghi:+.4f}], p(gap<=0)={gap_p:.4f}")
    print(f"  (2b) model-level EXACT permutation p ({len(combos)} assignments) = {perm_p_model:.4f}  [HEADLINE, one-sided]")
    print(f"       model-level EXACT two-sided p = {perm_p_model_two_sided:.4f}")
    print(f"  (2c) pair-level permutation p ({n_perm}) = {perm_p_pair:.4f}  (clustered/anti-conservative)")
    print(f"       pair-level two-sided p = {perm_p_pair_two_sided:.4f}")

    out = dict(n_pairs=n_pairs, n_models=len(models), n_boot=n_boot, n_perm=n_perm, seed=seed,
               families=sorted(map(str, np.unique(fam_arr))), per_model=rows,
               gap_wf_point=float(gap_pt), gap_wf_ci=[float(glo), float(ghi)], gap_wf_boot_p=gap_p,
               perm_p_model_level=perm_p_model, n_model_perms=len(combos),
               perm_p_model_level_two_sided=perm_p_model_two_sided,
               perm_p_pair_level=perm_p_pair, obs_pair_gap=float(obs_pair),
               perm_p_pair_level_two_sided=perm_p_pair_two_sided,
               mean_resid_wf_multi=float(np.mean([r['residual_wf'] for r in multi])),
               mean_resid_wf_en=float(np.mean([r['residual_wf'] for r in mono])),
               residual_wf_sig_per_model={r['model']: bool(r['rw_ci'][0] > 0) for r in rows})
    json.dump(out, open(f"{ctrl}/STATS.json", "w"), indent=2)
    print(f"\nwrote {ctrl}/STATS.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""F1 diagnosis: why the nine-cell bootstrap '95%' intervals exclude their own point estimates.

Registered in notes/PREREG_F1_bootstrap_diagnosis_20261004.md (2ec8d51, before this file).
Receipt-only, CPU, 0 GPU. Independent of code/analysis/: nothing is imported from the repo;
quantiles are a hand-written type-7 interpolation on sorted Python lists.

    /usr/bin/python3 results/f1_bootstrap_diag_20261004/f1_bootstrap_diag.py   # writes F1_BOOTSTRAP_DIAG.json

D1 percentile check, D2 bias, D3 nested Monte-Carlo noise, D4 estimand drift, D5 planted coverage
(synthetic only), then the C0/C1/C2 reading fixed in the prereg.
"""
import hashlib, json, math, os, shutil, statistics, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC_COMMIT = "14f05d3"
TORCH = "results/ninecell_20260926/cluster_18577359"
LOCAL = "results/ninecell_20260926/local_mps_cache"
CELLS = ["qwen3-1.7b__gemma2-2b", "qwen3-1.7b__gemma2-9b", "qwen3-1.7b__mistral7b",
         "qwen3-8b__gemma2-2b", "qwen3-8b__gemma2-9b", "qwen3-8b__mistral7b",
         "gemma2-2b__mistral7b", "gemma2-9b__mistral7b", "llama3.2-1b__mistral7b"]
R_MIN = 0.50


def q7(xs, p):
    """Type-7 quantile (R default; numpy 'linear'), written out by hand."""
    s = sorted(xs); n = len(s)
    h = (n - 1) * p / 100.0
    lo = math.floor(h); hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def sd(xs):
    return statistics.pstdev(xs)


def copy_receipts(tmp):
    """Copy every receipt used, and assert each copy is byte-identical to the committed blob."""
    shas = {}
    for sub, ref in ((TORCH, SRC_COMMIT), (LOCAL, "HEAD")):
        for kind in ("A", "B"):
            for c in CELLS:
                rel = f"{sub}/{kind}_{c}.json"
                src = REPO / rel
                if not src.exists():
                    continue
                dst = Path(tmp) / rel.replace("/", "__")
                shutil.copy(src, dst)
                blob = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{rel}"],
                                      capture_output=True, check=True).stdout
                h_copy = hashlib.sha256(dst.read_bytes()).hexdigest()
                assert h_copy == hashlib.sha256(blob).hexdigest(), rel
                shas[rel] = h_copy
    return shas


def interval_row(name, draws, point, printed):
    lo, hi = q7(draws, 2.5), q7(draws, 97.5)
    med = q7(draws, 50); s = sd(draws)
    frac_ge = sum(1 for d in draws if d >= point) / len(draws)
    shift = med - point
    return dict(
        name=name, point=point, n_draws=len(draws), printed=printed,
        recomputed=[lo, hi],
        D1_max_abs_diff=(max(abs(lo - printed[0]), abs(hi - printed[1])) if printed else None),
        contains_point=bool(lo <= point <= hi),
        boot_sd=s, median=med, shift=shift, shift_in_sd=(shift / s if s > 0 else None),
        frac_draws_ge_point=frac_ge,
        biased=bool((s > 0 and abs(shift) > s) or not (0.10 <= frac_ge <= 0.90)),
        basic=[2 * point - hi, 2 * point - lo],
        recentred=[lo - shift, hi - shift],
    )


def analyse_A(path):
    a = json.load(open(path))
    pt, bs = a["point"], a["bootstrap"]
    d = bs["draws"]
    rows = []
    keys = [k for k in pt if k != "base"]
    for k in keys:
        draws = [x[k]["R"] for x in d if x[k]["R"] is not None]
        if not draws:                     # local fallback ran n_boot = 0 (prereg deviation 1)
            continue
        r = interval_row(f"R_{k}", draws, pt[k]["R"], bs["ci95"].get(f"R_{k}"))
        # D3: nested Monte-Carlo noise inside one resample (point-estimate scale used as proxy)
        nsd = pt[k]["null_sd"]; ex = pt[k]["excess"]; cal = pt[k]["calibration"]
        part_var = max(cal["excess_sd"] ** 2 - nsd ** 2 / cal["n_perm_each"], 0.0)
        var_ex = nsd ** 2 / bs["n_perm"]
        var_cal = (part_var + nsd ** 2 / bs["n_perm_cal"]) / bs["n_cal"]
        r["D3_mc_sd"] = abs(pt[k]["R"]) * math.sqrt(var_ex / ex ** 2 + var_cal / cal["excess"] ** 2)
        r["D3_mc_over_boot_sd"] = r["D3_mc_sd"] / r["boot_sd"] if r["boot_sd"] > 0 else None
        r["D3_material"] = bool(r["D3_mc_over_boot_sd"] is not None and r["D3_mc_over_boot_sd"] >= 0.5)
        # decomposition: where does the resampled world move?
        r["boot_mean_excess_over_point"] = statistics.mean(x[k]["excess"] for x in d) / ex
        r["boot_mean_cal_over_point"] = statistics.mean(x[k]["cal"] for x in d) / cal["excess"]
        rows.append(r)
    base_pt = pt["base"]["excess"]
    if not d:
        return dict(rows=rows, D4=None, n_boot=0)
    frac_draws = [x["base"] / a["topk"] for x in d]
    rows.append(interval_row("excess_frac_base", frac_draws, base_pt / a["topk"],
                             bs["ci95"].get("excess_frac_base")))
    D4 = dict(obs_M=a["obs_M"], boot_mean_M=statistics.mean(x["M"] for x in d),
              M_ratio=statistics.mean(x["M"] for x in d) / a["obs_M"],
              base_excess_point=base_pt, boot_mean_base_excess=statistics.mean(x["base"] for x in d),
              base_excess_ratio=statistics.mean(x["base"] for x in d) / base_pt)
    D4["drift_gt_5pct"] = bool(abs(D4["base_excess_ratio"] - 1) > 0.05)
    blocks = {k: (min(v), statistics.median(v), max(v)) for k, v in a["block_sizes"].items()}
    return dict(rows=rows, D4=D4, block_sizes_min_med_max=blocks, n_boot=bs["n_boot"])


def analyse_B(path):
    b = json.load(open(path))
    eb = b["deltaE_bootstrap"]
    r = interval_row("deltaE", eb["draws"], eb["point"], eb["ci95"])
    nsd = b["consecutive"]["held_out"]["null_sd"]
    r["D3_mc_sd"] = nsd / math.sqrt(eb["n_perm_each"])
    r["D3_mc_over_boot_sd"] = r["D3_mc_sd"] / r["boot_sd"]
    r["D3_material"] = bool(r["D3_mc_over_boot_sd"] >= 0.5)
    rs = b["random_splits"]
    split = dict(n_splits=len(rs["splits"]), min_excess=rs["min_excess"], max_excess=rs["max_excess"],
                 max_p=rs["max_p"], rho=rs["rho"], all_p_floor=all(abs(x["p"] - 1 / 2001) < 1e-12 for x in rs["splits"]),
                 consecutive=b["consecutive"]["held_out"]["excess"],
                 n_pairs=b["consecutive"]["n_pairs_selected"])
    return dict(rows=[r], random_split=split)


# ---------------------------------------------------------------- D5 planted coverage (synthetic)
def planted(n_data=200, n_boot=200, n=1000, P=100, seed=20261004):
    import numpy as np
    rng = np.random.default_rng(seed)

    def gen(kind, m, g):
        if kind == "gaussian":
            L = g.standard_normal((m, P))
            return L + 0.8 * g.standard_normal((m, P)), L + 0.8 * g.standard_normal((m, P))
        q = {"sparse_q0.10": 0.10, "sparse_q0.02": 0.02}[kind]
        act = g.random((m, P)) < q
        mag = g.exponential(1.0, (m, P)) ** 2            # heavy right tail
        xa = act * mag * np.clip(1 + 0.5 * g.standard_normal((m, P)), 0, None)
        xb = act * mag * np.clip(1 + 0.5 * g.standard_normal((m, P)), 0, None)
        xa = xa + (g.random((m, P)) < q / 2) * g.exponential(1.0, (m, P)) ** 2
        xb = xb + (g.random((m, P)) < q / 2) * g.exponential(1.0, (m, P)) ** 2
        return xa, xb

    def z(X):
        mu = X.mean(0); s = X.std(0); ok = s > 1e-12
        Z = np.zeros_like(X); Z[:, ok] = (X[:, ok] - mu[ok]) / s[ok]
        return Z

    def stat(Xa, Xb, g, n_perm=20):
        Za, Zb = z(Xa), z(Xb); m = Za.shape[0]
        st = float((Za * Zb).sum(0).mean() / m)
        nl = [float((Za * Zb[g.permutation(m)]).sum(0).mean() / m) for _ in range(n_perm)]
        return st - float(np.mean(nl))

    out = {}
    for kind in ("gaussian", "sparse_q0.10", "sparse_q0.02"):
        big_a, big_b = gen(kind, 400_000, np.random.default_rng(seed + 1))
        truth = float(np.mean([np.corrcoef(big_a[:, j], big_b[:, j])[0, 1] for j in range(P)]))
        cover = 0; excl_point = 0; shifts = []; widths = []; pts = []
        for i in range(n_data):
            g = np.random.default_rng(seed + 1000 * (i + 1))
            Xa, Xb = gen(kind, n, g)
            pt = stat(Xa, Xb, g)
            draws = []
            for bb in range(n_boot):
                rr = g.choice(n, n, replace=True)
                draws.append(stat(Xa[rr], Xb[rr], g))
            lo, hi = q7(draws, 2.5), q7(draws, 97.5)
            cover += lo <= truth <= hi; excl_point += not (lo <= pt <= hi)
            shifts.append(q7(draws, 50) - pt); widths.append(hi - lo); pts.append(pt)
        out[kind] = dict(truth=truth, mean_point=float(np.mean(pts)), point_bias=float(np.mean(pts) - truth),
                         coverage=cover / n_data, frac_intervals_excluding_point=excl_point / n_data,
                         median_shift=float(np.median(shifts)), median_width=float(np.median(widths)),
                         n_data=n_data, n_boot=n_boot, n=n, P=P)
        print(f"[D5] {kind}: truth {truth:.4f} coverage {cover / n_data:.3f} "
              f"excl_point {excl_point / n_data:.3f} median shift {np.median(shifts):+.4f} "
              f"median width {np.median(widths):.4f}", flush=True)
    return out


def main():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        shas = copy_receipts(tmp)
        res = {}
        for plat, sub in (("cluster", TORCH), ("local", LOCAL)):
            res[plat] = {}
            for c in CELLS:
                fa = Path(tmp) / f"{sub}/A_{c}.json".replace("/", "__")
                fb = Path(tmp) / f"{sub}/B_{c}.json".replace("/", "__")
                if not fa.exists():
                    continue
                A = analyse_A(fa); B = analyse_B(fb)
                res[plat][c] = dict(A=A, B=B)
    D5 = planted()
    # ------------------------------------------------------------- verdict (cluster run governs)
    tr = res["cluster"]
    verdict_names = {"R_len", "R_top10", "R_top200", "R_top500", "deltaE"}
    allrows = [(c, r) for c, v in tr.items() for r in v["A"]["rows"] + v["B"]["rows"]]
    vrows = [(c, r) for c, r in allrows if r["name"] in verdict_names]
    d1_fail = [f"{c}:{r['name']}" for c, r in allrows if r["printed"] and r["D1_max_abs_diff"] > 1e-9]
    biased = [f"{c}:{r['name']}" for c, r in vrows if r["biased"]]
    excl = [f"{c}:{r['name']}" for c, r in allrows if not r["contains_point"]]
    mc_material = [f"{c}:{r['name']}" for c, r in vrows if r["D3_material"]]
    drift = {c: round(v["A"]["D4"]["base_excess_ratio"], 4) for c, v in tr.items()}
    sparse_bad = [k for k, v in D5.items() if k.startswith("sparse") and v["coverage"] < 0.90]
    instrument_failed = bool(d1_fail or biased or sparse_bad)

    def thr(kind):
        lows = []
        for c, r in vrows:
            lo = r[kind][0] if kind != "percentile" else r["recomputed"][0]
            ok = lo > 0 if r["name"] == "deltaE" else lo >= R_MIN
            lows.append((c, r["name"], lo, ok))
        return dict(all_pass=all(x[3] for x in lows),
                    min_R_lower=min(x[2] for x in lows if x[1] != "deltaE"),
                    min_deltaE_lower=min(x[2] for x in lows if x[1] == "deltaE"))

    rs = [v["B"]["random_split"] for v in tr.values()]
    out = dict(
        prereg="notes/PREREG_F1_bootstrap_diagnosis_20261004.md (2ec8d51)",
        source_commit=SRC_COMMIT, receipt_sha256=shas, python=sys.version.split()[0],
        cells=res, D5_planted=D5,
        summary=dict(
            D1_percentile_bug=bool(d1_fail), D1_failures=d1_fail,
            intervals_excluding_point=excl, n_intervals=len(allrows),
            D2_biased_verdict_intervals=biased, n_verdict_intervals=len(vrows),
            D3_nested_noise_material=mc_material,
            D4_base_excess_ratio=drift,
            D5_sparse_coverage_below_0p90=sparse_bad,
            QC_reading=("C0 - instrument failed, no interval reported" if instrument_failed else "evaluate C1/C2"),
            threshold_comparison_descriptive={k: thr(k) for k in ("percentile", "basic", "recentred")},
            random_splits=dict(n_tests=sum(x["n_splits"] for x in rs),
                               all_p_1_over_2001=all(x["all_p_floor"] for x in rs),
                               rho_range=[min(x["rho"] for x in rs), max(x["rho"] for x in rs)],
                               min_excess=min(x["min_excess"] for x in rs)),
        ),
        wall_s=time.time() - t0, date=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    (HERE / "F1_BOOTSTRAP_DIAG.json").write_text(json.dumps(out, indent=1))
    s = out["summary"]
    print(json.dumps({k: s[k] for k in s if k != "intervals_excluding_point"}, indent=1))
    print("intervals excluding point:", len(s["intervals_excluding_point"]), "/", s["n_intervals"])
    for c, v in tr.items():
        for r in v["A"]["rows"] + v["B"]["rows"]:
            print(f"{c:24s} {r['name']:17s} pt {r['point']:.3f} pct [{r['recomputed'][0]:.3f}, {r['recomputed'][1]:.3f}] "
                  f"shift {r['shift']:+.4f} ({r['shift_in_sd']:+.2f} sd) ge {r['frac_draws_ge_point']:.3f} "
                  f"mc/boot {r.get('D3_mc_over_boot_sd') or float('nan'):.2f} {'BIASED' if r['biased'] else ''}")
        print(f"{'':24s} D4 M {v['A']['D4']['M_ratio']:.3f}  base excess {v['A']['D4']['base_excess_ratio']:.3f}")


if __name__ == "__main__":
    main()

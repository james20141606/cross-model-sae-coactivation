#!/usr/bin/env python3
"""Nine-cell length/topic nulls, nine-cell random split, document-bootstrap intervals.

Registered in notes/PREREG_ninecell_ladder_20260926.md (commit 3a0b358, before this file).

Subcommands
  extract --model TAG --pooled DIR [--max_docs n]
        GPU. cross_model_sae_match.compute_pooled_features, unchanged (published read points,
        Gemma offset 0), on the 2,000-doc C4 slice -> DIR/TAG.npy (+ .meta.json, sha256).
  analyze_a --cell TAG --pooled DIR --out JSON [--device cuda|cpu|numpy] [--n_boot 200]
        Q-A point estimates (base / length-decile / topic-10 nulls + size-matched calibrations,
        n_perm 2000 / 20x200), G0a, and Q-C bootstrap intervals for R_len, R_top10, excess
        (+ R_k200 / R_k500 on the three ladder cells). Permutation counts on GPU in float64.
  analyze_b --cell TAG --pooled DIR --out JSON
        CPU numpy. G0b (consecutive split reproduces heldout_<cell>.json), Q-B ten seeded random
        splits through cross_model_sae_match.heldout_frozen_pairs (verbatim), Q-C Delta_E bootstrap.
  summarize --dir DIR
        Reads A_*.json / B_*.json, applies the registered decision rules, writes
        NINECELL_SUMMARY.json and tab_ninecell.tex (every number formatted from the receipts).
  selftest
        numpy vs torch count identity; planted length confound is removed by the length null and
        not by its calibration; planted length-free sharing survives; random split held-out on
        planted vs noise; decision-rule branches on synthetic inputs.
"""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from cross_model_sae_match import _zscore_cols, heldout_frozen_pairs, topk_features  # noqa: E402
from c4_covariate_null import size_matched_random_strata, strata_perm  # noqa: E402

C4_MD5 = "30fe9a250f82eb2bcc94ff2282e1407f"
MODELS = {
    "gemma2-2b": ("google/gemma-2-2b", "gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical"),
    "gemma2-9b": ("google/gemma-2-9b", "gemma-scope-9b-pt-res-canonical", "layer_20/width_16k/canonical"),
    "mistral7b": ("mistralai/Mistral-7B-v0.1", "mistral-7b-res-wg", "blocks.16.hook_resid_pre"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B-Base", "qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", "layer14"),
    "qwen3-8b": ("Qwen/Qwen3-8B-Base", "qwenpt:Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", "layer18"),
    "llama3.2-1b": ("meta-llama/Llama-3.2-1B", "chanind/sae-llama-3.2-1b-res", "blocks.8.hook_resid_post"),
}
CELLS = ["qwen3-1.7b__gemma2-2b", "qwen3-1.7b__gemma2-9b", "qwen3-1.7b__mistral7b",
         "qwen3-8b__gemma2-2b", "qwen3-8b__gemma2-9b", "qwen3-8b__mistral7b",
         "gemma2-2b__mistral7b", "gemma2-9b__mistral7b", "llama3.2-1b__mistral7b"]
LADDER = ["qwen3-1.7b__gemma2-2b", "gemma2-9b__mistral7b", "llama3.2-1b__mistral7b"]
TOPK = 1500
N_PERM = 2000
N_CAL, N_PERM_CAL = 20, 200
SPLIT_SEEDS = [20260926 + s for s in range(10)]
ALPHA_A = 0.05 / 18
ALPHA_B = 0.05 / 9
R_MIN = 0.50


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_rows(c4):
    assert hashlib.md5(open(c4, "rb").read()).hexdigest() == C4_MD5, f"C4 slice md5 mismatch: {c4}"
    prompts = [json.loads(l) for l in open(c4)]
    sel = [p for p in prompts if p.get("lang") == "en"]
    sel.sort(key=lambda p: p.get("pair_id", 0))
    return prompts, [p["prompt"] for p in sel]


# ------------------------------------------------------------------------------------ extract
def cmd_extract(a):
    import torch, transformers
    from cross_model_sae_match import compute_pooled_features
    prompts, rows = load_rows(a.c4)
    hf, rel, sid = MODELS[a.model]
    t0 = time.time()
    P = compute_pooled_features(hf, rel, sid, prompts, lang="en", max_prompts=a.max_docs)
    wall = time.time() - t0
    P = np.ascontiguousarray(P, dtype=np.float32)
    n = len(rows) if a.max_docs is None else min(a.max_docs, len(rows))
    assert P.shape[0] == n, (P.shape, n)
    out = Path(a.pooled); out.mkdir(parents=True, exist_ok=True)
    p = out / f"{a.model}.npy"
    np.save(p, P)
    meta = dict(model=a.model, hf_id=hf, sae_release=rel, sae_id=sid, shape=list(P.shape),
                pooled_file=str(p), pooled_sha256=sha256_file(p), corpus=a.c4, corpus_md5=C4_MD5,
                wall_s=wall, host=platform.node(), torch=torch.__version__,
                transformers=transformers.__version__,
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                script_sha256=sha256_file(__file__),
                matcher_sha256=sha256_file(HERE / "cross_model_sae_match.py"),
                date=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(meta, open(out / f"{a.model}.meta.json", "w"), indent=1)
    print(f"[saved] {p} {P.shape} {wall:.0f}s", flush=True)


# ----------------------------------------------------------------------------- count engines
class Engine:
    """Reciprocal-argmax count M and greedy mean for many row permutations of Zb.
    backend 'numpy' is c4_covariate_null._mutual_and_greedy in a loop; 'torch' batches the same
    arithmetic in float64 (cuda or cpu). selftest asserts identical counts."""

    def __init__(self, Za, Zb, backend="numpy", batch=50):
        assert Za.shape[0] == Zb.shape[0]
        self.N = Za.shape[0]; self.backend = backend; self.batch = batch
        if backend == "numpy":
            self.Za, self.Zb = Za, Zb
        else:
            import torch
            self.t = torch
            self.dev = torch.device(backend)
            self.ZaT = torch.as_tensor(np.ascontiguousarray(Za.T), dtype=torch.float64, device=self.dev)
            self.Zb = torch.as_tensor(Zb, dtype=torch.float64, device=self.dev)
            self.ar = torch.arange(Za.shape[1], device=self.dev)

    def counts(self, perms):
        perms = np.asarray(perms)
        if self.backend == "numpy":
            m = np.empty(len(perms), np.int64); g = np.empty(len(perms))
            for i, pm in enumerate(perms):
                C = (self.Za.T @ self.Zb[pm]) / self.N
                a2b = C.argmax(1); b2a = C.argmax(0)
                m[i] = int((b2a[a2b] == np.arange(C.shape[0])).sum()); g[i] = float(C.max(1).mean())
            return m, g
        t = self.t; ms, gs = [], []
        for s in range(0, len(perms), self.batch):
            idx = t.as_tensor(perms[s:s + self.batch], device=self.dev)
            C = t.matmul(self.ZaT.unsqueeze(0), self.Zb[idx]) / self.N      # [B, Ka, Kb]
            a2b = C.argmax(2); b2a = C.argmax(1)
            ms.append((t.gather(b2a, 1, a2b) == self.ar).sum(1).cpu().numpy())
            gs.append(C.max(2).values.mean(1).cpu().numpy())
            del C
        return np.concatenate(ms).astype(np.int64), np.concatenate(gs)


def draw_perms(N, strata, n_perm, seed):
    """Same RNG stream as c4_covariate_null.null_under: one rng, one permutation per draw."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(N) if strata is None else strata_perm(rng, strata)
                     for _ in range(n_perm)])


def arm(eng, obs_m, strata, n_perm, seed):
    m, _ = eng.counts(draw_perms(eng.N, strata, n_perm, seed))
    sd = float(m.std())
    return dict(null_mean=float(m.mean()), null_sd=sd, excess=float(obs_m - m.mean()),
                z=float((obs_m - m.mean()) / sd) if sd > 1e-12 else float("inf"),
                p=float((np.sum(m >= obs_m) + 1) / (n_perm + 1)), n_perm=int(n_perm), seed=int(seed))


def cal_arm(eng, obs_m, strata, n_rep, n_perm, part_seed0, perm_seed0):
    reps = [arm(eng, obs_m, size_matched_random_strata(strata, part_seed0 + 97 * r), n_perm, perm_seed0 + r)
            for r in range(n_rep)]
    ex = np.array([x["excess"] for x in reps])
    return dict(excess=float(ex.mean()), excess_sd=float(ex.std(ddof=1)) if n_rep > 1 else 0.0,
                excess_min=float(ex.min()), excess_max=float(ex.max()), n_rep=int(n_rep),
                n_perm_each=int(n_perm), part_seed0=int(part_seed0), perm_seed0=int(perm_seed0),
                reps=[{k: x[k] for k in ("excess", "null_mean", "p")} for x in reps])


def prep(Pa, Pb):
    Sa, ia = topk_features(Pa, TOPK); Sb, ib = topk_features(Pb, TOPK)
    Za, _ = _zscore_cols(Sa); Zb, _ = _zscore_cols(Sb)
    C = (Za.T @ Zb) / Za.shape[0]
    a2b = C.argmax(1); b2a = C.argmax(0)
    obs = int((b2a[a2b] == np.arange(C.shape[0])).sum())
    return Za, Zb, obs, ia, ib


def ladder_arms(eng, obs, strata, n_perm, n_cal, n_perm_cal, seeds):
    """seeds: (perm seed, calibration partition seed0, calibration perm seed0)."""
    x = arm(eng, obs, strata, n_perm, seeds[0])
    c = cal_arm(eng, obs, strata, n_cal, n_perm_cal, seeds[1], seeds[2])
    x["calibration"] = c
    x["R"] = float(x["excess"] / c["excess"]) if c["excess"] != 0 else None
    return x


def pct(a, q):
    a = np.asarray([v for v in a if v is not None and np.isfinite(v)], float)
    return [float(np.percentile(a, q[0])), float(np.percentile(a, q[1]))] if len(a) else None


def cmd_analyze_a(a):
    # Added 2026-09-26 17:50 while JID 18577359 ran: extra runners (srun --overlap) take cells from
    # the end of the list to raise GPU utilisation; a cell whose complete receipt already exists is
    # skipped. No statistic changed (see ELN).
    if Path(a.out).exists():
        try:
            if "bootstrap" in json.load(open(a.out)):
                print(f"[A:{a.cell}] receipt exists, skipped -> {a.out}", flush=True)
                return
        except Exception:
            pass
    A, B = a.cell.split("__")
    _, rows = load_rows(a.c4)
    Pa = np.load(Path(a.pooled) / f"{A}.npy"); Pb = np.load(Path(a.pooled) / f"{B}.npy")
    N = min(Pa.shape[0], Pb.shape[0], len(rows)); Pa, Pb, rows = Pa[:N], Pb[:N], rows[:N]
    ncl = [10] + ([200, 500] if (a.cell in LADDER and N >= 1000) else [])
    from c4_covariate_null import build_strata          # sklearn TF-IDF/KMeans, verbatim
    strata = {}
    for k in ncl:
        ls, bs, wc = build_strata(rows, 10, k, 0)
        strata["len"] = ls; strata[f"top{k}"] = bs
    t0 = time.time()
    Za, Zb, obs, ia, ib = prep(Pa, Pb)
    backend = a.device
    eng = Engine(Za, Zb, backend)
    out = dict(cell=a.cell, prereg="notes/PREREG_ninecell_ladder_20260926.md", N=int(N), topk=TOPK,
               backend=backend, platform=a.platform, host=platform.node(), script_sha256=sha256_file(__file__),
               pooled_sha256={s: sha256_file(Path(a.pooled) / f"{s}.npy") for s in (A, B)},
               wc_min_med_max=[int(wc.min()), int(np.median(wc)), int(wc.max())],
               n_len_bins_eff=int(len(np.unique(strata["len"]))),
               block_sizes={k: sorted(np.bincount(v).tolist()) for k, v in strata.items()},
               obs_M=obs)
    # G0a
    rc = Path(a.receipts) / f"c4n2k_{a.cell}.json"
    if rc.exists() and N == 2000:
        pub = json.load(open(rc))["observed"]["n_mutual_nn"]
        out["G0a"] = dict(pub_M=int(pub), M=obs, dM=int(obs - pub), tol=3, pass_=bool(abs(obs - pub) <= 3))
    npm, ncal, npc = (a.n_perm, a.n_cal, a.n_perm_cal)
    # ---- Q-A point estimates
    pt = {"base": arm(eng, obs, None, npm, 0)}
    pt["len"] = ladder_arms(eng, obs, strata["len"], npm, ncal, npc, (1, 1234, 3))
    pt["top10"] = ladder_arms(eng, obs, strata["top10"], npm, ncal, npc, (2, 5678, 103))
    for k in ncl[1:]:
        pt[f"top{k}"] = ladder_arms(eng, obs, strata[f"top{k}"], npm, ncal, npc, (2 + k, 9000 + k, 300 + k))
    for key in pt:
        if key != "base":
            pt[key]["raw_ratio_to_base"] = float(pt[key]["excess"] / pt["base"]["excess"])
    out["point"] = pt
    out["point_wall_s"] = time.time() - t0
    print(f"[A:{a.cell}] point done {out['point_wall_s']:.0f}s", flush=True)
    # ---- Q-C bootstrap
    boots = []
    t1 = time.time()
    for b in range(a.n_boot):
        rr = np.random.default_rng(777 + b).choice(N, N, replace=True)
        Zab, Zbb, ob, _, _ = prep(Pa[rr], Pb[rr])
        e = Engine(Zab, Zbb, backend)
        s0 = 100_000 * (b + 1)
        rec = {"b": b, "M": ob, "base": arm(e, ob, None, a.boot_perm, s0)["excess"]}
        for j, key in enumerate(["len", "top10"] + [f"top{k}" for k in ncl[1:]]):
            x = ladder_arms(e, ob, strata[key][rr], a.boot_perm, a.boot_cal, a.boot_cal_perm,
                            (s0 + 10 * (j + 1), s0 + 1000 * (j + 1), s0 + 10 * (j + 1) + 1))
            rec[key] = dict(excess=x["excess"], cal=x["calibration"]["excess"], R=x["R"])
        boots.append(rec)
        del e
    q = (2.5, 97.5)
    ci = {"excess_frac_base": pct([r["base"] / TOPK for r in boots], q)}
    for key in ["len", "top10"] + [f"top{k}" for k in ncl[1:]]:
        ci[f"R_{key}"] = pct([r[key]["R"] for r in boots], q)
    out["bootstrap"] = dict(n_boot=a.n_boot, seed="777+b", n_perm=a.boot_perm, n_cal=a.boot_cal,
                            n_perm_cal=a.boot_cal_perm, ci95=ci, wall_s=time.time() - t1, draws=boots)
    out["date"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{a.out}.tmp{os.getpid()}"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(f"[A:{a.cell}] done boot {out['bootstrap']['wall_s']:.0f}s -> {a.out}", flush=True)


# ---------------------------------------------------------------------------------- Q-B, Delta_E
def frozen_pairs(Pa_sel, Pb_sel, Pa_eval, Pb_eval):
    """Lines of heldout_frozen_pairs up to the frozen pair set, copied so the pairs can be reused
    for the Delta_E bootstrap; cmd_analyze_b asserts it reproduces heldout_frozen_pairs' obs."""
    Za_s, ka = _zscore_cols(np.asarray(Pa_sel, float)); Zb_s, kb = _zscore_cols(np.asarray(Pb_sel, float))
    Pa_e = np.asarray(Pa_eval, float)[:, ka]; Pb_e = np.asarray(Pb_eval, float)[:, kb]
    Za_e, ka2 = _zscore_cols(Pa_e); Zb_e, kb2 = _zscore_cols(Pb_e)
    Za_s = Za_s[:, ka2]; Zb_s = Zb_s[:, kb2]
    C = (Za_s.T @ Zb_s) / Za_s.shape[0]
    a2b = C.argmax(1); b2a = C.argmax(0)
    recip = b2a[a2b] == np.arange(C.shape[0])
    ia = np.arange(C.shape[0])[recip]; jb = a2b[recip]
    return Pa_e[:, ka2][:, ia], Pb_e[:, kb2][:, jb], Za_e, Zb_e, ia, jb


def run_split(Pa, Pb, sel, ev, n_perm):
    Fa, ja = topk_features(Pa[sel], TOPK); Fb, jb = topk_features(Pb[sel], TOPK)
    h = heldout_frozen_pairs(Fa, Fb, Pa[ev][:, ja], Pb[ev][:, jb], n_perm=n_perm, seed=0)
    return h, (Fa, Fb, Pa[ev][:, ja], Pb[ev][:, jb])


def zc(X):
    mu = X.mean(0); sd = X.std(0)
    ok = sd > 1e-12
    Z = np.zeros_like(X); Z[:, ok] = (X[:, ok] - mu[ok]) / sd[ok]
    return np.clip(Z, -50.0, 50.0)


def cmd_analyze_b(a):
    A, B = a.cell.split("__")
    Pa = np.load(Path(a.pooled) / f"{A}.npy"); Pb = np.load(Path(a.pooled) / f"{B}.npy")
    N = min(Pa.shape[0], Pb.shape[0]); Pa, Pb = Pa[:N], Pb[:N]; H = N // 2
    out = dict(cell=a.cell, prereg="notes/PREREG_ninecell_ladder_20260926.md", N=int(N),
               n_perm=a.n_perm, platform=a.platform, host=platform.node(), script_sha256=sha256_file(__file__),
               pooled_sha256={s: sha256_file(Path(a.pooled) / f"{s}.npy") for s in (A, B)})
    t0 = time.time()
    hc, parts = run_split(Pa, Pb, np.arange(H), np.arange(H, N), a.n_perm)
    out["consecutive"] = hc
    rc = Path(a.receipts) / f"heldout_{a.cell}.json"
    if rc.exists() and N == 2000:
        pub = json.load(open(rc))
        d = hc["held_out"]["excess"] - pub["held_out"]["excess"]
        out["G0b"] = dict(pub_excess=pub["held_out"]["excess"], excess=hc["held_out"]["excess"], d=float(d),
                          tol=0.005, pub_pairs=pub["n_pairs_selected"], pairs=hc["n_pairs_selected"],
                          pub_r=pub["ratio_heldout_over_insample"], r=hc["ratio_heldout_over_insample"],
                          pass_=bool(abs(d) <= 0.005))
    # Q-B
    splits = []
    for s in SPLIT_SEEDS:
        perm = np.random.default_rng(s).permutation(N)
        sel, ev = np.sort(perm[:H]), np.sort(perm[H:])
        h, _ = run_split(Pa, Pb, sel, ev, a.n_perm)
        splits.append(dict(split_seed=s, n_pairs=h["n_pairs_selected"], excess=h["held_out"]["excess"],
                           p=h["held_out"]["p"], z=h["held_out"]["z"], r=h["ratio_heldout_over_insample"],
                           insample_excess=h["in_sample_eval_own_pairs"]["excess"]))
        print(f"[B:{a.cell}] split {s} done", flush=True)
    ex = np.array([x["excess"] for x in splits]); rs = np.array([x["r"] for x in splits])
    cons = hc["held_out"]["excess"]
    out["random_splits"] = dict(splits=splits, min_excess=float(ex.min()), max_excess=float(ex.max()),
                                max_p=float(max(x["p"] for x in splits)),
                                median_r=float(np.median(rs)), r_range=[float(rs.min()), float(rs.max())],
                                rho=float(np.median(ex / cons)), alpha=ALPHA_B)
    # Q-C Delta_E bootstrap on the consecutive split's frozen pairs
    PAe, PBe, Za_e, Zb_e, ia, jb = frozen_pairs(*parts)
    obs_rep = float((Za_e[:, ia] * Zb_e[:, jb]).sum(0).mean() / Za_e.shape[0])
    assert abs(obs_rep - hc["held_out"]["obs_mean_pair_corr"]) < 1e-9, (obs_rep, hc["held_out"])
    ne = PAe.shape[0]; draws = []
    for b in range(a.n_boot_e):
        rng = np.random.default_rng(5000 + b)
        rr = rng.choice(ne, ne, replace=True)
        Xa, Xb = zc(PAe[rr]), zc(PBe[rr])
        st = float((Xa * Xb).sum(0).mean() / ne)
        nl = [float((Xa * Xb[rng.permutation(ne)]).sum(0).mean() / ne) for _ in range(a.boot_e_perm)]
        draws.append(st - float(np.mean(nl)))
    out["deltaE_bootstrap"] = dict(n_boot=a.n_boot_e, n_perm_each=a.boot_e_perm, seed="5000+b",
                                   point=hc["held_out"]["excess"], ci95=pct(draws, (2.5, 97.5)),
                                   conditional_on="consecutive-split frozen pair set",
                                   draws=[float(x) for x in draws])
    out["wall_s"] = time.time() - t0
    out["date"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[B:{a.cell}] done {out['wall_s']:.0f}s -> {a.out}", flush=True)


# ------------------------------------------------------------------------------------ decisions
def decide(A_res, B_res):
    """Registered rules (prereg sec. 5). A_res/B_res: {cell: receipt dict}."""
    cells = {}
    for c, r in A_res.items():
        g0 = r.get("G0a", {}).get("pass_", False) and B_res.get(c, {}).get("G0b", {}).get("pass_", False)
        arms = {}
        for key in ("len", "top10"):
            x = r["point"][key]
            arms[key] = dict(R=x["R"], p=x["p"], survives=bool(x["p"] <= ALPHA_A and x["R"] is not None and x["R"] >= R_MIN))
        cells[c] = dict(g0=bool(g0), arms=arms)
    counted = {c: v for c, v in cells.items() if v["g0"]}
    fail = [c for c, v in counted.items() if not all(a["survives"] for a in v["arms"].values())]
    len_hard = [c for c, v in counted.items()
                if (not v["arms"]["len"]["survives"]) and (v["arms"]["len"]["R"] is None or v["arms"]["len"]["R"] < R_MIN)]
    if not counted:
        qa = "A0"
    elif len(fail) >= 3 or len_hard:
        qa = "A3"
    elif fail:
        qa = "A2"
    else:
        qa = "A1"
    # Q-B
    bfail = {}
    rhos = {}
    for c, r in B_res.items():
        sp = r["random_splits"]["splits"]
        n_bad = sum(1 for x in sp if not (x["excess"] > 0 and x["p"] <= ALPHA_B))
        if n_bad:
            bfail[c] = n_bad
        rhos[c] = r["random_splits"]["rho"]
    qb = "B2" if bfail else "B1"
    lo = sum(1 for v in rhos.values() if v < 0.90); hi = sum(1 for v in rhos.values() if v > 1.10)
    opt = "consecutive may overstate" if lo >= 3 else ("consecutive conservative" if hi >= 3 else "within +-10%")
    # Q-C
    lows, not_computed = [], []
    for c, r in A_res.items():
        for k, v in r["bootstrap"]["ci95"].items():
            if not k.startswith("R_"):
                continue
            if v is None:                       # n_boot = 0 (prereg deviation 1): not computed
                not_computed.append(f"{c}:{k}")
            elif v[0] < R_MIN:
                lows.append(f"{c}:{k}")
    e_bad = [c for c, r in B_res.items() if not (r["deltaE_bootstrap"]["ci95"] and r["deltaE_bootstrap"]["ci95"][0] > 0)]
    qc = "C2" if (lows or e_bad) else "C1"
    return dict(QA=qa, QA_failing_cells=fail, QA_length_hard_fail=len_hard,
                QA_g0_blocked=[c for c, v in cells.items() if not v["g0"]], QA_cells=cells,
                QB=qb, QB_failing=bfail, QB_rho=rhos, QB_optimism=opt,
                QC=qc, QC_R_lower_below_0p5=lows, QC_deltaE_nonpositive=e_bad,
                QC_R_not_computed=not_computed,
                cells_run=sorted(A_res), n_cells_run=len(A_res),
                coverage_note=("" if len(A_res) == 9 else
                               f"{len(A_res)} of 9 cells: branch labels apply to these cells only"))


def f3(x):
    return f"{x:.3f}"


def cmd_summarize(a):
    d = Path(a.dir)
    A_res = {c: json.load(open(d / f"A_{c}.json")) for c in CELLS if (d / f"A_{c}.json").exists()}
    B_res = {c: json.load(open(d / f"B_{c}.json")) for c in CELLS if (d / f"B_{c}.json").exists()}
    dec = decide(A_res, B_res)
    rows = {}
    for c in CELLS:
        if c not in A_res or c not in B_res:
            continue
        r, s = A_res[c], B_res[c]
        pt, ci = r["point"], r["bootstrap"]["ci95"]
        rows[c] = dict(
            M=r["obs_M"], base_excess=pt["base"]["excess"], base_p=pt["base"]["p"],
            len_excess=pt["len"]["excess"], len_cal=pt["len"]["calibration"]["excess"], R_len=pt["len"]["R"],
            len_p=pt["len"]["p"], R_len_ci=ci["R_len"],
            top10_excess=pt["top10"]["excess"], top10_cal=pt["top10"]["calibration"]["excess"],
            R_top10=pt["top10"]["R"], top10_p=pt["top10"]["p"], R_top10_ci=ci["R_top10"],
            excess_frac_ci=ci["excess_frac_base"],
            R_top200=pt.get("top200", {}).get("R"), R_top200_ci=ci.get("R_top200"),
            R_top500=pt.get("top500", {}).get("R"), R_top500_ci=ci.get("R_top500"),
            dE_cons=s["consecutive"]["held_out"]["excess"], dE_ci=s["deltaE_bootstrap"]["ci95"],
            rs_min_excess=s["random_splits"]["min_excess"], rs_max_excess=s["random_splits"]["max_excess"],
            rs_max_p=s["random_splits"]["max_p"], rs_median_r=s["random_splits"]["median_r"],
            rs_r_range=s["random_splits"]["r_range"], rho=s["random_splits"]["rho"],
            G0a=r.get("G0a"), G0b=s.get("G0b"))
    plats = sorted({r.get("platform", "?") for r in list(A_res.values()) + list(B_res.values())})
    summ = dict(prereg="notes/PREREG_ninecell_ladder_20260926.md", decision=dec, rows=rows, platforms=plats,
                n_cells_A=len(A_res), n_cells_B=len(B_res), date=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(summ, open(d / "NINECELL_SUMMARY.json", "w"), indent=1)
    # LaTeX table (appendix), every number from the receipts above
    lab = {"qwen3-1.7b": "Q1.7", "qwen3-8b": "Q8", "gemma2-2b": "G2", "gemma2-9b": "G9",
           "mistral7b": "M", "llama3.2-1b": "L1"}
    L = ["% AUTO-GENERATED by code/analysis/ninecell_strengthen_20260926.py summarize -- DO NOT HAND-EDIT.",
         f"% Source: {os.path.relpath(d.resolve(), REPO)}/{{A,B}}_<cell>.json (platform: {', '.join(plats)}); prereg notes/PREREG_ninecell_ladder_20260926.md",
         "\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
         ("Cell & $M$ & $R_{\\text{len}}$ [95\\%] & $R_{\\text{top}}$ ($k{=}10$) [95\\%] & $\\Delta_E$ [95\\%] & "
          if any(v["R_len_ci"] for v in rows.values()) else
          "Cell & $M$ & $R_{\\text{len}}$ & $R_{\\text{top}}$ ($k{=}10$) & $\\Delta_E$ [95\\%] & ")
         + "random split $\\Delta_E$ & median $r$ \\\\", "\\midrule"]
    for c, v in rows.items():
        A_, B_ = c.split("__")
        ci_ = lambda x: f"[{f3(x[0])}, {f3(x[1])}]" if x else ""
        L.append(f"{lab[A_]}--{lab[B_]} & ${v['M']}$ & ${f3(v['R_len'])}$ {ci_(v['R_len_ci'])} & "
                 f"${f3(v['R_top10'])}$ {ci_(v['R_top10_ci'])} & ${f3(v['dE_cons'])}$ {ci_(v['dE_ci'])} & "
                 f"${f3(v['rs_min_excess'])}$--${f3(v['rs_max_excess'])}$ & ${f3(v['rs_median_r'])}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(d / "tab_ninecell.tex", "w").write("\n".join(L) + "\n")
    print(json.dumps({k: dec[k] for k in ("QA", "QA_failing_cells", "QA_length_hard_fail", "QA_g0_blocked",
                                          "QB", "QB_failing", "QB_optimism", "QC",
                                          "QC_R_lower_below_0p5", "QC_deltaE_nonpositive")}, indent=1))


# ------------------------------------------------------------------------------------ selftest
def selftest():
    ok = True
    rng = np.random.default_rng(0)
    N, K = 1000, 200
    # (1) numpy vs torch count identity
    Za = _zscore_cols(rng.standard_normal((N, K)))[0]; Zb = _zscore_cols(rng.standard_normal((N, K)))[0]
    perms = draw_perms(N, None, 30, 5)
    m1, g1 = Engine(Za, Zb, "numpy").counts(perms)
    try:
        m2, g2 = Engine(Za, Zb, "cpu", batch=7).counts(perms)
        same = bool(np.array_equal(m1, m2) and np.allclose(g1, g2, atol=1e-12))
    except ImportError:
        same = None
    print(f"[selftest 1] numpy==torch counts: {same}"); ok &= same is not False
    # (2) planted LENGTH confound: shared signal is a function of doc length only
    L = rng.integers(7, 68, N).astype(float)
    ls = np.digitize(L, np.unique(np.quantile(L, np.linspace(0, 1, 11)))[1:-1])
    S = 60
    cen = np.linspace(7, 67, S)
    F = np.exp(-(L[:, None] - cen[None, :]) ** 2 / (2 * 8.0 ** 2))       # length-tuned features
    F = F / F.std(0)
    Pa = np.hstack([F + 0.3 * rng.standard_normal((N, S)), rng.standard_normal((N, K - S))])
    Pb = np.hstack([F[:, rng.permutation(S)] + 0.3 * rng.standard_normal((N, S)), rng.standard_normal((N, K - S))])
    res = _planted(Pa, Pb, ls)
    c2 = res["len"]["R"] is not None and res["len"]["R"] < 0.5 and res["base"]["excess"] > 10
    print(f"[selftest 2] length-driven sharing: base excess {res['base']['excess']:.1f}, "
          f"R_len {res['len']['R']:.3f} (<0.5 expected) -> {c2}"); ok &= c2
    # (3) planted length-FREE sharing survives the length null
    S3 = 150      # M is a low-power count at small K (chance reciprocity ~K/2), so plant densely
    Lat = rng.standard_normal((N, S3))
    Pa3 = np.hstack([Lat + 0.3 * rng.standard_normal((N, S3)), rng.standard_normal((N, K - S3))])
    Pb3 = np.hstack([Lat[:, rng.permutation(S3)] + 0.3 * rng.standard_normal((N, S3)), rng.standard_normal((N, K - S3))])
    res3 = _planted(Pa3, Pb3, ls)
    c3 = res3["len"]["R"] > 0.8 and res3["len"]["p"] < 0.05
    print(f"[selftest 3] length-free sharing: R_len {res3['len']['R']:.3f} (>0.8 expected) p {res3['len']['p']:.3f} -> {c3}")
    ok &= c3
    # (4) random-split held-out: planted significant, noise not
    Pn = rng.standard_normal((N, K)); Pm = rng.standard_normal((N, K))
    perm = np.random.default_rng(1).permutation(N); sel, ev = np.sort(perm[:N // 2]), np.sort(perm[N // 2:])
    hp, _ = run_split_k(Pa3, Pb3, sel, ev, 200, K); hn, _ = run_split_k(Pn, Pm, sel, ev, 200, K)
    c4 = hp["held_out"]["p"] < 0.01 and hn["held_out"]["p"] > 0.05
    print(f"[selftest 4] held-out planted p {hp['held_out']['p']:.4f}, noise p {hn['held_out']['p']:.3f} -> {c4}")
    ok &= c4
    # (5) decision branches
    c5 = _branch_tests()
    print(f"[selftest 5] decision branches -> {c5}"); ok &= c5
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _planted(Pa, Pb, ls):
    Za = _zscore_cols(Pa)[0]; Zb = _zscore_cols(Pb)[0]
    C = (Za.T @ Zb) / Za.shape[0]; a2b = C.argmax(1); b2a = C.argmax(0)
    obs = int((b2a[a2b] == np.arange(C.shape[0])).sum())
    e = Engine(Za, Zb, "numpy")
    return {"base": arm(e, obs, None, 100, 0), "len": ladder_arms(e, obs, ls, 100, 5, 40, (1, 1234, 3))}


def run_split_k(Pa, Pb, sel, ev, n_perm, k):
    Fa, ja = topk_features(Pa[sel], k); Fb, jb = topk_features(Pb[sel], k)
    return heldout_frozen_pairs(Fa, Fb, Pa[ev][:, ja], Pb[ev][:, jb], n_perm=n_perm, seed=0), None


def _fake(R_len, R_top, p=1 / 2001, g0=True, rlo=0.8, elo=0.1, sp_p=1 / 2001, rho=1.0):
    A = {"G0a": {"pass_": g0}, "point": {"len": {"R": R_len, "p": p}, "top10": {"R": R_top, "p": p}},
         "bootstrap": {"ci95": {"R_len": [rlo, 1.0], "R_top10": [rlo, 1.0]}}}
    B = {"G0b": {"pass_": g0}, "random_splits": {"splits": [{"excess": 0.5, "p": sp_p}] * 10, "rho": rho},
         "deltaE_bootstrap": {"ci95": [elo, 0.9]}}
    return A, B


def _branch_tests():
    def run(specs):
        A, B = {}, {}
        for i, s in enumerate(specs):
            A[f"c{i}"], B[f"c{i}"] = _fake(**s)
        return decide(A, B)
    base = [dict(R_len=0.95, R_top=0.97)] * 9
    t = [run(base)["QA"] == "A1" and run(base)["QB"] == "B1" and run(base)["QC"] == "C1"]
    t.append(run(base[:8] + [dict(R_len=0.95, R_top=0.40)])["QA"] == "A2")
    t.append(run(base[:8] + [dict(R_len=0.40, R_top=0.97)])["QA"] == "A3")
    t.append(run(base[:6] + [dict(R_len=0.95, R_top=0.40)] * 3)["QA"] == "A3")
    t.append(run(base[:8] + [dict(R_len=0.95, R_top=0.97, p=0.01)])["QA"] == "A2")
    t.append(run(base[:8] + [dict(R_len=0.40, R_top=0.40, g0=False)])["QA"] == "A1")
    t.append(run(base[:8] + [dict(R_len=0.95, R_top=0.97, sp_p=0.02)])["QB"] == "B2")
    t.append(run(base[:8] + [dict(R_len=0.95, R_top=0.97, rlo=0.3)])["QC"] == "C2")
    t.append(run(base[:8] + [dict(R_len=0.95, R_top=0.97, elo=-0.01)])["QC"] == "C2")
    t.append(run([dict(R_len=0.95, R_top=0.97, rho=0.8)] * 3 + base[:6])["QB_optimism"] == "consecutive may overstate")
    return all(t)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c4d = os.environ.get("NC_C4", str(REPO / "data" / "c4_slice.jsonl"))
    rcd = os.environ.get("NC_RECEIPTS", str(REPO / "results" / "cross_model_sae"))
    e = sub.add_parser("extract"); e.add_argument("--model", required=True, choices=list(MODELS))
    e.add_argument("--pooled", required=True); e.add_argument("--max_docs", type=int, default=None)
    e.add_argument("--c4", default=c4d)
    x = sub.add_parser("analyze_a"); x.add_argument("--cell", required=True); x.add_argument("--pooled", required=True)
    x.add_argument("--out", required=True); x.add_argument("--device", default="cuda")
    x.add_argument("--c4", default=c4d); x.add_argument("--receipts", default=rcd)
    x.add_argument("--n_perm", type=int, default=N_PERM); x.add_argument("--n_cal", type=int, default=N_CAL)
    x.add_argument("--n_perm_cal", type=int, default=N_PERM_CAL)
    x.add_argument("--n_boot", type=int, default=200); x.add_argument("--boot_perm", type=int, default=100)
    x.add_argument("--boot_cal", type=int, default=5); x.add_argument("--boot_cal_perm", type=int, default=40)
    x.add_argument("--platform", default="cluster")
    y = sub.add_parser("analyze_b"); y.add_argument("--cell", required=True); y.add_argument("--pooled", required=True)
    y.add_argument("--out", required=True); y.add_argument("--receipts", default=rcd)
    y.add_argument("--n_perm", type=int, default=N_PERM)
    y.add_argument("--n_boot_e", type=int, default=1000); y.add_argument("--boot_e_perm", type=int, default=20)
    y.add_argument("--platform", default="cluster")
    s = sub.add_parser("summarize"); s.add_argument("--dir", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(selftest())
    {"extract": cmd_extract, "analyze_a": cmd_analyze_a, "analyze_b": cmd_analyze_b,
     "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()

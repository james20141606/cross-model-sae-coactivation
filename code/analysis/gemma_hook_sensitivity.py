#!/usr/bin/env python3
"""Gemma Scope read-point sensitivity (tier 1), as registered in
notes/PREREG_gemma_hook_sensitivity_20260924.md.

The only change under test: Gemma Scope `layer_N` dictionaries applied to hidden_states[N+1]
(their training hook, resid_post N) instead of hidden_states[N] (what every published
receipt used). The index comes from cross_model_sae_match.resolve_hs_idx with
gemma_hook_offset 0 / 1, and the statistics are cross_model_sae_match.match_and_null and
heldout_frozen_pairs, imported unchanged.

Subcommands
  extract --model TAG    one forward pass over the 2,000-doc C4 slice; mean-pooled SAE features
                         (all tokens incl. BOS, as compute_pooled_features) for every read point
                         (Gemma: N and N+1; others: their published index); FVU / L0 diagnostics
                         on the first 200 docs (gate H) and L0 over all docs, with/without BOS.
  match                  tier-1 cells at offset 0 and 1: screen (n_perm 200) + held-out
                         (n_perm 2000); gate G0 against the published receipts; gate H; decision
                         rule of prereg sec. 6; writes GEMMA_HOOK_SENSITIVITY.json.
"""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from cross_model_sae_match import (heldout_frozen_pairs, match_and_null, resolve_hs_idx,  # noqa: E402
                                   topk_features)

CACHE = Path(os.environ.get("GHS_CACHE", str(Path.home() / "gemmahook_cache" / "pooled")))
C4 = os.environ.get("GHS_C4", str(Path.home() / "lanfilt_cache" / "c4_slice.jsonl"))
C4_MD5 = "30fe9a250f82eb2bcc94ff2282e1407f"
OUTDIR = REPO / "results" / "cross_model_sae" / "gemma_hook_sensitivity"
N_DIAG = 200
# Gemma weights: "unsloth" (ungated re-uploads; the local MPS run, no HF token) or "google"
# (the gated originals every published receipt used; the cluster run, 2026-09-24 evening).
GEMMA_SRC = os.environ.get("GHS_GEMMA_WEIGHTS", "unsloth")
assert GEMMA_SRC in ("unsloth", "google"), GEMMA_SRC

MODELS = {
    "gemma2-2b": dict(hf=f"{GEMMA_SRC}/gemma-2-2b", gemma=True, saes=[
        ("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical", "16k", 82),
        ("gemma-scope-2b-pt-res-canonical", "layer_12/width_65k/canonical", "65k", 72)]),
    "gemma2-9b": dict(hf=f"{GEMMA_SRC}/gemma-2-9b", gemma=True, saes=[
        ("gemma-scope-9b-pt-res-canonical", "layer_20/width_16k/canonical", "16k", 68)]),
    "mistral7b": dict(hf="mistralai/Mistral-7B-v0.1", gemma=False, saes=[
        ("mistral-7b-res-wg", "blocks.16.hook_resid_pre", "wg", None)]),
    "qwen3-1.7b": dict(hf="Qwen/Qwen3-1.7B-Base", gemma=False, saes=[
        ("qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", "layer14", "w32k", None)]),
    "qwen3-8b": dict(hf="Qwen/Qwen3-8B-Base", gemma=False, saes=[
        ("qwenpt:Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", "layer18", "w64k", None)]),
}
# (receipt tag, side A (model, sae tag), side B, held-out cell?, published hidden_states idx of Gemma dicts)
CELLS = [
    ("qwen3-1.7b__gemma2-2b", ("qwen3-1.7b", "w32k"), ("gemma2-2b", "16k"), True),
    ("qwen3-8b__gemma2-2b", ("qwen3-8b", "w64k"), ("gemma2-2b", "16k"), True),
    ("qwen3-1.7b__gemma2-9b", ("qwen3-1.7b", "w32k"), ("gemma2-9b", "16k"), True),
    ("qwen3-8b__gemma2-9b", ("qwen3-8b", "w64k"), ("gemma2-9b", "16k"), True),
    ("gemma2-9b__mistral7b", ("gemma2-9b", "16k"), ("mistral7b", "wg"), True),
    ("gemma2-2b__mistral7b", ("gemma2-2b", "16k"), ("mistral7b", "wg"), True),
    ("gemma2-2b__gemma2-9b", ("gemma2-2b", "16k"), ("gemma2-9b", "16k"), False),
    ("w65k_gemma2-2b__mistral7b", ("gemma2-2b", "65k"), ("mistral7b", "wg"), False),
]
NON_BLIND = {"gemma2-9b__mistral7b"}


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_file(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def load_rows():
    assert md5_file(C4) == C4_MD5, f"C4 slice md5 mismatch: {C4}"
    prompts = [json.loads(l) for l in open(C4)]
    sel = [p for p in prompts if p.get("lang") == "en"]
    sel.sort(key=lambda p: p.get("pair_id", 0))
    rows = [p["prompt"] for p in sel]
    assert len(rows) == 2000
    return rows


def pooled_path(model, sae_tag, idx):
    return CACHE / f"{model}__{sae_tag}__hs{idx}.npy"


# ----------------------------------------------------------------------------- extract
def cmd_extract(args):
    import torch
    import sae_lens, transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE

    spec = MODELS[args.model]
    dev = torch.device(args.device)
    rows = load_rows()
    if args.max_docs:
        rows = rows[:args.max_docs]
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(spec["hf"], trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(spec["hf"], torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).to(dev).eval()
    revision = getattr(mdl.config, "_commit_hash", None)
    saes = []
    for release, sae_id, stag, nominal in spec["saes"]:
        if release.startswith("qwenpt:"):
            from qwen_sae_adapter import load_qwen_sae
            sae = load_qwen_sae(release[len("qwenpt:"):], sae_id, dev)
        else:
            sae = SAE.from_pretrained(release, sae_id, device=str(dev))
            if isinstance(sae, tuple):
                sae = sae[0]
            sae = sae.to(dev)
        cfg = getattr(sae, "cfg", sae)
        offsets = [0, 1] if spec["gemma"] else [0]
        idxs = {}
        for off in offsets:
            hl, idx, mh = resolve_hs_idx(cfg, sae_id, release, gemma_hook_offset=off)
            idxs[off] = idx
        wsha = hashlib.sha256(sae.W_enc.detach().float().cpu().numpy().tobytes()).hexdigest()
        m = getattr(cfg, "metadata", None)
        saes.append(dict(sae=sae, release=release, sae_id=sae_id, tag=stag, nominal=nominal,
                         hook_layer=hl, idxs=idxs, meta_hook=mh, dtype=sae.W_enc.dtype,
                         W_enc_sha256=wsha, d_sae=int(sae.W_enc.shape[0] if release.startswith("qwenpt:")
                                                      else sae.cfg.d_sae),
                         neuronpedia=getattr(m, "neuronpedia_id", None) if m is not None else None))
        print(f"[sae] {release}::{sae_id} hook_layer={hl} idxs={idxs} metadata.hook_name={mh}", flush=True)
    print(f"[load] {spec['hf']} rev={revision} {time.time()-t0:.0f}s", flush=True)

    pooled = {(s["tag"], i): np.zeros((len(rows), s["d_sae"]), np.float32)
              for s in saes for i in s["idxs"].values()}
    diag = {(s["tag"], i): dict(se=0.0, se_b=0.0, n=0, n_b=0, l0=0.0, l0_b=0.0, h=[], hb=[])
            for s in saes for i in s["idxs"].values()}
    l0all = {(s["tag"], i): dict(l0=0.0, n=0, l0_b=0.0, n_b=0) for s in saes for i in s["idxs"].values()}
    n_tok, n_bos = 0, 0
    t1 = time.time()
    for d, text in enumerate(rows):
        inp = tok(text, return_tensors="pt").to(dev)
        ids = inp["input_ids"][0]
        has_bos = tok.bos_token_id is not None and int(ids[0]) == tok.bos_token_id
        n_tok += int(ids.shape[0]); n_bos += int(has_bos)
        s0 = 1 if has_bos else 0
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True)
            for s in saes:
                for idx in set(s["idxs"].values()):
                    h = out.hidden_states[idx][0]                       # [T, d_model]
                    acts = s["sae"].encode(h.to(s["dtype"]))             # as compute_pooled_features
                    if acts.dim() == 3:
                        acts = acts[0]
                    pooled[(s["tag"], idx)][d] = acts.mean(0).float().cpu().numpy()
                    nz = (acts > 0).sum(1).float()
                    la = l0all[(s["tag"], idx)]
                    la["l0_b"] += float(nz.sum()); la["n_b"] += int(nz.shape[0])
                    la["l0"] += float(nz[s0:].sum()); la["n"] += int(nz.shape[0] - s0)
                    if d < N_DIAG and s["release"].startswith("gemma-scope"):
                        rec = s["sae"].decode(acts).float()
                        hf_ = h.float()
                        r2 = ((rec - hf_) ** 2).sum(1)
                        g = diag[(s["tag"], idx)]
                        g["se_b"] += float(r2.sum()); g["n_b"] += int(r2.shape[0])
                        g["se"] += float(r2[s0:].sum()); g["n"] += int(r2.shape[0] - s0)
                        g["l0_b"] += float(nz.sum()); g["l0"] += float(nz[s0:].sum())
                        g["h"].append(hf_[s0:].cpu()); g["hb"].append(hf_.cpu())
        if (d + 1) % 250 == 0:
            print(f"  [{d+1}/{len(rows)}] {time.time()-t1:.0f}s", flush=True)
    wall = time.time() - t1

    import torch as _t
    for s in saes:
        for off, idx in s["idxs"].items():
            key = (s["tag"], idx)
            p = pooled_path(args.model, s["tag"], idx) if not args.max_docs else \
                CACHE / f"smoke{args.max_docs}_{args.model}__{s['tag']}__hs{idx}.npy"
            np.save(p, pooled[key])
            dg = diag[key]
            fvu = None
            if dg["h"]:
                H = _t.cat(dg["h"], 0); Hb = _t.cat(dg["hb"], 0)
                tv = float(((H - H.mean(0)) ** 2).sum()); tvb = float(((Hb - Hb.mean(0)) ** 2).sum())
                fvu = dict(n_docs=min(N_DIAG, len(rows)),
                           FVU_noBOS=dg["se"] / tv, FVU_withBOS=dg["se_b"] / tvb,
                           L0_noBOS=dg["l0"] / dg["n"], L0_withBOS=dg["l0_b"] / dg["n_b"],
                           n_tok_noBOS=dg["n"], n_tok_withBOS=dg["n_b"])
            la = l0all[key]
            meta = dict(model=args.model, hf_id=spec["hf"], hf_revision=revision,
                        sae_release=s["release"], sae_id=s["sae_id"], sae_tag=s["tag"],
                        metadata_hook_name=s["meta_hook"], hook_layer=s["hook_layer"],
                        gemma_hook_offset=off, hs_idx=idx, published_hs_idx=s["idxs"][0],
                        nominal_L0=s["nominal"], W_enc_sha256=s["W_enc_sha256"], d_sae=s["d_sae"],
                        corpus=C4, corpus_md5=C4_MD5, n_docs=len(rows), n_tokens=n_tok,
                        n_docs_with_bos=n_bos, pooled_file=str(p), pooled_sha256=sha256_file(p),
                        diag_first_docs=fvu,
                        L0_all_docs=dict(noBOS=la["l0"] / max(la["n"], 1), withBOS=la["l0_b"] / max(la["n_b"], 1)),
                        device=str(dev), platform=platform.platform(), torch=_t.__version__,
                        transformers=transformers.__version__, sae_lens=sae_lens.__version__,
                        script_sha256=sha256_file(__file__),
                        matcher_sha256=sha256_file(HERE / "cross_model_sae_match.py"),
                        wall_s=wall, per_doc_s=wall / len(rows),
                        date=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            json.dump(meta, open(str(p)[:-4] + ".meta.json", "w"), indent=1)
            print(f"[saved] {p.name} off={off} idx={idx} diag={fvu} L0all={meta['L0_all_docs']}", flush=True)
    print(f"[done] {args.model} {len(rows)} docs {wall:.0f}s ({wall/len(rows):.3f} s/doc)", flush=True)


# ----------------------------------------------------------------------------- match
def _load_side(model, stag, off):
    spec = MODELS[model]
    metas = [json.load(open(p)) for p in sorted(CACHE.glob(f"{model}__{stag}__hs*.meta.json"))]
    want = [m for m in metas if (m["gemma_hook_offset"] == off if spec["gemma"] else m["gemma_hook_offset"] == 0)]
    assert len(want) == 1, (model, stag, off, [m["pooled_file"] for m in metas])
    m = want[0]
    P = np.load(m["pooled_file"])
    assert sha256_file(m["pooled_file"]) == m["pooled_sha256"]
    return P, m


def run_cell(tag, A, B, heldout, off):
    Pa, ma = _load_side(*A, off)
    Pb, mb = _load_side(*B, off)
    out = dict(tag=tag, gemma_hook_offset=off,
               side_a=dict(model=ma["hf_id"], rev=ma["hf_revision"], sae=f"{ma['sae_release']}::{ma['sae_id']}",
                           hs_idx=ma["hs_idx"], pooled_sha256=ma["pooled_sha256"]),
               side_b=dict(model=mb["hf_id"], rev=mb["hf_revision"], sae=f"{mb['sae_release']}::{mb['sae_id']}",
                           hs_idx=mb["hs_idx"], pooled_sha256=mb["pooled_sha256"]))
    # screen: exactly the published c4_* path (topk 1500 on all 2,000 docs, n_perm 200, seed 0)
    Sa, ia = topk_features(Pa, 1500); Sb, ib = topk_features(Pb, 1500)
    r = match_and_null(Sa, Sb, n_perm=200, seed=0, n_perm_hungarian=20)
    out["screen"] = dict(M=r["observed"]["n_mutual_nn"], null_mu=r["null_mutual_nn"]["mean"],
                         null_sd=r["null_mutual_nn"]["sd"], z=r["mutual_z"], p=r["mutual_p"],
                         excess_frac=(r["observed"]["n_mutual_nn"] - r["null_mutual_nn"]["mean"]) / 1500,
                         greedy_z=r["greedy_z"], n_perm=200, seed=0)
    if heldout:
        # held-out: exactly the published heldout_* path (universe + pairs on docs 0-999 = c4_fit,
        # frozen pairs scored on docs 1000-1999 = c4_heldout, eval-only null n_perm 2000, seed 0)
        Fa, ja = topk_features(Pa[:1000], 1500); Fb, jb = topk_features(Pb[:1000], 1500)
        h = heldout_frozen_pairs(Fa, Fb, Pa[1000:, ja], Pb[1000:, jb], n_perm=2000, seed=0)
        out["heldout"] = dict(n_pairs=h["n_pairs_selected"], excess=h["held_out"]["excess"],
                              p=h["held_out"]["p"], z=h["held_out"]["z"],
                              r=h["ratio_heldout_over_insample"], n_perm=2000, seed=0)
    return out


def cmd_match(args):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pub = REPO / "results" / "cross_model_sae"
    cells = {}
    for tag, A, B, held in CELLS:
        cells[tag] = {}
        for off in (0, 1):
            try:
                res = run_cell(tag, A, B, held, off)
            except AssertionError as e:
                print(f"[skip] {tag} off={off}: missing pooled features {e}", flush=True)
                continue
            res["script_sha256"] = sha256_file(__file__)
            res["matcher_sha256"] = sha256_file(HERE / "cross_model_sae_match.py")
            json.dump(res, open(OUTDIR / f"off{off}_{tag}.json", "w"), indent=1)
            cells[tag][off] = res
            s = res["screen"]; hh = res.get("heldout", {})
            print(f"[cell] {tag} off={off} M={s['M']} mu={s['null_mu']:.1f} z={s['z']:.2f} p={s['p']:.4f}"
                  + (f" | held exc={hh['excess']:.4f} p={hh['p']:.5f} r={hh['r']:.4f} pairs={hh['n_pairs']}" if hh else ""),
                  flush=True)

    # gate G0: offset 0 reproduces the published receipts
    g0 = {}
    for tag, A, B, held in CELLS:
        if 0 not in cells.get(tag, {}):
            continue
        c = json.load(open(pub / f"c4_{tag}.json"))
        mine = cells[tag][0]
        e = dict(pub_M=c["observed"]["n_mutual_nn"], loc_M=mine["screen"]["M"],
                 dM=mine["screen"]["M"] - c["observed"]["n_mutual_nn"],
                 pub_z=c["mutual_z"], loc_z=mine["screen"]["z"])
        ok = abs(e["dM"]) <= 2
        if held:
            hp = json.load(open(pub / f"heldout_{tag}.json"))
            e.update(pub_excess=hp["held_out"]["excess"], loc_excess=mine["heldout"]["excess"],
                     d_excess=mine["heldout"]["excess"] - hp["held_out"]["excess"],
                     pub_r=hp["ratio_heldout_over_insample"], loc_r=mine["heldout"]["r"],
                     pub_pairs=hp["n_pairs_selected"], loc_pairs=mine["heldout"]["n_pairs"])
            ok = ok and abs(e["d_excess"]) <= 0.005
        e["pass"] = bool(ok)
        g0[tag] = e

    # gate H: per Gemma dictionary, FVU lower and L0 closer to nominal at N+1 (BOS excluded)
    gh = {}
    for model in ("gemma2-2b", "gemma2-9b"):
        for release, sae_id, stag, nominal in MODELS[model]["saes"]:
            try:
                _, m0 = _load_side(model, stag, 0); _, m1 = _load_side(model, stag, 1)
            except AssertionError:
                continue
            d0, d1 = m0["diag_first_docs"], m1["diag_first_docs"]
            ok = d1["FVU_noBOS"] < d0["FVU_noBOS"] and abs(d1["L0_noBOS"] - nominal) < abs(d0["L0_noBOS"] - nominal)
            gh[f"{model}:{stag}"] = dict(nominal_L0=nominal, idx_N=m0["hs_idx"], idx_N1=m1["hs_idx"],
                                         N=d0, N1=d1, L0_all_docs_N=m0["L0_all_docs"],
                                         L0_all_docs_N1=m1["L0_all_docs"], pass_=bool(ok))

    # decision (prereg sec. 6)
    dec = {}
    for tag, A, B, held in CELLS:
        if not held or 1 not in cells.get(tag, {}):
            continue
        gem = [x for x in (A, B) if MODELS[x[0]]["gemma"]]
        blocked = (not g0.get(tag, {}).get("pass", False)) or \
            any(not gh.get(f"{m}:{s}", {}).get("pass_", False) for m, s in gem)
        hp = json.load(open(pub / f"heldout_{tag}.json"))
        n1 = cells[tag][1]
        retained = (n1["heldout"]["p"] <= 1 / 2001 + 1e-12 and n1["heldout"]["excess"] > 0
                    and n1["screen"]["p"] <= 1 / 201 + 1e-12)
        dx = n1["heldout"]["excess"] - hp["held_out"]["excess"]
        dr = n1["heldout"]["r"] - hp["ratio_heldout_over_insample"]
        dec[tag] = dict(blocked_by_gate=bool(blocked), non_blind=tag in NON_BLIND,
                        excess_N=hp["held_out"]["excess"], excess_N1=n1["heldout"]["excess"], d_excess=dx,
                        r_N=hp["ratio_heldout_over_insample"], r_N1=n1["heldout"]["r"], d_r=dr,
                        p_N1=n1["heldout"]["p"], screen_p_N1=n1["screen"]["p"],
                        z_N=json.load(open(pub / f"c4_{tag}.json"))["mutual_z"], z_N1=n1["screen"]["z"],
                        retained=bool(retained), in_band=bool(abs(dx) <= 0.05 and abs(dr) <= 0.05))
    blind = [d for t, d in dec.items() if not d["non_blind"] and not d["blocked_by_gate"]]
    outcome = None
    if blind:
        if not all(d["retained"] for d in blind):
            outcome = "C"
        elif all(d["in_band"] for d in blind):
            outcome = "A"
        else:
            outcome = "B"
    summary = dict(prereg="notes/PREREG_gemma_hook_sensitivity_20260924.md",
                   date=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   script_sha256=sha256_file(__file__),
                   matcher_sha256=sha256_file(HERE / "cross_model_sae_match.py"),
                   G0=g0, H=gh, decision=dec,
                   n_blind_evaluated=len(blind),
                   outcome_on_blind_cells=outcome,
                   non_blind_cells=sorted(NON_BLIND),
                   screen={t: {o: cells[t][o]["screen"] for o in cells[t]} for t in cells})
    json.dump(summary, open(OUTDIR / "GEMMA_HOOK_SENSITIVITY.json", "w"), indent=1)
    print(json.dumps(dict(G0={t: e["pass"] for t, e in g0.items()}, H={k: v["pass_"] for k, v in gh.items()},
                          outcome=outcome, n_blind=len(blind)), indent=1))


# ----------------------------------------------------------------------------- interp (exploratory)
def cmd_interp(args):
    """EXPLORATORY, NOT in the registration (added 2026-09-24 after the tier-1 gates, before this
    was run). Re-runs the top-document audit of interpret_matched_features.py (top-20 mutual-NN
    pairs by pooled corr, top-5 docs by pooled activation, all on the 2,000-doc slice) for
    Gemma-2-9B <-> Mistral-7B at offset 0 and 1, and tracks the abstract's obituary pair
    (Gemma 9492 <-> Mistral 2371 in interp_gemma2-9b__mistral7b.json)."""
    from cross_model_sae_match import _zscore_cols
    pub = json.load(open(REPO / "results/cross_model_sae/interp_gemma2-9b__mistral7b.json"))
    OBIT = (9492, 2371)
    out = dict(exploratory=True, note=cmd_interp.__doc__, published_mean_overlap=pub["mean_top_doc_overlap"],
               published_pairs=[[q["feature_a"], q["feature_b"], q["top_doc_overlap"]] for q in pub["pairs"]])
    for off in (0, 1):
        Pa, ma = _load_side("gemma2-9b", "16k", off); Pb, mb = _load_side("mistral7b", "wg", off)
        PaK, ia = topk_features(Pa, 1500); PbK, ib = topk_features(Pb, 1500)
        Za, ka = _zscore_cols(PaK); Zb, kb = _zscore_cols(PbK)
        ia, ib = ia[ka], ib[kb]
        C = (Za.T @ Zb) / Za.shape[0]
        a2b = C.argmax(1); b2a = C.argmax(0)
        recip = np.where(b2a[a2b] == np.arange(C.shape[0]))[0]
        order = recip[np.argsort(C[recip, a2b[recip]])[::-1]][:20]
        pairs = []
        for r in order:
            fa = int(ia[r]); fb = int(ib[a2b[r]])
            ta = np.argsort(Pa[:, fa])[::-1][:5]; tb = np.argsort(Pb[:, fb])[::-1][:5]
            pairs.append([fa, fb, len(set(ta.tolist()) & set(tb.tolist())), float(C[r, a2b[r]])])
        # obituary pair at this read point
        ob = {}
        fa, fb = OBIT
        ta = np.argsort(Pa[:, fa])[::-1][:5]; tb = np.argsort(Pb[:, fb])[::-1][:5]
        ob["top5_overlap_same_indices"] = len(set(ta.tolist()) & set(tb.tolist()))
        ob["gemma_top5_docs"] = ta.tolist(); ob["mistral_top5_docs"] = tb.tolist()
        ob["pooled_corr"] = float(np.corrcoef(Pa[:, fa], Pb[:, fb])[0, 1])
        ob["gemma_in_topk"] = bool(fa in set(ia.tolist()))
        if ob["gemma_in_topk"]:
            r = int(np.where(ia == fa)[0][0]); j = int(a2b[r])
            ob["gemma_best_mistral"] = int(ib[j]); ob["mutual_nn"] = bool(b2a[j] == r)
            ob["rank_among_mutual_by_corr"] = (int(np.where(recip[np.argsort(C[recip, a2b[recip]])[::-1]] == r)[0][0]) + 1
                                               if ob["mutual_nn"] else None)
        out[f"off{off}"] = dict(hs_idx_gemma=ma["hs_idx"], mean_top_doc_overlap=float(np.mean([q[2] for q in pairs])),
                                n_pairs_5of5=sum(q[2] == 5 for q in pairs), pairs=pairs, obituary=ob)
        print(f"[interp] off={off} idx={ma['hs_idx']} mean overlap={out[f'off{off}']['mean_top_doc_overlap']:.2f} "
              f"obituary={ob}", flush=True)
    out["script_sha256"] = sha256_file(__file__)
    json.dump(out, open(OUTDIR / "interp_explore_gemma2-9b__mistral7b.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.add_argument("--model", required=True, choices=sorted(MODELS))
    e.add_argument("--device", default="mps"); e.add_argument("--max_docs", type=int, default=None)
    sp.add_parser("match")
    sp.add_parser("interp")
    a = ap.parse_args()
    {"extract": cmd_extract, "match": cmd_match, "interp": cmd_interp}[a.cmd](a)


if __name__ == "__main__":
    main()

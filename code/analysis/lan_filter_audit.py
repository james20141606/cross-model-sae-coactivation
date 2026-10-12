#!/usr/bin/env python3
"""Lan et al. filtering steps added to the Lan-protocol audit (App. app:lan_filters).

Pre-registration: notes/PREREG_lan_filters_20260924.md (commit 4cbd289, written before any
activation was extracted). Filters follow arXiv 2410.06981v4 and, where text and code differ,
their code (github.com/wlg1/univ_feat_geom run_pipeline/run_expm_fns.py::run_expm):
  F1 keyword : drop a pair if either feature's top-5 activating tokens contain a non-concept
               token; otherwise keep it only if the two top-5 token lists share a string.
  F2 1-1     : targets hit once in the UNFILTERED pairing keep their pair; for a target hit
               several times keep only the first surviving pair (iteration order).
  F3 low corr: keep pairs with max correlation > 0.1.
Everything else is the Table 19 pipeline, imported unchanged from lan_protocol_audit.py
(_z, svcca, rsa, lan_statistics): mean-pooled per-document SAE features, top-512-by-variance
pool per model, A->B argmax pairing.

Two stages:
  extract : one forward pass per (model, corpus) that mirrors
            cross_model_sae_match.compute_pooled_features (same tokenization, hidden-state
            index, SAE encode, mean pooling) and additionally records, for every SAE feature,
            its top-5 activating token positions (activation > 0) over the whole corpus.
  analyze : G0 reproduction gate against the lan{T,C}_* receipts, then the filtered pipeline
            (primary P, secondary S1 literal strings, S2 paper-text 1-1), with Lan's
            shuffle baseline and our permuted-pairing null pushed through the whole pipeline.

Selftest (CPU, no model): planted keyword structure -> filters keep planted pairs, drop
non-concept and keyword-mismatched pairs; F2 keeps one pair per target; F3 threshold.
"""
import argparse, hashlib, json, os, platform, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lan_protocol_audit import _z, svcca, rsa, lan_statistics  # noqa: E402

NONCONCEPT_STRIPPED = {"", ".", ",", "!", "?", "-", "\\n"}
# Lan's list verbatim (run_expm): used by secondary S1 (raw decoded strings, no strip)
NONCONCEPT_LITERAL = {".", "\\n", "\n", "", " ", "-", ",", "!", "?", "<|endoftext|>", "<bos>",
                      "|bos|", "<pad>"}
MIN_K = 21          # SVCCA uses 20 PCs
N_DRAWS = 200
CORR_FLOOR = 0.1


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- extract
MODELS = {
    # tag: (hf model id actually loaded, receipt model id, sae release, sae id, hidden-state idx)
    "gemma2-9b": ("unsloth/gemma-2-9b", "google/gemma-2-9b", "gemma-scope-9b-pt-res-canonical",
                  "layer_20/width_16k/canonical", 20),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B-Base",
                   "qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", "layer14", 15),
    "pythia70m": ("EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-70m-deduped",
                  "pythia-70m-deduped-res-sm", "blocks.3.hook_resid_post", 4),
    "gpt2": ("gpt2", "gpt2", "gpt2-small-res-jb", "blocks.6.hook_resid_pre", 6),
    "mistral7b": ("mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.1", "mistral-7b-res-wg",
                  "blocks.16.hook_resid_pre", 16),
}


def load_rows(path, max_prompts=2000):
    prompts = [json.loads(l) for l in open(path)]
    sel = [p for p in prompts if p.get("lang") == "en"]      # lang="en" as in the audit
    sel.sort(key=lambda p: p.get("pair_id", 0))
    return [p["prompt"] for p in sel][:max_prompts]


def extract(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    hf_id, _, release, sae_id, hs_idx = MODELS[args.model]
    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).to(dev).eval()
    if release.startswith("qwenpt:"):
        from qwen_sae_adapter import load_qwen_sae
        sae = load_qwen_sae(release[len("qwenpt:"):], sae_id, dev)
    else:
        from sae_lens import SAE
        sae = SAE.from_pretrained(release, sae_id, device=str(dev))
        if isinstance(sae, tuple):
            sae = sae[0]
    sae_dtype = sae.W_enc.dtype
    W_dec = sae.W_dec.detach().float().cpu().numpy()
    d_sae = W_dec.shape[0]
    rows = load_rows(args.prompts)
    print(f"[extract] {args.model} {args.corpus} N={len(rows)} d_sae={d_sae} hs_idx={hs_idx} dev={dev}",
          flush=True)

    pooled = np.zeros((len(rows), d_sae), np.float32)
    top_v = torch.full((5, d_sae), -1.0, device=dev)
    top_doc = torch.full((5, d_sae), -1, dtype=torch.long, device=dev)
    top_pos = torch.full((5, d_sae), -1, dtype=torch.long, device=dev)
    top_tid = torch.full((5, d_sae), -1, dtype=torch.long, device=dev)
    n_tokens = 0
    for i, text in enumerate(rows):
        inp = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True)
            h = out.hidden_states[hs_idx]
            acts = sae.encode(h.to(sae_dtype))[0].float()          # [T, d_sae]
            pooled[i] = acts.mean(0).cpu().numpy()
            T = acts.shape[0]; n_tokens += T
            ids = inp["input_ids"][0]
            cand_v = torch.cat([top_v, acts], 0)
            cand_doc = torch.cat([top_doc, torch.full((T, d_sae), i, dtype=torch.long, device=dev)], 0)
            cand_pos = torch.cat([top_pos, torch.arange(T, device=dev)[:, None].expand(T, d_sae)], 0)
            cand_tid = torch.cat([top_tid, ids[:, None].expand(T, d_sae)], 0)
            k = min(5, cand_v.shape[0])
            v, ix = torch.topk(cand_v, k, dim=0)
            top_v = v; top_doc = cand_doc.gather(0, ix); top_pos = cand_pos.gather(0, ix)
            top_tid = cand_tid.gather(0, ix)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    top_v = top_v.cpu().numpy(); top_tid = top_tid.cpu().numpy()
    top_doc = top_doc.cpu().numpy(); top_pos = top_pos.cpu().numpy()
    uniq = sorted(set(int(t) for t in top_tid.ravel() if t >= 0))
    decode = {str(t): tok.decode([t]) for t in uniq}
    special = sorted(int(s) for s in (tok.all_special_ids or []))
    outdir = Path(args.cache); outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / f"{args.model}__{args.corpus}"
    np.save(f"{stem}__pooled.npy", pooled)
    np.savez(f"{stem}__top5.npz", val=top_v, tid=top_tid, doc=top_doc, pos=top_pos)
    wpath = outdir / f"{args.model}__wdec.npy"
    if not wpath.exists():
        np.save(wpath, W_dec)
    meta = {"model_tag": args.model, "hf_id_loaded": hf_id, "sae_release": release,
            "sae_id": sae_id, "hs_idx": hs_idx, "corpus": args.corpus, "prompts": args.prompts,
            "prompts_sha256": sha256(args.prompts), "n_rows": len(rows), "n_tokens": int(n_tokens),
            "d_sae": int(d_sae), "decode": decode, "special_ids": special,
            "device": str(dev), "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "sae_lens": getattr(__import__("sae_lens"), "__version__", "n/a") if not release.startswith("qwenpt:") else "qwen_sae_adapter",
            "script_sha256": sha256(__file__)}
    json.dump(meta, open(f"{stem}__meta.json", "w"))
    print(f"[extract] wrote {stem}__*  tokens={n_tokens}", flush=True)


# ----------------------------------------------------------------------------- filters
def feature_tokens(top5_tid, top5_val, feat, decode, special, literal=False):
    """Top-5 (activation > 0) token strings of one feature; returns (strings, is_nonconcept)."""
    toks = [int(t) for t, v in zip(top5_tid[:, feat], top5_val[:, feat]) if v > 0 and t >= 0]
    raw = [decode[str(t)] for t in toks]
    if literal:
        strs = raw
        nonc = any(s in NONCONCEPT_LITERAL for s in raw) or any(t in special for t in toks)
    else:
        strs = [s.strip() for s in raw]
        nonc = any(s in NONCONCEPT_STRIPPED for s in strs) or any(t in special for t in toks)
    return set(strs), nonc, len(toks)


def apply_filters(pair, cmax, a_ok, b_ok, a_set, b_set, one2one="code"):
    """pair[i] = B-pool index for A-pool index i; cmax[i] = its correlation.
    Returns kept A-pool indices (in iteration order) and per-step counts."""
    n = len(pair)
    f1 = [i for i in range(n) if a_ok[i] and b_ok[pair[i]] and (a_set[i] & b_set[pair[i]])]
    if one2one == "code":
        counts = Counter(pair.tolist())                  # unfiltered pairing, as in run_expm
        seen, f2 = set(), []
        for i in f1:
            b = int(pair[i])
            if counts[b] == 1:
                f2.append(i)
            elif b not in seen:
                seen.add(b); f2.append(i)
    else:                                                # S2: paper text, drop many-to-1
        c1 = Counter(int(pair[i]) for i in f1)
        f2 = [i for i in f1 if c1[int(pair[i])] == 1]
    f3 = [i for i in f2 if cmax[i] > CORR_FLOOR]
    return f3, {"after_F1": len(f1), "after_F2": len(f2), "after_F3": len(f3)}


def pool_and_pairing(PhiA, PhiB, topk=512):
    """Exactly the pooling/selection/pairing of lan_protocol_audit.lan_statistics."""
    ia = np.argsort(PhiA.var(0))[::-1][:topk]
    ib = np.argsort(PhiB.var(0))[::-1][:topk]
    Za, ka = _z(PhiA[:, ia]); Zb, kb = _z(PhiB[:, ib])
    ia, ib = ia[ka], ib[kb]
    C = (Za.T @ Zb) / Za.shape[0]
    return ia, ib, Za, Zb, C


def filtered_cell(PhiA, PhiB, WdecA, WdecB, topA, topB, metaA, metaB, variant="P", seed_rand=0,
                  seed_perm=1, n_draws=N_DRAWS):
    literal = variant == "S1"
    one2one = "text" if variant == "S2" else "code"
    ia, ib, Za, Zb, C = pool_and_pairing(PhiA, PhiB)
    spA = set(metaA["special_ids"]); spB = set(metaB["special_ids"])
    a_info = [feature_tokens(topA["tid"], topA["val"], f, metaA["decode"], spA, literal) for f in ia]
    b_info = [feature_tokens(topB["tid"], topB["val"], f, metaB["decode"], spB, literal) for f in ib]
    a_ok = [not x[1] for x in a_info]; b_ok = [not x[1] for x in b_info]
    a_set = [x[0] for x in a_info]; b_set = [x[0] for x in b_info]
    pair = C.argmax(1); cmax = C[np.arange(len(pair)), pair]
    kept, steps = apply_filters(pair, cmax, a_ok, b_ok, a_set, b_set, one2one)
    out = {"variant": variant, "pool_A": int(len(ia)), "pool_B": int(len(ib)),
           "nonconcept_A": int(sum(not x for x in a_ok)), "nonconcept_B": int(sum(not x for x in b_ok)),
           "steps": steps, "K_f": len(kept), "retention": len(kept) / len(ia),
           "n_unique_targets_unfiltered": int(len(set(pair.tolist())))}
    if len(kept) < MIN_K:
        out["estimable"] = False
        return out
    out["estimable"] = True
    DA = WdecA[ia[kept]]; DB = WdecB[ib[pair[kept]]]
    out["rsa"] = rsa(DA, DB); out["svcca"] = svcca(DA, DB)
    out["mean_corr_kept"] = float(cmax[kept].mean())
    rng = np.random.default_rng(seed_rand)
    rr, rs = [], []
    for _ in range(n_draws):                               # Lan's shuffle_rand on the kept set
        p = rng.permutation(len(kept))
        rr.append(rsa(DA, DB[p])); rs.append(svcca(DA, DB[p]))
    out["rsa_rand"] = float(np.mean(rr)); out["rsa_rand_sd"] = float(np.std(rr))
    out["svcca_rand"] = float(np.mean(rs)); out["svcca_rand_sd"] = float(np.std(rs))
    rng = np.random.default_rng(seed_perm)
    pr, ps, ks = [], [], []
    for _ in range(n_draws):                               # permuted pairing, whole pipeline
        Cp = (Za.T @ Zb[rng.permutation(Zb.shape[0])]) / Za.shape[0]
        pp = Cp.argmax(1); cp = Cp[np.arange(len(pp)), pp]
        kp, _ = apply_filters(pp, cp, a_ok, b_ok, a_set, b_set, one2one)
        ks.append(len(kp))
        if len(kp) >= MIN_K:
            dA = WdecA[ia[kp]]; dB = WdecB[ib[pp[kp]]]
            pr.append(rsa(dA, dB)); ps.append(svcca(dA, dB))
    out["perm_kept_sizes_median"] = float(np.median(ks))
    out["perm_kept_sizes_q05_q95"] = [float(np.quantile(ks, .05)), float(np.quantile(ks, .95))]
    out["perm_estimable_frac"] = len(pr) / n_draws
    out["perm_estimable"] = len(pr) >= n_draws / 2
    if out["perm_estimable"]:
        out["rsa_perm"] = float(np.mean(pr)); out["rsa_perm_sd"] = float(np.std(pr))
        out["svcca_perm"] = float(np.mean(ps)); out["svcca_perm_sd"] = float(np.std(ps))
        out["rsa_exc"] = out["rsa"] - out["rsa_perm"]
        for s in ("rsa", "svcca"):
            den = out[s] - out[f"{s}_rand"]
            out[f"{s}_share"] = (out[f"{s}_perm"] - out[f"{s}_rand"]) / den if abs(den) > 1e-9 else None
    return out


# ----------------------------------------------------------------------------- analyze
def _clean_meta(m):
    """Extraction metadata for the receipt: no decode table, no absolute (identifying) paths."""
    out = {k: v for k, v in m.items() if k != "decode"}
    out["prompts"] = os.path.basename(out.get("prompts", ""))
    return out


PAIRS = ["gemma2-9b", "qwen3-1.7b", "pythia70m", "gpt2"]
CORPORA = {"templates": "T", "c4": "C"}


def analyze(args):
    cache = Path(args.cache)
    res_dir = Path(args.results)
    receipts_dir = Path(args.receipts)
    for a in (args.pairs.split(",") if args.pairs else PAIRS):
        for corpus, letter in CORPORA.items():
            stemA = cache / f"{a}__{corpus}"; stemB = cache / f"mistral7b__{corpus}"
            if not Path(f"{stemA}__pooled.npy").exists() or not Path(f"{stemB}__pooled.npy").exists():
                print(f"[skip] {a} {corpus}: cache missing"); continue
            inputs = {os.path.basename(p): sha256(p) for p in [f"{stemA}__pooled.npy", f"{stemB}__pooled.npy",
                                             f"{stemA}__top5.npz", f"{stemB}__top5.npz",
                                             str(cache / f"{a}__wdec.npy"), str(cache / "mistral7b__wdec.npy")]}
            PhiA = np.load(f"{stemA}__pooled.npy"); PhiB = np.load(f"{stemB}__pooled.npy")
            n = min(len(PhiA), len(PhiB)); PhiA, PhiB = PhiA[:n], PhiB[:n]
            WdecA = np.load(cache / f"{a}__wdec.npy"); WdecB = np.load(cache / "mistral7b__wdec.npy")
            topA = dict(np.load(f"{stemA}__top5.npz")); topB = dict(np.load(f"{stemB}__top5.npz"))
            metaA = json.load(open(f"{stemA}__meta.json")); metaB = json.load(open(f"{stemB}__meta.json"))
            rec_path = receipts_dir / f"lan{letter}_{a}__mistral7b.json"
            receipt = json.load(open(rec_path))[corpus]
            # G0: unchanged lan_statistics, seed 0, n_base 20 (as the receipts)
            g0 = lan_statistics(PhiA, PhiB, WdecA, WdecB, topk=512, n_base=20, seed=0)
            d_rsa = g0["rsa"] - receipt["rsa"]; d_sv = g0["svcca"] - receipt["svcca"]
            gate = {"local": g0, "receipt": receipt, "receipt_path": rec_path.name,
                    "delta_rsa": d_rsa, "delta_svcca": d_sv,
                    "pass": bool(abs(d_rsa) <= 0.02 and abs(d_sv) <= 0.02)}
            print(f"[G0] {a} {corpus}: rsa {g0['rsa']:.4f} vs {receipt['rsa']:.4f} ; svcca "
                  f"{g0['svcca']:.4f} vs {receipt['svcca']:.4f} -> {'PASS' if gate['pass'] else 'FAIL'}",
                  flush=True)
            unf200 = lan_statistics(PhiA, PhiB, WdecA, WdecB, topk=512, n_base=N_DRAWS, seed=0)
            out = {"pair": f"{a}__mistral7b", "corpus": corpus, "n_docs": int(n),
                   "prereg": "notes/PREREG_lan_filters_20260924.md (commit 4cbd289)",
                   "G0": gate, "unfiltered_n200": unf200, "filtered": {}}
            for v in ("P", "S1", "S2"):
                out["filtered"][v] = filtered_cell(PhiA, PhiB, WdecA, WdecB, topA, topB, metaA, metaB, v)
                f = out["filtered"][v]
                print(f"  [{v}] K_f={f['K_f']} steps={f['steps']} " +
                      (f"rsa={f['rsa']:.4f} rand={f['rsa_rand']:.4f} perm={f.get('rsa_perm')} "
                       f"perm_est={f['perm_estimable_frac']:.2f}" if f["estimable"] else "not estimable"),
                      flush=True)
            out["provenance"] = {"inputs_sha256": inputs, "script": "code/analysis/lan_filter_audit.py",
                                 "script_sha256": sha256(__file__),
                                 "lan_protocol_audit_sha256": sha256(HERE / "lan_protocol_audit.py"),
                                 "seeds": {"G0": 0, "unfiltered_n200": 0, "rand": 0, "perm": 1},
                                 "n_draws": N_DRAWS, "min_k": MIN_K, "corr_floor": CORR_FLOOR,
                                 "meta_A": _clean_meta(metaA), "meta_B": _clean_meta(metaB),
                                 "python": platform.python_version(), "numpy": np.__version__}
            res_dir.mkdir(parents=True, exist_ok=True)
            op = res_dir / f"lanfilt{letter}_{a}__mistral7b.json"
            json.dump(out, open(op, "w"), indent=1)
            print(f"[done] -> {op}", flush=True)


# ----------------------------------------------------------------------------- selftest
def selftest():
    ok = True
    # F1/F2/F3 on a hand-built pairing
    pair = np.array([0, 1, 1, 2, 3, 3])
    cmax = np.array([.5, .5, .4, .05, .6, .6])
    a_ok = [True, True, True, True, False, True]
    b_ok = [True, True, True, True]
    a_set = [{"cat"}, {"dog"}, {"dog"}, {"sun"}, {"x"}, {"moon"}]
    b_set = [{"cat"}, {"dog", "puppy"}, {"sun"}, {"star"}]
    kept, st = apply_filters(pair, cmax, a_ok, b_ok, a_set, b_set, "code")
    # 0 keep; 1,2 -> target 1 hit twice -> keep first (1); 3 -> r=.05 dropped by F3;
    # 4 nonconcept; 5 no shared keyword
    ok &= kept == [0, 1] and st == {"after_F1": 4, "after_F2": 3, "after_F3": 2}
    kept2, _ = apply_filters(pair, cmax, a_ok, b_ok, a_set, b_set, "text")
    ok &= kept2 == [0]                                   # S2 drops both members of target 1
    # token normalisation
    top_tid = np.array([[5, 7], [6, 8], [9, 9], [-1, 9], [-1, 9]])
    top_val = np.array([[1., 1.], [1., 1.], [1., 1.], [0., 1.], [0., 1.]])
    dec = {"5": " the", "6": "cat", "7": "<s>", "8": " dog", "9": "\n"}
    s, nc, n = feature_tokens(top_tid, top_val, 0, dec, {7})
    ok &= s == {"the", "cat", ""} and nc and n == 3      # "\n".strip() == "" -> non-concept
    s, nc, _ = feature_tokens(top_tid, top_val, 1, dec, {7})
    ok &= nc                                             # special id
    s, nc, _ = feature_tokens(top_tid[:2], top_val[:2], 0, dec, {7})
    ok &= (s == {"the", "cat"}) and not nc
    s, nc, _ = feature_tokens(top_tid[:2], top_val[:2], 0, dec, {7}, literal=True)
    ok &= (s == {" the", "cat"}) and not nc
    # planted end-to-end: planted pairs share keywords, noise features do not
    rng = np.random.default_rng(0)
    N, K, d = 400, 60, 32
    lat = rng.standard_normal((N, K))
    PhiA = np.abs(lat + .1 * rng.standard_normal((N, K)))
    perm = rng.permutation(K)
    PhiB = np.abs(lat[:, perm] + .1 * rng.standard_normal((N, K)))
    R, _ = np.linalg.qr(rng.standard_normal((d, d)))
    WA = rng.standard_normal((K, d)); WB = (WA @ R)[perm] + .05 * rng.standard_normal((K, d))
    inv = np.argsort(perm)                              # B feature j <-> A feature perm[j]
    topA = {"tid": np.tile(np.arange(K), (5, 1)), "val": np.ones((5, K))}
    topB = {"tid": np.tile(perm, (5, 1)), "val": np.ones((5, K))}
    dec = {str(i): f" w{i}" for i in range(K)}
    meta = {"decode": dec, "special_ids": []}
    r = filtered_cell(PhiA, PhiB, WA, WB, topA, topB, meta, meta, "P", n_draws=20)
    ok &= r["estimable"] and r["K_f"] >= 50 and r["rsa"] > r["rsa_rand"] + .3
    ok &= r["perm_estimable_frac"] < 0.5                # random pairs rarely share a planted keyword
    _ = inv
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "analyze", "selftest"])
    ap.add_argument("--model"); ap.add_argument("--corpus"); ap.add_argument("--prompts")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--cache", default=os.path.expanduser("~/lanfilt_cache"))
    ap.add_argument("--results", default=str(HERE.parent.parent / "results" / "cross_model_sae"))
    ap.add_argument("--receipts", default=str(HERE.parent.parent / "results" / "cross_model_sae"))
    ap.add_argument("--pairs", default="")
    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(selftest())
    (extract if args.cmd == "extract" else analyze)(args)


if __name__ == "__main__":
    main()

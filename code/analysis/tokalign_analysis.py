#!/usr/bin/env python3
"""Token-aligned / order-sensitive sharing analysis (one cell per invocation).

Implements PREREGISTRATION_token_aligned.md (commit df330c6) EXACTLY; registered
constants are duplicated below so a reader can diff them against the registration.
GPU-optional: torch device auto-detect (the permutation loops are memory-bandwidth
bound; a GPU turns hours into minutes but changes no number's definition).

Pipeline: load two sides' per-token dumps (tokalign_extract.py) ->
  G0  frozen-pair recovery: match the cached top-1500 pooled columns
      (cache/lextok20k_*.npz, md5-pinned in the registration) against this run's
      full-dictionary pooled matrix; r >= 0.999, >= 1490/1500 per side, >= 95% of the
      cell's frozen pairs recovered; plus a sparse-dump integrity assertion.
  G1  alignment QA on registered docs 0-49 + full-corpus doc QA rule.
  ALIGN  whitespace words via offset_mapping; mean-pool token acts per word (primary),
      max-pool (sensitivity); occurrence included iff >=1 token BOTH sides, w != 0,
      w != last, doc QA ok both sides.
  G2  support gate: >= 50 occurrences with either side nonzero per pair; < 100
      surviving pairs -> rung NOT-TESTABLE (gate outcome, not a verdict).
  S1  within-doc-centered frozen-pair correlation vs N1 (within-doc order shuffle,
      n_perm=2000) and N2 (across-doc within relative-position-decile shuffle).
  S2  word-type fixed effects (types with >= 20 occurrences) vs N3 (within-type
      derangement). Headline R_tok is SUPPORT-MATCHED: excess(S2)/excess(S1 computed
      on the S2-retained occurrences and S2-passing pairs); naive ratio also reported.
  Arms (reported, never verdict-bearing): S2c (within-doc centering THEN type
      demeaning -- separates position-level from doc-level survival; amendment logged
      in experiments_detail.md before any receipt existed), bigram cells
      (prev-type, type) >= 10 occurrences, max-pool sensitivity.

--selftest (no GPU, no dumps) runs the FULL pipeline on planted synthetic dumps:
  planted unigram pairs   -> S1 passes, S2 collapses (R_tok ~ 0)
  planted context pairs   -> S1 passes, S2 survives  (R_tok large)
  planted doc-topic pairs -> S1 at null (within-doc centering kills doc-scale signal)
  + G0 recovery on a synthetic dictionary, + doc-QA rejection of corrupted offsets.
"""
import argparse, json, os, re, string, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- registered constants (PREREGISTRATION_token_aligned.md, commit df330c6) ----
PREREG = "df330c6"
N_PERM = 2000
G0_COL_R = 0.999
G0_MIN_COLS = 1490          # of 1500
G0_MIN_PAIR_FRAC = 0.95
G1_QA_DOCS = 50             # registered fixed sample: docs 0-49
G1_CHAR_COV = 0.995
G1_WORD_COV = 0.95
G1_XWORD_MAX = 0.005
G1_DOC_CHAR_COV = 0.99      # per-doc QA rule (full corpus)
G1_MAX_DOC_FAIL = 0.01
G2_MIN_OCC = 50
G2_MIN_PAIRS = 100
S2_TYPE_MIN = 20
BIGRAM_MIN = 10
PUNCT = string.punctuation


# ================================ alignment ================================
def word_spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def norm_type(word):
    return word.lower().strip(PUNCT)


def assign_tokens(spans, off_s, off_e):
    """word id per token (-1 = special/empty/no overlap); + cross-word flag per token."""
    wid = np.full(len(off_s), -1, np.int32)
    xword = np.zeros(len(off_s), bool)
    if not spans:
        return wid, xword
    ws = np.array([s for s, _ in spans]); we = np.array([e for _, e in spans])
    for t in range(len(off_s)):
        s, e = int(off_s[t]), int(off_e[t])
        if e <= s:
            continue                       # special/empty span
        ov = np.minimum(e, we) - np.maximum(s, ws)
        ov = np.maximum(ov, 0)
        if ov.max() <= 0:
            continue                       # pure-whitespace token
        wid[t] = int(np.argmax(ov))        # max overlap, ties -> leftmost
        xword[t] = int((ov > 0).sum()) >= 2
    return wid, xword


def doc_qa(text, off_s, off_e):
    """(ok, char_coverage, n_invalid). Invalid: out-of-bounds spans, or overlap RUNS of
    >= 2 chars. Single-char overlap runs are benign byte-fallback (a multi-byte char --
    e.g. CJK -- split across byte-BPE tokens maps several tokens onto one char span;
    observed on Qwen at smoke, doc 10 char 147; plumbing amendment A2, logged in the ELN
    before any verdict-bearing receipt existed). Byte-fallback chars are counted."""
    n = len(text)
    nonspace = np.array([not c.isspace() for c in text], bool)
    n_ns = int(nonspace.sum())
    count = np.zeros(n, np.int32)
    invalid = 0
    real = [(int(s), int(e)) for s, e in zip(off_s, off_e) if e > s]
    for s, e in real:
        if s < 0 or e > n:
            invalid += 1
            continue
        count[s:e] += 1
    multi = count > 1
    # runs of multiply-covered chars: length 1 -> benign byte-fallback; >= 2 -> invalid
    n_bf = 0
    i = 0
    while i < n:
        if multi[i]:
            j = i
            while j < n and multi[j]:
                j += 1
            if j - i >= 2:
                invalid += 1
            else:
                n_bf += 1
            i = j
        else:
            i += 1
    cov = int((count.astype(bool) & nonspace).sum()) / max(1, n_ns)
    return (invalid == 0 and cov >= G1_DOC_CHAR_COV), cov, invalid


def align_side(texts, meta):
    """meta: dict(doc_idx, tok_pos, off_start, off_end) np arrays over all tokens.
    Returns per-token word ids (global occurrence keying deferred), per-doc QA, stats."""
    n_docs = len(texts)
    doc_of = meta["doc_idx"]
    order = np.argsort(doc_of, kind="stable")
    assert (order == np.arange(len(order))).all(), "token rows must be doc-sorted"
    starts = np.searchsorted(doc_of, np.arange(n_docs), "left")
    ends = np.searchsorted(doc_of, np.arange(n_docs), "right")
    wid = np.full(len(doc_of), -1, np.int32)
    xw = np.zeros(len(doc_of), bool)
    qa_ok = np.zeros(n_docs, bool)
    cov = np.zeros(n_docs)
    for d in range(n_docs):
        sl = slice(starts[d], ends[d])
        w, x = assign_tokens(word_spans(texts[d]), meta["off_start"][sl],
                             meta["off_end"][sl])
        wid[sl] = w; xw[sl] = x
        qa_ok[d], cov[d], _ = doc_qa(texts[d], meta["off_start"][sl],
                                     meta["off_end"][sl])
    return {"wid": wid, "xword": xw, "qa_ok": qa_ok, "cov": cov,
            "starts": starts, "ends": ends}


def g1_report(texts, meta, al, n_qa=G1_QA_DOCS):
    """Registered G1 on docs 0..n_qa-1 for one side."""
    n_tok = n_words = n_wcov = n_x = 0
    xlist = []
    all_valid = True
    for d in range(min(n_qa, len(texts))):
        sl = slice(al["starts"][d], al["ends"][d])
        wid = al["wid"][sl]
        real = (meta["off_end"][sl] > meta["off_start"][sl])
        n_tok += int(real.sum())
        nw = len(word_spans(texts[d]))
        n_words += nw
        n_wcov += len(set(wid[wid >= 0].tolist()))
        xs = np.where(al["xword"][sl])[0]
        n_x += len(xs)
        for t in xs[:20]:
            s, e = int(meta["off_start"][sl][t]), int(meta["off_end"][sl][t])
            xlist.append({"doc": d, "tok": int(t), "span": [s, e],
                          "text": texts[d][s:e]})
        ok, c, inv = doc_qa(texts[d], meta["off_start"][sl], meta["off_end"][sl])
        if inv > 0 or c < G1_CHAR_COV:
            all_valid = False
    word_cov = n_wcov / max(1, n_words)
    x_rate = n_x / max(1, n_tok)
    passed = bool(all_valid and word_cov >= G1_WORD_COV and x_rate <= G1_XWORD_MAX)
    return {"n_docs_qa": min(n_qa, len(texts)), "spans_valid_and_covered": all_valid,
            "word_coverage": word_cov, "crossword_rate": x_rate,
            "crossword_instances": xlist, "passed": passed}


# ============================ per-token dump I/O ============================
def load_side(dirpath):
    """Concatenate all shards of one side. Returns meta arrays + CSR nnz + pooled."""
    d = Path(dirpath)
    shard_files = sorted(d.glob("shard_*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"no shards in {dirpath}")
    rec = json.load(open(d / "extract_receipt.json"))
    metas, feats, vals, pooled = [], [], [], []
    indptr_all = [np.zeros(1, np.int64)]
    off = 0
    for f in shard_files:
        z = np.load(f, allow_pickle=False)
        metas.append({k: z[k] for k in
                      ("doc_idx", "tok_pos", "off_start", "off_end", "input_id")})
        ip = z["indptr"]
        indptr_all.append(ip[1:] + off)
        off += int(ip[-1])
        feats.append(z["feat"]); vals.append(z["val"]); pooled.append(z["pooled"])
    meta = {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}
    return {"meta": meta, "indptr": np.concatenate(indptr_all),
            "feat": np.concatenate(feats), "val": np.concatenate(vals),
            "pooled": np.vstack(pooled), "receipt": rec,
            "d_sae": int(rec["d_sae"]), "n_docs": int(rec["n_docs"])}


# ================================== G0 =====================================
def g0_match(pooled, cached, device="cpu", chunk=8192):
    """Match each cached column (top-1500 pooled matrix) to the full-dictionary
    feature maximizing Pearson r over the shared doc axis. Returns per-column
    best feature, float64-refined r, and second-best r (identifiability margin)."""
    import torch
    N, K = cached.shape
    assert pooled.shape[0] == N, f"doc-axis mismatch {pooled.shape[0]} vs {N}"
    C64 = np.asarray(cached, np.float64)
    Cz = C64 - C64.mean(0); sd = Cz.std(0); sd[sd < 1e-12] = 1.0; Cz /= sd
    Zc = torch.tensor(Cz, dtype=torch.float32, device=device)
    d = pooled.shape[1]
    best_r = np.full(K, -2.0, np.float32)
    best_f = np.full(K, -1, np.int64)
    second_r = np.full(K, -2.0, np.float32)
    for c0 in range(0, d, chunk):
        X = torch.tensor(np.asarray(pooled[:, c0:c0 + chunk], np.float32),
                         device=device)
        mu = X.mean(0, keepdim=True); s = X.std(0, keepdim=True)
        dead = (s < 1e-8).squeeze(0)
        s = torch.where(s < 1e-8, torch.ones_like(s), s)
        Xz = (X - mu) / s
        R = (Zc.T @ Xz) / N                       # [K, F]
        R[:, dead] = -2.0
        top = torch.topk(R, min(2, R.shape[1]), dim=1)
        tv = top.values.cpu().numpy(); ti = top.indices.cpu().numpy() + c0
        for j in range(tv.shape[1]):
            v, i = tv[:, j], ti[:, j]
            newbest = v > best_r
            second_r = np.where(newbest, best_r, np.maximum(second_r, v))
            best_f = np.where(newbest, i, best_f)
            best_r = np.where(newbest, v, best_r)
    # float64 refinement of the winning correlation
    r64 = np.full(K, -2.0)
    P64 = None
    for k in range(K):
        x = np.asarray(pooled[:, best_f[k]], np.float64)
        y = C64[:, k]
        sx, sy = x.std(), y.std()
        if sx < 1e-12 or sy < 1e-12:
            continue
        r64[k] = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
    return best_f, r64, second_r.astype(np.float64)


def run_g0(side, cached, frozen_cols, device):
    best_f, r64, r2nd = g0_match(side["pooled"], cached, device=device)
    matched = r64 >= G0_COL_R
    # injectivity on the matched set
    mf = best_f[matched]
    uniq, cnt = np.unique(mf, return_counts=True)
    dup_feats = set(uniq[cnt > 1].tolist())
    ok_col = matched & ~np.isin(best_f, list(dup_feats) if dup_feats else [])
    n999 = int(matched.sum())
    side_pass = bool(n999 >= G0_MIN_COLS)
    pair_col_ok = ok_col[frozen_cols]
    return {"n_cols": int(len(r64)), "n_cols_matched_999": n999,
            "n_cols_injective": int(ok_col.sum()),
            "min_r_matched": float(r64[matched].min()) if n999 else None,
            "median_margin": float(np.median(r64[matched] - r2nd[matched]))
                             if n999 else None,
            "passed": side_pass}, best_f, ok_col, pair_col_ok


def dump_integrity(side, feat_ids, colmap_pos):
    """Sparse dump must reproduce this run's stored pooled on the given features."""
    meta, indptr = side["meta"], side["indptr"]
    n_docs = side["n_docs"]
    P = len(feat_ids)
    fmap = np.full(side["d_sae"], -1, np.int64)
    fmap[feat_ids] = np.arange(P)
    tok_doc = meta["doc_idx"].astype(np.int64)
    nnz_tok = np.repeat(np.arange(len(tok_doc)), np.diff(indptr))
    f = fmap[side["feat"]]
    keep = f >= 0
    flat = tok_doc[nnz_tok[keep]] * P + f[keep]
    S = np.bincount(flat, weights=side["val"][keep].astype(np.float64),
                    minlength=n_docs * P).reshape(n_docs, P)
    Tdoc = np.bincount(tok_doc, minlength=n_docs).astype(np.float64)
    S /= np.maximum(Tdoc, 1)[:, None]
    ref = np.asarray(side["pooled"][:, feat_ids], np.float64)
    rs = []
    for j in range(P):
        x, y = S[:, j], ref[:, j]
        sx, sy = x.std(), y.std()
        rs.append(1.0 if (sx < 1e-12 and sy < 1e-12) else
                  (0.0 if (sx < 1e-12 or sy < 1e-12) else
                   float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))))
    rs = np.array(rs)
    return {"n_features": P, "min_r": float(rs.min()) if P else None,
            "n_below_999": int((rs < G0_COL_R).sum()),
            "passed": bool(P == 0 or (rs >= G0_COL_R).all())}


# ============================ occurrence building ============================
def build_occurrences(texts, alA, alB):
    """Registered inclusion: >=1 token both sides, w != 0, w != last, doc QA both."""
    occ_doc, occ_w, occ_nw, occ_type, occ_prev = [], [], [], [], []
    type_of = {}
    tid = []
    for d, text in enumerate(texts):
        if not (alA["qa_ok"][d] and alB["qa_ok"][d]):
            continue
        spans = word_spans(text)
        nw = len(spans)
        if nw < 3:
            continue
        ca = np.bincount(alA["wid"][alA["starts"][d]:alA["ends"][d]][
            alA["wid"][alA["starts"][d]:alA["ends"][d]] >= 0], minlength=nw)
        cb = np.bincount(alB["wid"][alB["starts"][d]:alB["ends"][d]][
            alB["wid"][alB["starts"][d]:alB["ends"][d]] >= 0], minlength=nw)
        words = [text[s:e] for s, e in spans]
        for w in range(1, nw - 1):
            if ca[w] == 0 or cb[w] == 0:
                continue
            occ_doc.append(d); occ_w.append(w); occ_nw.append(nw)
            t = norm_type(words[w]); p = norm_type(words[w - 1])
            for key, lst in ((t, occ_type), (p, occ_prev)):
                if key not in type_of:
                    type_of[key] = len(type_of)
                lst.append(type_of[key])
    return {"doc": np.array(occ_doc, np.int64), "widx": np.array(occ_w, np.int32),
            "nwords": np.array(occ_nw, np.int32),
            "type": np.array(occ_type, np.int64), "prev": np.array(occ_prev, np.int64),
            "type_names": {v: k for k, v in type_of.items()}}


def word_matrices(side, al, occ, feat_ids, allowed_docs=None):
    """[n_occ x P] mean- and max-pooled word-level activations for one side.

    AMENDMENT A4(a): `allowed_docs` gates the TOKEN stream. Without it, a document
    outside `occ` whose word index exceeds the radix (occ widx max + 2) can alias into
    an in-`occ` (doc, word) slot via the integer key. Tokens from disallowed documents
    are masked to -1 before any pooling."""
    n_occ = len(occ["doc"])
    P = len(feat_ids)
    fmap = np.full(side["d_sae"], -1, np.int64)
    fmap[feat_ids] = np.arange(P)
    # occurrence id per (doc, word): build lookup
    key = occ["doc"] * (occ["widx"].max() + 2).astype(np.int64) + occ["widx"]
    lut = dict(zip(key.tolist(), range(n_occ)))
    tok_doc = side["meta"]["doc_idx"].astype(np.int64)
    wid = al["wid"].astype(np.int64)
    tok_key = tok_doc * (occ["widx"].max() + 2) + wid
    occ_of_tok = np.array([lut.get(k, -1) if w >= 0 else -1
                           for k, w in zip(tok_key.tolist(), wid.tolist())], np.int64)
    if allowed_docs is not None:
        occ_of_tok[~np.isin(tok_doc, np.asarray(allowed_docs))] = -1
    # token counts per occurrence (denominator of the word mean)
    cnt = np.bincount(occ_of_tok[occ_of_tok >= 0], minlength=n_occ).astype(np.float32)
    nnz_tok = np.repeat(np.arange(len(tok_doc)), np.diff(side["indptr"]))
    f = fmap[side["feat"]]
    o = occ_of_tok[nnz_tok]
    keep = (f >= 0) & (o >= 0)
    o, f, v = o[keep], f[keep], side["val"][keep].astype(np.float32)
    S = np.zeros((n_occ, P), np.float32)
    np.add.at(S, (o, f), v)
    M = np.zeros((n_occ, P), np.float32)
    np.maximum.at(M, (o, f), v)
    S /= np.maximum(cnt, 1)[:, None]
    return S, M


# ============================== statistics core ==============================
def _t():
    import torch
    return torch


def group_mean_center(X, g, n_groups):
    torch = _t()
    S = torch.zeros((n_groups, X.shape[1]), dtype=X.dtype, device=X.device)
    S.index_add_(0, g, X)
    c = torch.zeros(n_groups, dtype=X.dtype, device=X.device)
    c.index_add_(0, g, torch.ones_like(g, dtype=X.dtype))
    mean = S / torch.clamp(c, min=1).unsqueeze(1)
    return X - mean[g]


def within_group_perm(g, gen, device):
    torch = _t()
    n = g.shape[0]
    u1 = torch.rand(n, generator=gen, device=device, dtype=torch.float64)
    u2 = torch.rand(n, generator=gen, device=device, dtype=torch.float64)
    s1 = torch.argsort(g.double() * 2.0 + u1)
    s2 = torch.argsort(g.double() * 2.0 + u2)
    pi = torch.empty(n, dtype=torch.long, device=device)
    pi[s1] = s2
    return pi


def within_group_derangement(g, gen, device):
    """Random CYCLIC derangement within each group: order each group's rows
    uniformly at random, map every row to its successor in that order (cyclic).
    Zero fixed points by construction for groups of size >= 2 (asserted); this
    satisfies the registered 'derangement within word type' exactly and is
    O(n log n) per draw (an iterative fixed-point rejection scheme stalls when
    many groups each hold a single fixed point)."""
    torch = _t()
    n = g.shape[0]
    u = torch.rand(n, generator=gen, device=device, dtype=torch.float64)
    s = torch.argsort(g.double() * 2.0 + u)      # grouped, random order within group
    gs = g[s]
    idx = torch.arange(n, device=device)
    start_mask = torch.ones(n, dtype=torch.bool, device=device)
    start_mask[1:] = gs[1:] != gs[:-1]
    starts = start_mask.nonzero().squeeze(1)
    seg_id = torch.cumsum(start_mask.long(), 0) - 1
    seg_sizes = torch.diff(torch.cat([starts, torch.tensor([n], device=device)]))
    assert int(seg_sizes.min()) >= 2, "derangement group of size 1 (gate should prevent)"
    nxt = idx + 1
    end_mask = torch.zeros(n, dtype=torch.bool, device=device)
    end_mask[:-1] = gs[1:] != gs[:-1]
    end_mask[-1] = True
    nxt = torch.where(end_mask, starts[seg_id], nxt)
    pi = torch.empty(n, dtype=torch.long, device=device)
    pi[s] = s[nxt]
    return pi


def _pair_r(Ac, Bc, na, nb):
    torch = _t()
    s = torch.zeros(Ac.shape[1], dtype=torch.float32, device=Ac.device)
    for c0 in range(0, Ac.shape[1], 256):
        s[c0:c0 + 256] = (Ac[:, c0:c0 + 256] * Bc[:, c0:c0 + 256]).sum(0)
    return s / (na * nb)


def stat_block(Ac, Bc, mask, perm_group_ids, n_perm, gen, derange=False):
    """obs + permutation null of the mean per-pair correlation over `mask` pairs.
    Ac/Bc are CENTERED series; the null permutes Bc's rows within perm_group_ids."""
    torch = _t()
    na = torch.sqrt(torch.clamp((Ac ** 2).sum(0), min=0))
    nb = torch.sqrt(torch.clamp((Bc ** 2).sum(0), min=0))
    valid = (na > 1e-8) & (nb > 1e-8)
    m = mask & valid
    na_ = torch.where(valid, na, torch.ones_like(na))
    nb_ = torch.where(valid, nb, torch.ones_like(nb))
    per_pair = _pair_r(Ac, Bc, na_, nb_)
    per_pair = torch.where(valid, per_pair, torch.zeros_like(per_pair))
    obs = float(per_pair[m].mean()) if int(m.sum()) else float("nan")
    null = np.empty(n_perm, np.float64)
    permfn = within_group_derangement if derange else within_group_perm
    for t in range(n_perm):
        pi = permfn(perm_group_ids, gen, Ac.device)
        pr = _pair_r(Ac, Bc[pi], na_, nb_)
        pr = torch.where(valid, pr, torch.zeros_like(pr))
        null[t] = float(pr[m].mean()) if int(m.sum()) else float("nan")
    return {"obs": obs, "null": null, "per_pair": per_pair.cpu().numpy(),
            "n_pairs_scored": int(m.sum()), "mask": m}


def null_summary(obs, null, n_perm):
    return {"mean": float(np.mean(null)), "sd": float(np.std(null, ddof=1)),
            "ci95": [float(np.percentile(null, 2.5)),
                     float(np.percentile(null, 97.5))],
            "p": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
            "n_perm": int(n_perm)}


def run_stats(Am, Bm, Ax, Bx, occ, cfg):
    """Full registered statistics battery on the occurrence matrices of one cell."""
    torch = _t()
    dev = cfg["device"]
    n_perm, n_sens, seed = cfg["n_perm"], cfg["n_perm_sens"], cfg["seed"]
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    A = torch.tensor(Am, device=dev); B = torch.tensor(Bm, device=dev)
    doc = torch.tensor(occ["doc"], device=dev)
    n_docs = int(doc.max()) + 1 if len(occ["doc"]) else 1
    typ = torch.tensor(occ["type"], device=dev)
    res = {}

    # ---- G2(S1) support ----
    sup1 = ((A > 0) | (B > 0)).float().sum(0)
    mask1 = sup1 >= cfg["g2_min_occ"]
    res["g2_s1"] = {"n_pairs_in": int(A.shape[1]),
                    "n_pairs_pass": int(mask1.sum()),
                    "min_occ": cfg["g2_min_occ"],
                    "testable": bool(int(mask1.sum()) >= cfg["g2_min_pairs"])}

    # ---- S1 vs N1 ----
    Ac = group_mean_center(A, doc, n_docs)
    Bc = group_mean_center(B, doc, n_docs)
    s1 = stat_block(Ac, Bc, mask1, doc, n_perm, gen)
    res["s1"] = {"obs_mean_pair_r": s1["obs"],
                 "null_n1": null_summary(s1["obs"], s1["null"], n_perm),
                 "excess": float(s1["obs"] - s1["null"].mean()),
                 "per_pair_quartiles": [float(q) for q in np.percentile(
                     s1["per_pair"][s1["mask"].cpu().numpy()], [25, 50, 75])]
                     if int(s1["mask"].sum()) else None}
    # ---- N2 (position control): permute centered B across docs within pos-decile ----
    posbin = torch.tensor(np.minimum(
        (10 * occ["widx"].astype(np.int64)) // np.maximum(occ["nwords"], 1), 9),
        device=dev)
    s1n2 = stat_block(Ac, Bc, mask1, posbin, n_perm, gen)
    res["s1"]["null_n2"] = null_summary(s1n2["obs"], s1n2["null"], n_perm)
    res["s1"]["excess_n2"] = float(s1n2["obs"] - s1n2["null"].mean())

    # ---- S2 support: types with >= S2_TYPE_MIN occurrences, nonempty ----
    tc = np.bincount(occ["type"], minlength=int(occ["type"].max()) + 1
                     if len(occ["type"]) else 1)
    names = occ["type_names"]
    nonempty = np.array([names[i] != "" for i in range(len(tc))])
    keep_type = (tc >= cfg["type_min"]) & nonempty
    ret = torch.tensor(keep_type[occ["type"]], device=dev)
    ridx = ret.nonzero().squeeze(1)
    Ar, Br = A[ridx], B[ridx]
    tr = typ[ridx]
    dr = doc[ridx]
    # contiguous type ids on the retained subset
    tr_u, tr_c = torch.unique(tr, return_inverse=True)
    sup2 = ((Ar > 0) | (Br > 0)).float().sum(0)
    mask2 = sup2 >= cfg["g2_min_occ"]
    res["g2_s2"] = {"n_pairs_in": int(A.shape[1]),
                    "n_pairs_pass": int(mask2.sum()),
                    "min_occ": cfg["g2_min_occ"],
                    "testable": bool(int(mask2.sum()) >= cfg["g2_min_pairs"])}
    res["s2_support"] = {"n_types_retained": int(keep_type.sum()),
                         "n_occ_retained": int(len(ridx)),
                         "n_occ_total": int(A.shape[0])}

    # ---- S2 vs N3 (type FE, derangement within type) ----
    At = group_mean_center(Ar, tr_c, len(tr_u))
    Bt = group_mean_center(Br, tr_c, len(tr_u))
    s2 = stat_block(At, Bt, mask2, tr_c, n_perm, gen, derange=True)
    res["s2"] = {"obs_mean_pair_r": s2["obs"],
                 "null_n3": null_summary(s2["obs"], s2["null"], n_perm),
                 "excess": float(s2["obs"] - s2["null"].mean())}

    # ---- S1 on the S2 support (denominator of the registered R_tok) ----
    Ar_c = group_mean_center(Ar, dr, n_docs)
    Br_c = group_mean_center(Br, dr, n_docs)
    s1r = stat_block(Ar_c, Br_c, mask2, dr, n_perm, gen)
    res["s2"]["s1_on_s2_support"] = {
        "obs_mean_pair_r": s1r["obs"],
        "null_n1": null_summary(s1r["obs"], s1r["null"], n_perm),
        "excess": float(s1r["obs"] - s1r["null"].mean())}
    den = res["s2"]["s1_on_s2_support"]["excess"]
    num = res["s2"]["excess"]
    res["s2"]["R_tok"] = float(num / den) if den > 0 else float("nan")
    den_n = res["s1"]["excess"]
    res["s2"]["R_tok_naive"] = float(num / den_n) if den_n > 0 else float("nan")

    # ---- S2c arm (amendment, reported): doc-centering THEN type demeaning ----
    At2 = group_mean_center(Ar_c, tr_c, len(tr_u))
    Bt2 = group_mean_center(Br_c, tr_c, len(tr_u))
    s2c = stat_block(At2, Bt2, mask2, tr_c, n_sens, gen, derange=True)
    res["s2c"] = {"obs_mean_pair_r": s2c["obs"],
                  "null_n3": null_summary(s2c["obs"], s2c["null"], n_sens),
                  "excess": float(s2c["obs"] - s2c["null"].mean())}

    # ---- bigram narrowing arm (reported) ----
    big = occ["prev"] * (int(occ["type"].max()) + 2) + occ["type"]
    bu, binv, bc = np.unique(big, return_inverse=True, return_counts=True)
    keep_bg = (bc >= cfg["bigram_min"])[binv]
    bgi = torch.tensor(np.where(keep_bg)[0], device=dev)
    if len(bgi) > 0:
        Ab, Bb = A[bgi], B[bgi]
        bg_ids = torch.tensor(binv[keep_bg], device=dev)
        bg_u, bg_c = torch.unique(bg_ids, return_inverse=True)
        Abt = group_mean_center(Ab, bg_c, len(bg_u))
        Bbt = group_mean_center(Bb, bg_c, len(bg_u))
        supb = ((Ab > 0) | (Bb > 0)).float().sum(0)
        maskb = supb >= cfg["g2_min_occ"]
        sb = stat_block(Abt, Bbt, maskb, bg_c, n_sens, gen, derange=True)
        res["bigram"] = {"n_cells": int(len(bg_u)), "n_occ": int(len(bgi)),
                         "n_pairs_scored": sb["n_pairs_scored"],
                         "obs_mean_pair_r": sb["obs"],
                         "null": null_summary(sb["obs"], sb["null"], n_sens),
                         "excess": float(sb["obs"] - sb["null"].mean())}
    else:
        res["bigram"] = {"n_cells": 0, "note": "no bigram cell reached min occupancy"}

    # ---- max-pool sensitivity (reported) ----
    Amx = torch.tensor(Ax, device=dev); Bmx = torch.tensor(Bx, device=dev)
    Axc = group_mean_center(Amx, doc, n_docs)
    Bxc = group_mean_center(Bmx, doc, n_docs)
    sx1 = stat_block(Axc, Bxc, mask1, doc, n_sens, gen)
    Axr, Bxr = Amx[ridx], Bmx[ridx]
    Axt = group_mean_center(Axr, tr_c, len(tr_u))
    Bxt = group_mean_center(Bxr, tr_c, len(tr_u))
    sx2 = stat_block(Axt, Bxt, mask2, tr_c, n_sens, gen, derange=True)
    res["maxpool"] = {
        "s1": {"obs_mean_pair_r": sx1["obs"],
               "null_n1": null_summary(sx1["obs"], sx1["null"], n_sens),
               "excess": float(sx1["obs"] - sx1["null"].mean())},
        "s2": {"obs_mean_pair_r": sx2["obs"],
               "null_n3": null_summary(sx2["obs"], sx2["null"], n_sens),
               "excess": float(sx2["obs"] - sx2["null"].mean())}}
    return res


# ================================ selftest =================================
def _synth_side(docs_words, split_long=False, corrupt_doc=None):
    """Fake tokenizer: 1 token/word (A) or 2 tokens for words len>4 (B), + BOS."""
    doc_idx, tok_pos, off_s, off_e, iid = [], [], [], [], []
    for d, words in enumerate(docs_words):
        text = " ".join(words)
        spans = word_spans(text)
        t = 0
        doc_idx.append(d); tok_pos.append(t); off_s.append(0); off_e.append(0)
        iid.append(1); t += 1                                # BOS, empty span
        for (s, e) in spans:
            if split_long and e - s > 4:
                mid = s + (e - s) // 2
                for (a, b) in ((s, mid), (mid, e)):
                    doc_idx.append(d); tok_pos.append(t)
                    off_s.append(a); off_e.append(b); iid.append(7); t += 1
            else:
                doc_idx.append(d); tok_pos.append(t)
                off_s.append(s); off_e.append(e); iid.append(5); t += 1
        if corrupt_doc is not None and d == corrupt_doc:
            off_e[-1] = off_s[-1]  # kill the last token's span -> coverage drops
    return {"doc_idx": np.array(doc_idx, np.int32),
            "tok_pos": np.array(tok_pos, np.int32),
            "off_start": np.array(off_s, np.int32),
            "off_end": np.array(off_e, np.int32),
            "input_id": np.array(iid, np.int32)}


def _synth_acts(meta, docs_words, kind, n_pairs, d_sae, struct_seed, noise_seed,
                gain=3.0):
    """Dense per-token activation matrix [n_tok x d_sae] for planted features 0..P-1.
    The LATENT structure (trigger sets, doc loadings) comes from struct_seed (shared
    across the two sides of a pair); the measurement noise from noise_seed (independent
    per side -- so only genuinely shared structure can correlate)."""
    srng = np.random.default_rng(struct_seed)
    rng = np.random.default_rng(noise_seed)
    n_tok = len(meta["doc_idx"])
    A = np.zeros((n_tok, d_sae), np.float32)
    vocab = sorted({w for ws in docs_words for w in ws})
    trig = {p: set(srng.choice(vocab, 5, replace=False).tolist())
            for p in range(n_pairs)}
    zdoc = {d: srng.uniform(0.5, 2.0) for d in range(len(docs_words))}
    for r in range(n_tok):
        d = int(meta["doc_idx"][r])
        s, e = int(meta["off_start"][r]), int(meta["off_end"][r])
        if e <= s:
            continue
        text = " ".join(docs_words[d])
        spans = word_spans(text)
        ov = [max(0, min(e, we) - max(s, ws)) for ws, we in spans]
        w = int(np.argmax(ov)) if max(ov) > 0 else -1
        if w < 0:
            continue
        word = docs_words[d][w]
        prev = docs_words[d][w - 1] if w > 0 else ""
        for p in range(n_pairs):
            if kind == "unigram":
                base = gain if word in trig[p] else 0.0
            elif kind == "context":
                base = gain if prev in trig[p] else 0.0
            else:  # doctopic
                base = zdoc[d]
            if base > 0:
                A[r, p] = base * (1.0 + 0.05 * rng.standard_normal())
    return np.maximum(A, 0.0)


def _dense_to_side(meta, A):
    """Wrap a dense [n_tok x d] activation matrix in the dump format."""
    n_tok, d_sae = A.shape
    indptr = [0]; feat = []; val = []
    for r in range(n_tok):
        nz = np.where(A[r] > 0)[0]
        feat.append(nz.astype(np.int32)); val.append(A[r, nz].astype(np.float16))
        indptr.append(indptr[-1] + len(nz))
    n_docs = int(meta["doc_idx"].max()) + 1
    pooled = np.zeros((n_docs, d_sae), np.float32)
    for d in range(n_docs):
        pooled[d] = A[meta["doc_idx"] == d].mean(0)
    return {"meta": meta, "indptr": np.array(indptr, np.int64),
            "feat": np.concatenate(feat), "val": np.concatenate(val),
            "pooled": pooled.astype(np.float16), "d_sae": d_sae, "n_docs": n_docs,
            "receipt": {"d_sae": d_sae, "n_docs": n_docs}}


def selftest():
    import torch  # noqa
    rng = np.random.default_rng(0)
    ok = {}
    n_docs, n_words, P, D = 240, 30, 24, 200
    # variable-length vocab: long words get split into 2 tokens on side B, so the
    # cross-tokenizer alignment path (multi-token words, mean-pooling) is exercised
    vocab = [f"w{i:03d}" if i % 2 == 0 else f"wordform{i:03d}" for i in range(60)]
    probs = np.array([1.0 / (i + 1) for i in range(60)]); probs /= probs.sum()
    docs = [[str(v) for v in rng.choice(vocab, n_words, p=probs)]
            for _ in range(n_docs)]
    texts = [" ".join(ws) for ws in docs]
    cfg = {"n_perm": 200, "n_perm_sens": 100, "seed": 0, "device": "cpu",
           "g2_min_occ": 30, "g2_min_pairs": 10, "type_min": 10, "bigram_min": 5}
    floor = 1.0 / (cfg["n_perm"] + 1)

    def run_case(kind):
        mA = _synth_side(docs, split_long=False)
        mB = _synth_side(docs, split_long=True)
        # SAME planted latent (struct_seed 42), INDEPENDENT noise per side
        A = _synth_acts(mA, docs, kind, P, D, struct_seed=42, noise_seed=1)
        B = _synth_acts(mB, docs, kind, P, D, struct_seed=42, noise_seed=2)
        sa, sb = _dense_to_side(mA, A), _dense_to_side(mB, B)
        alA = align_side(texts, sa["meta"]); alB = align_side(texts, sb["meta"])
        occ = build_occurrences(texts, alA, alB)
        fa = np.arange(P)
        Am, Ax = word_matrices(sa, alA, occ, fa)
        Bm, Bx = word_matrices(sb, alB, occ, fa)
        return run_stats(Am, Bm, Ax, Bx, occ, cfg), alA, alB, occ

    # CASE A: unigram detectors -> S1 passes, S2 collapses
    rA, alA, alB, occ = run_case("unigram")
    ok["A_s1_passes"] = (rA["s1"]["null_n1"]["p"] <= floor * 1.001
                         and rA["s1"]["obs_mean_pair_r"] >= 0.05)
    ok["A_s2_collapses"] = (np.isnan(rA["s2"]["R_tok"])
                            or rA["s2"]["R_tok"] < 0.10)
    print(f"CASE A unigram   S1 obs={rA['s1']['obs_mean_pair_r']:.3f} "
          f"p={rA['s1']['null_n1']['p']:.4f} | S2 excess={rA['s2']['excess']:+.4f} "
          f"R_tok={rA['s2']['R_tok']:.3f} (want S1 pass, R_tok<0.10)")

    # CASE B: context detectors -> S1 passes, S2 survives
    rB, _, _, _ = run_case("context")
    ok["B_s1_passes"] = (rB["s1"]["null_n1"]["p"] <= floor * 1.001
                         and rB["s1"]["obs_mean_pair_r"] >= 0.05)
    ok["B_s2_survives"] = (rB["s2"]["R_tok"] >= 0.30
                           and rB["s2"]["null_n3"]["p"] <= floor * 1.001)
    print(f"CASE B context   S1 obs={rB['s1']['obs_mean_pair_r']:.3f} "
          f"p={rB['s1']['null_n1']['p']:.4f} | S2 excess={rB['s2']['excess']:+.4f} "
          f"R_tok={rB['s2']['R_tok']:.3f} p={rB['s2']['null_n3']['p']:.4f} "
          f"(want both rungs pass)")

    # CASE C: doc-topic detectors -> S1 at null (within-doc centering kills it)
    rC, _, _, _ = run_case("doctopic")
    band = rC["s1"]["null_n1"]["ci95"]
    ok["C_s1_at_null"] = band[0] <= rC["s1"]["obs_mean_pair_r"] <= band[1]
    print(f"CASE C doctopic  S1 obs={rC['s1']['obs_mean_pair_r']:.4f} "
          f"band=[{band[0]:.4f},{band[1]:.4f}] (want inside)")

    # G0 recovery on a synthetic dictionary
    mA = _synth_side(docs, split_long=False)
    r3 = np.random.default_rng(7)
    Afull = r3.standard_normal((len(mA["doc_idx"]), D)).astype(np.float32)
    Afull = np.maximum(Afull, 0)
    sa = _dense_to_side(mA, Afull)
    cols = r3.choice(D, 40, replace=False)
    cached = np.asarray(sa["pooled"][:, cols], np.float64)
    cached += 1e-4 * r3.standard_normal(cached.shape)      # r stays > 0.999
    best_f, r64, _ = g0_match(sa["pooled"], cached, device="cpu", chunk=64)
    ok["G0_exact_recovery"] = bool((best_f == cols).all() and (r64 >= G0_COL_R).all())
    print(f"G0 synthetic     recovered {(best_f == cols).sum()}/40 columns, "
          f"min r64={r64.min():.6f} (want 40/40, all >= {G0_COL_R})")
    integ = dump_integrity(sa, cols[:10], None)
    ok["dump_integrity"] = integ["passed"]
    print(f"dump integrity   min r={integ['min_r']:.6f} passed={integ['passed']}")

    # G1: clean side passes; corrupted offsets fail the doc-QA rule
    alA = align_side(texts, mA)
    g1 = g1_report(texts, mA, alA)
    mBad = _synth_side(docs, split_long=False, corrupt_doc=3)
    alBad = align_side(texts, mBad)
    ok["G1_clean_passes"] = g1["passed"]
    ok["G1_corrupt_fails_doc"] = not bool(alBad["qa_ok"][3]) and bool(alBad["qa_ok"][4])
    print(f"G1 clean passed={g1['passed']} wordcov={g1['word_coverage']:.3f} "
          f"xword={g1['crossword_rate']:.4f} | corrupted doc3 qa_ok="
          f"{bool(alBad['qa_ok'][3])} (want False), doc4={bool(alBad['qa_ok'][4])}")

    # amendment A2: single-char overlap (byte fallback on a multi-byte char) is benign;
    # an overlap RUN of >= 2 chars is still invalid
    tqa = "abc def"
    ok_bf, cov_bf, _ = doc_qa(tqa, np.array([0, 4, 4]), np.array([3, 7, 5]))
    ok_bad, _, _ = doc_qa(tqa, np.array([0, 4, 4]), np.array([3, 7, 6]))
    ok["A2_bytefallback_benign"] = bool(ok_bf and cov_bf == 1.0)
    ok["A2_multichar_overlap_invalid"] = not ok_bad
    print(f"A2 byte-fallback single-char overlap ok={ok_bf} (want True) | "
          f"2-char overlap ok={ok_bad} (want False)")

    print()
    for k, v in ok.items():
        print(f"PASS {k} = {v}")
    good = all(ok.values())
    print("SELFTEST", "PASS" if good else "FAIL",
          "(pipeline can return every registered rung outcome)" if good else "")
    return 0 if good else 1


# ================================== main ===================================
def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="tag, e.g. qwen3-1.7b__gemma2-2b")
    ap.add_argument("--dir_a", required=True); ap.add_argument("--dir_b", required=True)
    ap.add_argument("--pairs_receipt", required=True)
    ap.add_argument("--cache_npz", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    ap.add_argument("--n_perm_sens", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing run: relaxed doc-count assert, receipt tagged smoke")
    args = ap.parse_args()

    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    from tokalign_extract import load_docs
    sa = load_side(args.dir_a); sb = load_side(args.dir_b)
    n_docs = min(sa["n_docs"], sb["n_docs"])
    assert sa["n_docs"] == sb["n_docs"], "sides extracted different doc counts"
    if not args.smoke:
        assert n_docs == 20000, f"full run requires 20000 docs, got {n_docs}"
    texts = load_docs(args.corpus, "en", n_docs)

    rec = json.load(open(args.pairs_receipt))
    fp = rec["frozen_pairs"]
    cols_a = np.array(fp["feat_a"], np.int64)
    cols_b = np.array(fp["feat_b"], np.int64)
    z = np.load(args.cache_npz)
    Pa_c, Pb_c = z["Pa"][:n_docs], z["Pb"][:n_docs]
    print(f"[in] cell={args.cell} n_docs={n_docs} frozen_pairs={fp['n']} "
          f"cache Pa{Pa_c.shape} Pb{Pb_c.shape} device={dev}", flush=True)

    # ---------------- G0 ----------------
    g0a, map_a, okcol_a, pair_ok_a = run_g0(sa, Pa_c, cols_a, dev)
    g0b, map_b, okcol_b, pair_ok_b = run_g0(sb, Pb_c, cols_b, dev)
    pair_ok = pair_ok_a & pair_ok_b
    frac = float(pair_ok.mean())
    g0_pass = bool(g0a["passed"] and g0b["passed"] and frac >= G0_MIN_PAIR_FRAC)
    feat_a = map_a[cols_a]; feat_b = map_b[cols_b]
    print(f"[G0] A cols@999={g0a['n_cols_matched_999']}/1500 "
          f"B cols@999={g0b['n_cols_matched_999']}/1500 "
          f"pairs recovered={int(pair_ok.sum())}/{len(pair_ok)} ({frac:.3f}) "
          f"-> {'PASS' if g0_pass else 'FAIL'}", flush=True)
    ia = dump_integrity(sa, np.unique(feat_a[pair_ok]), None)
    ib = dump_integrity(sb, np.unique(feat_b[pair_ok]), None)
    assert ia["passed"] and ib["passed"], \
        f"dump-integrity failed (A min_r={ia['min_r']}, B min_r={ib['min_r']}) -- code bug"
    gates = {"G0": {"side_a": g0a, "side_b": g0b,
                    "n_frozen_pairs": int(len(pair_ok)),
                    "n_recovered_pairs": int(pair_ok.sum()),
                    "frac_recovered": frac, "min_pair_frac": G0_MIN_PAIR_FRAC,
                    "dump_integrity_a": ia, "dump_integrity_b": ib,
                    "passed": g0_pass}}

    # ---------------- G1 + alignment ----------------
    alA = align_side(texts, sa["meta"]); alB = align_side(texts, sb["meta"])
    g1a = g1_report(texts, sa["meta"], alA); g1b = g1_report(texts, sb["meta"], alB)
    frac_fail = float(1.0 - (alA["qa_ok"] & alB["qa_ok"]).mean())
    g1_pass = bool(g1a["passed"] and g1b["passed"] and frac_fail <= G1_MAX_DOC_FAIL)
    print(f"[G1] A wordcov={g1a['word_coverage']:.3f} xw={g1a['crossword_rate']:.4f} "
          f"| B wordcov={g1b['word_coverage']:.3f} xw={g1b['crossword_rate']:.4f} "
          f"| doc-fail={frac_fail:.4f} -> {'PASS' if g1_pass else 'FAIL'}", flush=True)
    gates["G1"] = {"side_a": g1a, "side_b": g1b, "frac_docs_failed": frac_fail,
                   "passed": g1_pass}

    out = {"cell": f"tokalign20k_{args.cell}.json", "tag": args.cell,
           "prereg_commit": PREREG, "smoke": bool(args.smoke),
           "model_a": rec["model_a"], "model_b": rec["model_b"],
           "sae_a": rec["sae_a"], "sae_b": rec["sae_b"],
           "n_docs": int(n_docs), "n_perm": args.n_perm,
           "n_perm_sens": args.n_perm_sens, "seed": args.seed, "device": dev,
           "pairs_receipt": os.path.basename(args.pairs_receipt),
           "gates": gates}

    if not (g0_pass and g1_pass):
        # registered: G0/G1 failure -> stop, nothing downstream interpretable
        out["gates"]["G2"] = {"s1": {"testable": False, "n_pairs_pass": 0,
                                     "n_pairs_in": int(len(pair_ok))},
                              "s2": {"testable": False, "n_pairs_pass": 0,
                                     "n_pairs_in": int(len(pair_ok))}}
        Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)
        print(f"[GATE] G0/G1 FAILED -- receipt written, no statistics computed -> "
              f"{args.output}", file=sys.stderr)
        return 3

    # ---------------- occurrences + matrices ----------------
    occ = build_occurrences(texts, alA, alB)
    print(f"[align] {len(occ['doc'])} aligned occurrences "
          f"({len(occ['type_names'])} word types)", flush=True)
    keep = pair_ok
    fa, fb = feat_a[keep], feat_b[keep]
    ufa, inv_a = np.unique(fa, return_inverse=True)
    ufb, inv_b = np.unique(fb, return_inverse=True)
    Am_u, Ax_u = word_matrices(sa, alA, occ, ufa)
    Bm_u, Bx_u = word_matrices(sb, alB, occ, ufb)
    Am, Ax = Am_u[:, inv_a], Ax_u[:, inv_a]
    Bm, Bx = Bm_u[:, inv_b], Bx_u[:, inv_b]
    print(f"[matrices] A {Am.shape} B {Bm.shape}", flush=True)

    cfg = {"n_perm": args.n_perm, "n_perm_sens": args.n_perm_sens, "seed": args.seed,
           "device": dev, "g2_min_occ": G2_MIN_OCC, "g2_min_pairs": G2_MIN_PAIRS,
           "type_min": S2_TYPE_MIN, "bigram_min": BIGRAM_MIN}
    res = run_stats(Am, Bm, Ax, Bx, occ, cfg)

    out["gates"]["G2"] = {"s1": res.pop("g2_s1"), "s2": res.pop("g2_s2")}
    out["alignment"] = {"n_occurrences": int(len(occ["doc"])),
                        "n_word_types": int(len(occ["type_names"])),
                        **res.pop("s2_support")}
    out.update(res)
    out["elapsed_s"] = round(time.time() - t0, 1)
    if args.smoke:
        d, b = os.path.split(args.output)
        if not b.startswith("smoke_"):
            args.output = os.path.join(d, "smoke_" + b)
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    chk = json.load(open(args.output))
    assert chk["gates"]["G0"]["n_recovered_pairs"] == int(pair_ok.sum())
    s1, s2 = out["s1"], out["s2"]
    print(f"[S1] obs={s1['obs_mean_pair_r']:.4f} N1 p={s1['null_n1']['p']:.4f} "
          f"excess={s1['excess']:+.4f} | N2 excess={s1['excess_n2']:+.4f}", flush=True)
    print(f"[S2] obs={s2['obs_mean_pair_r']:.4f} N3 p={s2['null_n3']['p']:.4f} "
          f"excess={s2['excess']:+.4f} R_tok={s2['R_tok']:.3f} "
          f"(naive {s2['R_tok_naive']:.3f})", flush=True)
    print(f"[done] -> {args.output} ({out['elapsed_s']}s)", flush=True)
    nt = (not out["gates"]["G2"]["s1"]["testable"]
          or not out["gates"]["G2"]["s2"]["testable"])
    return 2 if nt else 0


if __name__ == "__main__":
    sys.exit(main())

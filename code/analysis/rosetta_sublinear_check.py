#!/usr/bin/env python3
"""plan-CT-2: Rosetta sublinear power-law check on cross-lingual probe-attribution
alignment (mean_pairwise_r) vs model scale, within and across Qwen3/Gemma3 families.

Tests the hypothesis (from Neuron Populations Divergent Selectivity Scale, 2606.03990,
cited in discussion.tex) that cross-model universal/aligned representation follows a
SUBLINEAR power law with parameter count, and that architecture sets the level more
than scale. Uses ONLY local results/probe_attribution_v2/*.json (no GPU/SSH).
"""
import json, glob, os, math
import numpy as np

V2 = "results/probe_attribution_v2"
# nominal param counts in billions
PARAMS = {
    "Qwen/Qwen3-0.6B":0.6, "Qwen/Qwen3-1.7B":1.7, "Qwen/Qwen3-4B":4.0,
    "Qwen/Qwen3-8B":8.0, "Qwen/Qwen3-14B":14.0,
    "google/gemma-3-1b-it":1.0, "google/gemma-3-4b-it":4.0, "google/gemma-3-12b-it":12.0, "google/gemma-3-27b-it":27.0,
    "EleutherAI/pythia-70m-deduped":0.07, "EleutherAI/pythia-160m-deduped":0.16,
    "EleutherAI/pythia-410m-deduped":0.41, "EleutherAI/pythia-1b-deduped":1.0,
    "EleutherAI/pythia-1.4b-deduped":1.4,
}
rows=[]
for fp in sorted(glob.glob(f"{V2}/*.json")):
    d=json.load(open(fp))
    m=d["model"]; rows.append(dict(model=m, family=("Qwen3" if "Qwen" in m else "Pythia" if "pythia" in m.lower() else "Gemma3"),
        params=PARAMS[m], r=float(d["mean_pairwise_r"]),
        npos=d["n_positive_pairs"], ntot=d["n_pairs_total"], n_layers=d["n_layers"]))

def powerfit(pts):
    # r = c * N^b  ->  log r = log c + b log N  (OLS on log-log); pts=[(N,r)...] r>0
    x=np.log(np.array([p[0] for p in pts])); y=np.log(np.array([p[1] for p in pts]))
    b,logc=np.polyfit(x,y,1)
    yhat=b*x+logc; ss_res=np.sum((y-yhat)**2); ss_tot=np.sum((y-y.mean())**2)
    r2=1-ss_res/ss_tot if ss_tot>0 else float('nan')
    return b, math.exp(logc), r2

def fam(fam_name, exclude=()):
    pts=[(r["params"],r["r"]) for r in rows if r["family"]==fam_name and r["model"] not in exclude and r["r"]>0]
    pts=sorted(pts)
    return pts

out={"rows":rows, "fits":{}}
print("=== raw alignment vs scale ===")
for r in sorted(rows,key=lambda z:(z["family"],z["params"])):
    print(f"  {r['family']:7s} {r['params']:5.1f}B  r={r['r']:+.3f}  npos={r['npos']}/{r['ntot']}  L={r['n_layers']}")

# Qwen3 primary (exclude 14B small-positive-pair count) + sensitivity (incl 14B)
q_excl = fam("Qwen3", exclude=("Qwen/Qwen3-14B",))
q_incl = fam("Qwen3")
g_all  = fam("Gemma3")
p_all  = fam("Pythia")
for label,pts in [("Qwen3_excl14B",q_excl),("Qwen3_incl14B",q_incl),("Gemma3",g_all),("Pythia",p_all)]:
    if len(pts)>=2:
        b,c,r2=powerfit(pts)
        out["fits"][label]=dict(exponent_b=b,coef_c=c,r2=r2,n=len(pts),
            params=[p[0] for p in pts], r=[p[1] for p in pts])
        print(f"\n[{label}] power-law r=c*N^b : b={b:+.3f}  c={c:.3f}  R^2={r2:.3f}  (n={len(pts)})")
        print(f"   interpretation: {'SUBLINEAR (b<1, diminishing returns)' if b<1 else 'super/linear'}; "
              f"params {pts[-1][0]/pts[0][0]:.1f}x -> r {pts[-1][1]/pts[0][1]:.2f}x")

# Cross-family: smallest Gemma vs every Qwen3
gemma_min = min((r for r in rows if r["family"]=="Gemma3"), key=lambda z:z["params"])
qmax = max((r for r in rows if r["family"]=="Qwen3"), key=lambda z:z["r"])
out["cross_family"]=dict(gemma_min_model=gemma_min["model"], gemma_min_r=gemma_min["r"],
    gemma_min_params=gemma_min["params"], qwen_best_model=qmax["model"], qwen_best_r=qmax["r"],
    gemma4b_exceeds_all_qwen=all(gemma_min["r"]>r["r"] for r in rows if r["family"]=="Qwen3"))
print(f"\n=== cross-family ===")
print(f"  Gemma-{gemma_min['params']:.0f}B r={gemma_min['r']:+.3f}  vs  best Qwen3 ({qmax['model'].split('/')[-1]}) r={qmax['r']:+.3f}")
print(f"  Gemma-4B exceeds ALL Qwen3 sizes: {out['cross_family']['gemma4b_exceeds_all_qwen']}")
print(f"  scale-vs-architecture: Gemma-4B ({gemma_min['r']:.3f}) / Qwen3-14B ({[r['r'] for r in rows if r['model']=='Qwen/Qwen3-14B'][0]:.3f}) "
      f"= {gemma_min['r']/[r['r'] for r in rows if r['model']=='Qwen/Qwen3-14B'][0]:.1f}x at 1/3.5 the params")

json.dump(out, open("results/rosetta_check/rosetta_sublinear_v0.json","w"), indent=2)
print("\nwrote results/rosetta_check/rosetta_sublinear_v0.json")

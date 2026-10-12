import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d=json.load(open("results/rosetta_check/rosetta_sublinear_v0.json"))
rows=d["rows"]
fig,ax=plt.subplots(figsize=(5.2,3.6))
for fam,col,mk in [("Qwen3","#4676e5","o"),("Gemma3","#d75644","s"),("Pythia","#3a9e54","^")]:
    pts=sorted([(r["params"],r["r"],r["npos"]) for r in rows if r["family"]==fam])
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.plot(xs,ys,mk+"-",color=col,label=fam,ms=7,lw=1.6)
    for x,y,npos in pts:
        ax.annotate(f"{x:.0f}B" if x>=1 else f"{x}B",(x,y),textcoords="offset points",xytext=(4,4),fontsize=7,color=col)
# power-law fit lines
for label,col in [("Gemma3","#d75644"),("Qwen3_excl14B","#4676e5"),("Pythia","#3a9e54")]:
    f=d["fits"][label]; xs=np.array(sorted(f["params"])); b,c=f["exponent_b"],f["coef_c"]
    xx=np.linspace(min(xs),max(xs),50); ax.plot(xx,c*xx**b,"--",color=col,lw=1,alpha=0.6)
ax.set_xscale("log"); ax.set_xlabel("parameters (B, log scale)")
ax.set_ylabel("cross-lingual probe-attribution mean $r$")
ax.axhline(0,color="gray",lw=0.5,ls=":")
ax.set_title("Architecture gates scale: within-family slope\nfamily-conditional (Pythia rises, Qwen3 flat); gap dominates",fontsize=9)
ax.legend(fontsize=8,loc="center left"); ax.grid(alpha=0.25)
g=[r["r"] for r in rows if r["model"]=="google/gemma-3-4b-it"][0]
q=[r["r"] for r in rows if r["model"]=="Qwen/Qwen3-14B"][0]
ax.annotate(f"{g/q:.0f}× gap\n@1/3.5 params",(4,0.572),textcoords="offset points",xytext=(10,-22),fontsize=7,color="#d75644",
            arrowprops=dict(arrowstyle="->",color="#d75644",lw=0.8))
plt.tight_layout(); plt.savefig("figures/fig_rosetta_sublinear.pdf"); plt.savefig("figures/fig_rosetta_sublinear.png",dpi=140)
print("wrote figures/fig_rosetta_sublinear.{pdf,png}")

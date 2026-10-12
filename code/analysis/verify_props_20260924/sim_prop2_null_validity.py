import numpy as np
rng=np.random.default_rng(1)
def zs(X): X=X-X.mean(0); return X/X.std(0)
K,N,nperm,reps=100,200,199,200
pf,pc,pr=[],[],[]
for _ in range(reps):
    As,Bs,Ae,Be=[zs(rng.standard_normal((N,K))) for _ in range(4)]   # H0: A,B independent
    def mnn(C):
        a=C.argmax(1); b=C.argmax(0); i=np.where(b[a]==np.arange(K))[0]; return i,a[i]
    i,j=mnn(As.T@Bs/N)                 # frozen on D_S
    T=lambda Bm,ii,jj:(Ae[:,ii]*Bm[:,jj]).sum(0).mean()/N
    obs=T(Be,i,j); null=[T(Be[rng.permutation(N)],i,j) for _ in range(nperm)]
    pf.append((1+np.sum(np.array(null)>=obs))/(nperm+1))
    ie,je=mnn(Ae.T@Be/N)               # re-selected on D_E
    obs=T(Be,ie,je); null=[T(Be[rng.permutation(N)],ie,je) for _ in range(nperm)]   # fixed-pair (circular) null, as in code
    pc.append((1+np.sum(np.array(null)>=obs))/(nperm+1))
    def Tsel(Bm):
        ii,jj=mnn(Ae.T@Bm/N); return T(Bm,ii,jj)
    obs=Tsel(Be); null=[Tsel(Be[rng.permutation(N)]) for _ in range(nperm)]          # re-run selection per draw
    pr.append((1+np.sum(np.array(null)>=obs))/(nperm+1))
for n,p in [('frozen pairs',pf),('re-selected, fixed-pair null',pc),('re-selected, re-run null',pr)]:
    p=np.array(p); print(f"{n:32s} P(p<=.05)={np.mean(p<=.05):.3f}  P(p<=.10)={np.mean(p<=.10):.3f} mean p={p.mean():.3f}")

import numpy as np
rng = np.random.default_rng(0)
def zs(X):
    X = X - X.mean(0); s = X.std(0); keep = s > 1e-12
    return X[:, keep] / s[keep]
def M(Za, Zb):
    C = Za.T @ Zb / Za.shape[0]
    a = C.argmax(1); b = C.argmax(0)
    return int((b[a] == np.arange(C.shape[0])).sum()), C.shape
def run(gen, KA, KB, N, reps):
    ms = []
    for _ in range(reps):
        A, B = gen(N, KA, 'A'), gen(N, KB, 'B')
        m, sh = M(zs(A), zs(B)); ms.append(m)
    ms = np.array(ms); ka, kb = sh
    th = ka*kb/(ka+kb-1)
    return ms.mean(), ms.std()/np.sqrt(reps), th
gauss = lambda N,K,s: rng.standard_normal((N,K))
def sparse(N,K,s):   # SAE-like: nonnegative, ~3% active, exponential magnitudes
    return (rng.random((N,K))<0.03)*rng.exponential(1.0,(N,K)) + 1e-9*rng.random((N,K))
def mixed(N,K,s):    # A sparse, B Gaussian
    return sparse(N,K,s) if s=='A' else gauss(N,K,s)
def lowrank(N,K,s):  # within-model correlated: 5 shared latent factors + noise, independent across models
    F = rng.standard_normal((N,5)); W = rng.standard_normal((5,K))
    return F@W + 1.0*rng.standard_normal((N,K))
def lognorm(N,K,s):  # iid heavy-tailed continuous features
    return rng.lognormal(0,1.5,(N,K))
for name, g in [('gauss',gauss),('lognormal',lognorm),('sparse',sparse),('A sparse/B gauss',mixed),('low-rank within',lowrank)]:
    for K in (50,100):
        for N in (30, 200, 1000):
            m,se,th = run(g,K,K,N,400 if K==50 else 200)
            print(f"{name:18s} K={K:3d} N={N:5d}  E[M]_sim={m:7.2f} ±{se:5.2f}  formula={th:6.2f}  diff/se={(m-th)/se:+6.1f}")
# unequal K
for N in (200,):
    m,se,th = run(gauss,40,120,N,400); print(f"gauss KA=40 KB=120 N={N} sim={m:.2f}±{se:.2f} formula={th:.2f}")

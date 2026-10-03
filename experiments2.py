"""
Round-2 experiments. Hypotheses declared BEFORE this script was run (round 1 = experiments.py: H1-H4).
NHST families (paired Wilcoxon signed-rank, each over the 10 (D, function) cells):
  H5  1/5-rule hill climber  vs  best fixed-sigma hill climber (sigma* picked in-sample from {0.003,0.01,0.03,0.1,0.3,1})
  H6  ML-BJS free-probe       vs  static p=0.2
  H7  ML-BJS probe-reuse      vs  ML-BJS (charged)
  H8  linearly decaying p     vs  static p=0.2
Holm correction over all 80 NHST tests (H1-H8).
Equivalence (TOST on paired log10 ratio, margin = factor 2): E1 history-BJS vs p=0.2; E2 ML-BJS free-probe vs p=0.4.
Holm over the 20 TOST tests.
"""
import sys, csv, time
import numpy as np
from collections import deque
from multiprocessing import Pool
from sklearn.ensemble import RandomForestRegressor
import experiments as E
from experiments import (FUNCS, shifted, Counted, psi, search, one_fifth, Const, History, MLProbe, MLFree,
                         Policy, LO, HI, SIGMA, K, SEEDS, BUDGET, DIMS, train_phi)

def zakharov(x):
    s = np.sum(0.5*np.arange(1, x.shape[-1]+1)*x, axis=-1)
    return np.sum(x**2, axis=-1) + s**2 + s**4
def levy(x):
    w = 1 + (x-1)/4
    return (np.sin(np.pi*w[..., 0])**2
            + np.sum((w[..., :-1]-1)**2*(1+10*np.sin(np.pi*w[..., :-1]+1)**2), axis=-1)
            + (w[..., -1]-1)**2*(1+np.sin(2*np.pi*w[..., -1])**2))

def train_table(D, fl, rng_seed, rf_state):
    rng = np.random.default_rng(rng_seed); X, y = [], []
    for fn, lab in fl:
        for _ in range(300):
            x = rng.uniform(LO, HI, D); X.append([psi(fn, x, float(fn(x)), rng)]); y.append(lab)
    rf = RandomForestRegressor(n_estimators=30, random_state=rf_state).fit(X, y)
    return rf.predict([[j/K] for j in range(K+1)]), np.array(X)[:, 0], np.array(y)

class Decay(Policy):                       # time-dependent (counter) policy, zero cost
    def __init__(self, B, p0=0.5): self.B, self.p0 = B, p0
    def p(self, ctx): return self.p0*max(0.0, 1 - ctx["f"].n/self.B)
class HistLow(Policy):                     # history policy, lower range
    def p(self, ctx):
        h = ctx["hist"]; return 0.1 if not h else 0.02 + 0.2*(1 - np.mean(h))

def search_adapt_sigma(f, D, B, pol, rng):  # 1/5 rule on sigma (local steps only) + policy for p
    x = rng.uniform(LO, HI, D); fx = f(x); s = SIGMA; hist = deque(maxlen=20)
    while f.n + pol.cost <= B:
        p = pol.p(dict(f=f, x=x, fx=fx, hist=hist, rng=rng))
        local = rng.random() >= p
        c = np.clip(x + rng.normal(0, s, D), LO, HI) if local else rng.uniform(LO, HI, D)
        fc = f(c); ok = fc < fx
        if ok: x, fx = c, fc
        if local: s = float(np.clip(s*1.5 if ok else s*1.5**-0.25, 1e-12, 10))
        hist.append(ok)
    return fx

def _step(x, p, D, rng):
    return rng.uniform(LO, HI, D) if rng.random() < p else np.clip(x + rng.normal(0, SIGMA, D), LO, HI)

def search_ml_reuse(f, D, B, table, rng):   # improving probes are accepted as moves
    x = rng.uniform(LO, HI, D); fx = f(x)
    while f.n + 1 + K <= B:
        P = np.clip(x + rng.normal(0, SIGMA, (K, D)), LO, HI); fp = f.many(P); j = int(np.argmin(fp))
        if fp[j] < fx: x, fx = P[j], float(fp[j]); continue
        c = _step(x, table[0], D, rng); fc = f(c)
        if fc < fx: x, fx = c, fc
    return fx

def search_ml_every(f, D, B, table, rng, m=4):   # probe every m-th iteration only
    x = rng.uniform(LO, HI, D); fx = f(x); p = 0.2; it = 0
    while True:
        need = (1 + K) if it % m == 0 else 1
        if f.n + need > B: break
        if it % m == 0: p = table[int(round(K*psi(lambda X: f.many(X), x, fx, rng)))]
        c = _step(x, p, D, rng); fc = f(c)
        if fc < fx: x, fx = c, fc
        it += 1
    return fx

def l3_refine(f, D, B, rng, n, centres):    # tabulate a grid, jump to its argmin, hill-climb with the rest
    if centres:
        h = (HI-LO)/n; ax = LO + h*(np.arange(n)+0.5)
    else:
        ax = np.linspace(LO, HI, n)
    G = np.stack(np.meshgrid(*[ax]*D, indexing="ij"), -1).reshape(-1, D)
    v = f.many(G); j = int(np.argmin(v)); x, fx = G[j], float(v[j])
    while f.n + 1 <= B:
        c = np.clip(x + rng.normal(0, SIGMA, D), LO, HI); fc = f(c)
        if fc < fx: x, fx = c, fc
    return fx

SIG = (0.003, 0.01, 0.03, 0.3, 1.0)
LARGE = {2: 5, 10: 3}                       # budget multipliers for the larger-budget check

def task(args):
    D, name, seed, tabs = args
    fidx = list(FUNCS).index(name)
    shift = np.random.default_rng([seed, D, fidx]).uniform(-2, 2, D)     # identical instances to round 1
    raw = shifted(name, shift); B = BUDGET(D); rows = []
    mid = [300]
    def rng_next():
        mid[0] += 1; return np.random.default_rng([seed, D, fidx, mid[0]])
    def run(label, fn, budget=B):
        f = Counted(raw); rows.append((D, name, label, seed, fn(f, budget), f.n))
    for p in (0.3, 0.4):
        r = rng_next(); run(f"p={p}", lambda f, b: search(f, D, b, Const(p), r))
    for s in SIG:
        r = rng_next(); run(f"p=0.0,sigma={s}", lambda f, b: search(f, D, b, Const(0.0), r, sigma=s))
    r = rng_next(); run("1/5sigma+p=0.2", lambda f, b: search_adapt_sigma(f, D, b, Const(0.2), r))
    r = rng_next(); run("decay-p", lambda f, b: search(f, D, b, Decay(b), r))
    r = rng_next(); run("history-BJS low", lambda f, b: search(f, D, b, HistLow(), r))
    t0 = tabs["orig"]
    r = rng_next(); run("ML-BJS reuse", lambda f, b: search_ml_reuse(f, D, b, t0, r))
    r = rng_next(); run("ML-BJS probe/4", lambda f, b: search_ml_every(f, D, b, t0, r))
    for j in range(1, 5):
        r = rng_next(); run(f"ML-BJS train{j}", lambda f, b: search(f, D, b, MLProbe(tabs[j]), r))
    r = rng_next(); run("ML-BJS disjoint", lambda f, b: search(f, D, b, MLProbe(tabs["disj"]), r))
    n_m = max(2, int(np.floor((B/2)**(1/D))))
    r = rng_next(); run("L3 matched+refine", lambda f, b: l3_refine(f, D, b, r, n_m, True))
    if D == 2:
        r = rng_next(); run("L3 full+refine", lambda f, b: l3_refine(f, D, b, r, 41, False))
    # larger budget, subset of methods
    BL = B*LARGE[D]
    r = rng_next(); run("B+ p=0.2", lambda f, b: search(f, D, b, Const(0.2), r), BL)
    r = rng_next(); run("B+ ML-BJS", lambda f, b: search(f, D, b, MLProbe(t0), r), BL)
    r = rng_next(); run("B+ ML-BJS free-probe", lambda f, b: search(f, D, b, MLFree(t0, raw), r), BL)
    r = rng_next(); run("B+ history-BJS", lambda f, b: search(f, D, b, History(), r), BL)
    r = rng_next(); run("B+ 1/5-rule", lambda f, b: one_fifth(f, D, b, r), BL)
    return rows

if __name__ == "__main__":
    D = int(sys.argv[1])
    tabs = {"orig": train_phi(D)}
    for j in range(1, 5):
        tabs[j] = train_table(D, [(FUNCS["sphere"], 0.05), (FUNCS["rastrigin"], 0.8)], 12345 + D + 1000*j, j)[0]
    tabs["disj"] = train_table(D, [(zakharov, 0.05), (levy, 0.8)], 777 + D, 0)[0]
    for k, v in tabs.items(): print(D, k, np.round(v, 3), flush=True)
    jobs = [(D, n, s, tabs) for n in FUNCS for s in range(SEEDS)]
    out = []; t0 = time.time()
    with Pool(1) as pool:
        for rows in pool.imap_unordered(task, jobs): out += rows
    print(f"D={D} round-2 done in {time.time()-t0:.0f}s", flush=True)
    with open(f"raw2_D{D}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["D", "function", "method", "seed", "best", "FEs"]); w.writerows(out)

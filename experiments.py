"""
Extended experiments for the three-level information-capacity hierarchy.

Pre-specified confirmatory comparisons (fixed BEFORE running; see analysis.py):
  H1  ML-BJS (probe cost charged)  vs  static p=0.2
  H2  ML-BJS (probe cost charged)  vs  ML-BJS with free probes (overhead ablation)
  H3  history-BJS (zero-cost)      vs  static p=0.2
  H4  1/5-rule hill climber        vs  static p=0.2
each for every (dimension, function) cell; Holm correction over all 40 tests.
"""
import sys, csv, time
import numpy as np
from collections import deque
from multiprocessing import Pool
from sklearn.ensemble import RandomForestRegressor

LO, HI, SIGMA, K = -5.12, 5.12, 0.1, 8
SEEDS, DIMS = 30, (2, 10)
BUDGET = lambda D: 1000 * D            # 2000 FEs at D=2, 10000 at D=10

# ---- benchmark functions (vectorised over the last axis; global minimum 0) ----
FUNCS = {
 "sphere":    lambda x: np.sum(x**2, axis=-1),
 "rastrigin": lambda x: 10*x.shape[-1] + np.sum(x**2 - 10*np.cos(2*np.pi*x), axis=-1),
 "ackley":    lambda x: (-20*np.exp(-0.2*np.sqrt(np.mean(x**2, axis=-1)))
                         - np.exp(np.mean(np.cos(2*np.pi*x), axis=-1)) + 20 + np.e),
 "griewank":  lambda x: (1 + np.sum(x**2, axis=-1)/4000
                         - np.prod(np.cos(x/np.sqrt(np.arange(1, x.shape[-1]+1))), axis=-1)),
 "rosenbrock":lambda x: np.sum(100*(x[...,1:]-x[...,:-1]**2)**2 + (1-x[...,:-1])**2, axis=-1),
}
TRAIN = ("sphere", "rastrigin")          # ML policy is trained on these only (unshifted)

def shifted(name, shift):
    f = FUNCS[name]
    return lambda x: f(x - shift)

class Counted:
    def __init__(self, f): self.f, self.n = f, 0
    def __call__(self, x): self.n += 1; return float(self.f(x))
    def many(self, X): self.n += len(X); return self.f(X)

# ---- level-2 policy: psi(x) = fraction of K probes that improve; phi = 9-entry table ----
def psi(fvec, x, fx, rng):
    probes = np.clip(x + rng.normal(0, SIGMA, (K, len(x))), LO, HI)
    return np.mean(fvec(probes) < fx)

def train_phi(D):
    """Random forest (30 trees, random_state=0) fitted on psi->p targets (0.05 smooth, 0.8 rugged);
    uniform random points on unshifted Sphere/Rastrigin. Since psi has K+1 values, phi is a 9-entry table."""
    rng = np.random.default_rng(12345 + D)
    X, y = [], []
    for name, lab in (("sphere", 0.05), ("rastrigin", 0.8)):
        for _ in range(300):
            x = rng.uniform(LO, HI, D)
            X.append([psi(FUNCS[name], x, FUNCS[name](x), rng)]); y.append(lab)
    rf = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
    return rf.predict([[j/K] for j in range(K+1)])      # table indexed by round(K*psi)

# ---- the one search engine (Levels 1 and 2) ----
class Policy:
    cost = 1                                  # FEs per iteration (candidate only)
    def p(self, ctx): raise NotImplementedError
class Const(Policy):
    def __init__(self, p): self.v = p
    def p(self, ctx): return self.v
class History(Policy):                        # zero-cost: recent rejection rate
    def p(self, ctx):
        h = ctx["hist"]
        return 0.2 if not h else 0.05 + 0.5*(1 - np.mean(h))
class MLProbe(Policy):                        # probes charged to the budget
    cost = 1 + K
    def __init__(self, table): self.t = table
    def p(self, ctx):
        z = psi(lambda X: ctx["f"].many(X), ctx["x"], ctx["fx"], ctx["rng"])
        return self.t[int(round(K*z))]
class MLFree(Policy):                         # ablation: probes NOT charged (not budget-fair)
    def __init__(self, table, raw): self.t, self.raw = table, raw
    def p(self, ctx):
        z = psi(self.raw, ctx["x"], ctx["fx"], ctx["rng"])
        return self.t[int(round(K*z))]

def search(f, D, B, policy, rng, sigma=SIGMA):
    x = rng.uniform(LO, HI, D); fx = f(x); hist = deque(maxlen=20)
    while f.n + policy.cost <= B:             # strict budget: never exceeds B
        p = policy.p(dict(f=f, x=x, fx=fx, hist=hist, rng=rng))
        c = rng.uniform(LO, HI, D) if rng.random() < p else np.clip(x + rng.normal(0, sigma, D), LO, HI)
        fc = f(c); ok = fc < fx
        if ok: x, fx = c, fc
        hist.append(ok)
    return fx

def one_fifth(f, D, B, rng):                  # (1+1)-ES with the 1/5 success rule on sigma
    x = rng.uniform(LO, HI, D); fx = f(x); s = SIGMA
    while f.n < B:
        c = np.clip(x + rng.normal(0, s, D), LO, HI); fc = f(c)
        if fc < fx: x, fx, s = c, fc, s*1.5
        else: s *= 1.5**-0.25
    return fx

def grid_table(fvec, D, n=41):                # Level 3: tabulate f on an n^D grid, take argmin
    ax = np.linspace(LO, HI, n)
    G = np.stack(np.meshgrid(*[ax]*D, indexing="ij"), -1)
    return float(fvec(G).min()), n**D

PS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)

def task(args):
    D, name, seed, table = args
    rs = np.random.default_rng([seed, D, list(FUNCS).index(name)])
    shift = rs.uniform(-2, 2, D)                            # instance: random optimum location
    raw = shifted(name, shift)
    B = BUDGET(D); rows = []
    def run(label, fn, **kw):
        f = Counted(raw)
        val = fn(f)
        rows.append((D, name, label, seed, val, f.n))
    for p in PS:
        lab = f"p={p}"; r = np.random.default_rng([seed, D, list(FUNCS).index(name), PS.index(p)])
        run(lab, lambda f: search(f, D, B, Const(p), r))
    for s in (0.03, 0.3):
        r = np.random.default_rng([seed, D, list(FUNCS).index(name), 100 + int(s*100)])
        run(f"p=0.2,sigma={s}", lambda f: search(f, D, B, Const(0.2), r, sigma=s))
    r = np.random.default_rng([seed, D, list(FUNCS).index(name), 200])
    run("ML-BJS", lambda f: search(f, D, B, MLProbe(table), r))
    r = np.random.default_rng([seed, D, list(FUNCS).index(name), 201])
    run("ML-BJS free-probe", lambda f: search(f, D, B, MLFree(table, raw), r))
    r = np.random.default_rng([seed, D, list(FUNCS).index(name), 202])
    run("history-BJS", lambda f: search(f, D, B, History(), r))
    r = np.random.default_rng([seed, D, list(FUNCS).index(name), 203])
    run("1/5-rule", lambda f: one_fifth(f, D, B, r))
    if D == 2:
        v, n = grid_table(raw, D); rows.append((D, name, "L3 grid", seed, v, n))
    return rows

if __name__ == "__main__":
    dims = [int(a) for a in sys.argv[1:]] or list(DIMS)
    out = []
    for D in dims:
        table = train_phi(D); print("D", D, "phi table", np.round(table, 3), flush=True)
        jobs = [(D, n, s, table) for n in FUNCS for s in range(SEEDS)]
        t0 = time.time()
        with Pool(1) as pool:
            for rows in pool.imap_unordered(task, jobs): out += rows
        print(f"D={D} done in {time.time()-t0:.0f}s", flush=True)
        with open(f"raw_D{D}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["D","function","method","seed","best","FEs"]); w.writerows(out_ for out_ in out if out_[0]==D)

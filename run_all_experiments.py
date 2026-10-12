"""
=============================================================================
 Information state and evaluation cost in single-solution heuristic search
 ONE-FILE REPRODUCTION SCRIPT  (Google Colab or any Python >= 3.9)
=============================================================================
This single file merges the five original scripts (experiments.py,
experiments2.py, logging_exp.py, analysis2.py, fig_info.py) without changing
any experimental logic, random streams or seeds. It therefore regenerates the
published raw results exactly (when run with the same library versions) and
then every table, statistic and figure of the paper.

HOW TO RUN IN GOOGLE COLAB
  1. Upload this file (or paste it into one cell and run it).
  2. Run it:    !python run_all_experiments.py        (or  %run run_all_experiments.py)
     Nothing needs to be installed (numpy, scipy, scikit-learn, matplotlib are preinstalled).
  3. Everything is written to ./results and zipped to results.zip (auto-download in Colab).

RUNTIME  about 25 CPU-minutes in total (about 13 minutes on Colab's 2 cores).
         Set QUICK = True for a ~2 minute smoke test with 2 instances per cell.

PIPELINE (round 1 -> round 2 -> diagnostics -> analysis -> figures -> check)
  Round 1  : constants, ML-BJS (probe policy), free-probe ablation, history rule, 1/5 rule, Level-3 table   (H1-H4)
  Round 2  : fixed-step hill climbers, decay schedule, probe reuse / probe-every-4 / retraining / disjoint
             training, budget-matched grid + refinement, larger-budget check                               (H5-H8)
  Diagnostics: what the learned and history policies actually do (realised information)
  Analysis : Wilcoxon tests with Holm correction, equivalence tests, accounting of information, LaTeX table
             bodies (t*.tex), facts.json, figures
  Check    : compares the regenerated raw CSVs with the published ones (downloaded from GitHub)
=============================================================================
"""
import os, sys, csv, json, math, time, zipfile, urllib.request
import multiprocessing as mp
from collections import deque
import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------- configuration -------------------------------
QUICK = False                      # True = tiny smoke test (2 instances per cell)
WORKERS = os.cpu_count() or 1      # instances are independent, so results do not depend on this
OUT = "results"                    # all outputs go here
VERIFY = True                      # compare with the published raw CSVs
REF_BASE = "https://raw.githubusercontent.com/matinkhah/three-level-heuristics/main/"
REF_DIR = os.environ.get("REF_DIR", "")   # optional local folder holding the published CSVs
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

# ============================================================================
# PART A  Round 1 (was experiments.py): benchmarks, policies, engine, H1-H4 runs
# ============================================================================
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

def task_round1(args):
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


# ============================================================================
# PART B  Round 2 (was experiments2.py): H5-H8, ablations, Level 3, larger budget
# ============================================================================
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

def task_round2(args):
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


# ============================================================================
# PART C  Realised-information diagnostics (was logging_exp.py)
# ============================================================================
def entropy(vals):
    _, c = np.unique(vals, return_counts=True); p = c/c.sum(); return float(-(p*np.log2(p)).sum())

def mutual_info(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ja = {v: i for i, v in enumerate(np.unique(a))}; jb = {v: i for i, v in enumerate(np.unique(b))}
    M = np.zeros((len(ja), len(jb)))
    for u, v in zip(a, b): M[ja[u], jb[v]] += 1
    M /= M.sum(); pa = M.sum(1, keepdims=True); pb = M.sum(0, keepdims=True)
    nz = M > 0; return float((M[nz]*np.log2(M[nz]/(pa@pb)[nz])).sum())

class Logged(Policy):
    def __init__(self, inner): self.inner, self.cost, self.zs, self.ps = inner, inner.cost, [], []
    def p(self, ctx):
        v = self.inner.p(ctx); self.ps.append(v)
        if hasattr(self.inner, "last_z"): self.zs.append(self.inner.last_z)
        return v

class MLProbeLog(MLProbe):
    def __init__(self, t): super().__init__(t); self.ps, self.zs = [], []
    def p(self, ctx):
        z = psi(lambda Xp: ctx["f"].many(Xp), ctx["x"], ctx["fx"], ctx["rng"])
        j = int(round(K*z)); self.zs.append(j); self.ps.append(self.t[j]); return self.t[j]

def search_xy(f, D, B, policy, rng):   # engine copy that also returns start and final points
    x = rng.uniform(LO, HI, D); x0 = x.copy(); fx = f(x); hist = deque(maxlen=20)
    while f.n + policy.cost <= B:
        p = policy.p(dict(f=f, x=x, fx=fx, hist=hist, rng=rng))
        c = rng.uniform(LO, HI, D) if rng.random() < p else np.clip(x + rng.normal(0, SIGMA, D), LO, HI)
        fc = f(c); ok = fc < fx
        if ok: x, fx = c, fc
        hist.append(ok)
    return fx, x0, x


def logging_train_record(D):
    table, Xtr, ytr = train_table(D, [(FUNCS["sphere"], 0.05), (FUNCS["rastrigin"], 0.8)], 12345 + D, 0)
    assert np.allclose(table, train_phi(D))                      # same table as the main runs
    zt = np.round(Xtr*K).astype(int)
    return dict(table=table.tolist(), mean_label=float(ytr.mean()), mean_abs_dev_from_mean_label=float(np.abs(table - ytr.mean()).mean()),
                H_psi_bits=entropy(zt), MI_psi_label_bits=mutual_info(zt, (ytr > 0.4).astype(int)),
                frac_psi0=float((zt == 0).mean()))


def logging_cell(args):
    """Realised-information record for one (dimension, function) cell (loop body of logging_exp.py)."""
    D, fn = args
    table = train_phi(D)
    B = BUDGET(D); fidx = list(FUNCS).index(fn)
    ml_z, ml_p, hi_p, ml_mean, hi_mean = [], [], [], [], []
    same_hc, same_p2 = [], []
    for seed in range(SEEDS):
        shift = np.random.default_rng([seed, D, fidx]).uniform(-2, 2, D); raw = shifted(fn, shift)
        r = np.random.default_rng([seed, D, fidx, 900])
        pol = MLProbeLog(table); f = Counted(raw); search(f, D, B, pol, r)
        ml_z.append(np.array(pol.zs)); ml_p.append(np.array(pol.ps))
        hist_pol = Logged(History()); f = Counted(raw); search(f, D, B, hist_pol, np.random.default_rng([seed, D, fidx, 901]))
        hi_p.append(np.array(hist_pol.ps))
        if fn == "rastrigin":
            for lab, p, lst in (("hc", 0.0, same_hc), ("p02", 0.2, same_p2)):
                f = Counted(raw); fx, x0, xf = search_xy(f, D, B, Const(p), np.random.default_rng([seed, D, fidx, 902]))
                lst.append(bool(np.all(np.rint(xf - shift) == np.rint(x0 - shift))))
    allp = np.concatenate(ml_p); zvis = np.concatenate(ml_z)
    hp = np.concatenate(hi_p)
    rec = dict(ml_mean_p=float(np.mean(allp)), ml_std_p=float(np.std(allp)), ml_H_p_bits=entropy(np.round(allp/0.05)*0.05),
               ml_iters_per_run=float(np.mean([len(a) for a in ml_p])), ml_H_psi_bits=entropy(zvis), ml_frac_psi0=float(np.mean(zvis == 0)),
               hist_mean_p=float(np.mean(hp)), hist_std_p=float(np.std(hp)), hist_H_p_bits=entropy(np.round(hp/0.05)*0.05),
               hist_mean_p_first_quarter=float(np.mean([a[:max(1, len(a)//4)].mean() for a in hi_p])),
               hist_mean_p_last_quarter=float(np.mean([a[-max(1, len(a)//4):].mean() for a in hi_p])))
    if fn == "rastrigin":
        rec.update(frac_same_basin_hc=float(np.mean(same_hc)), frac_same_basin_p02=float(np.mean(same_p2)))
    return f"D{D}_{fn}", rec

# ============================================================================
# PART D  Analysis, tables and figures (was analysis2.py)
# ============================================================================
def analyze():

    FLOOR = 1e-12
    FUNCS = ["sphere", "rastrigin", "ackley", "griewank", "rosenbrock"]
    DATA = {}
    for fn_ in ("raw_D2.csv", "raw_D10.csv", "raw2_D2.csv", "raw2_D10.csv"):
        for r in csv.DictReader(open(fn_)):
            DATA.setdefault((int(r["D"]), r["function"], r["method"]), {})[int(r["seed"])] = (float(r["best"]), int(r["FEs"]))
    def get(D, fn, m): return np.array([DATA[(D, fn, m)][s][0] for s in sorted(DATA[(D, fn, m)])])
    def clip(v): return np.maximum(v, FLOOR)
    def med(D, fn, m): return float(np.median(get(D, fn, m)))
    def iqr(D, fn, m): return np.percentile(get(D, fn, m), [25, 75])
    def has(D, fn, m): return (D, fn, m) in DATA
    PS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    SIGS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    def hc(s): return "p=0.0" if s == 0.1 else f"p=0.0,sigma={s}"

    # ---- budget check ----
    for (D, fn, m), d in DATA.items():
        B = 1000*D*({2: 5, 10: 3}[D] if m.startswith("B+") else 1)
        if m.startswith("L3 grid") or m == "L3 full+refine" or m == "L3 matched+refine" or m.startswith("L3"): 
            assert max(v[1] for v in d.values()) <= B, (D, fn, m)
        else:
            assert max(v[1] for v in d.values()) <= B, (D, fn, m, max(v[1] for v in d.values()))
    print("all runs within budget")

    def fmt(x, sig=3, sci_sig=None):
        sci_sig = sci_sig or sig
        if x < FLOOR: return "<\\!10^{-12}"
        if x == 0: return "0"
        e = int(math.floor(math.log10(abs(x))))
        if -2 <= e <= 2: return format(x, f".{sig}g")
        return f"{x/10**e:.{sci_sig-1}f}\\mathrm{{e}}{{{e}}}"
    def cell(x, bold=False): s = fmt(x); return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"
    def rfmt(r):
        if r == 0: return "$0$"
        def sci(v): e = int(math.floor(math.log10(v))); return f"${v/10**e:.1f}\\times10^{{{e}}}$"
        if r < 0.01 or r >= 1000: return sci(r)
        return f"{r:.2f}" if r < 10 else f"{r:.0f}"

    # ---------------- main tables ----------------
    MAIN = [("p=0.0", "$p{=}0$"), ("p=0.2", "$p{=}0.2$"), ("p=1.0", "$p{=}1$"), ("ML-BJS", "ML-BJS"), ("history-BJS", "Hist-BJS"), ("decay-p", "Decay-$p$"), ("1/5-rule", "1/5-rule")]
    for D in (2, 10):
        lines = []
        for fn in FUNCS:
            meds = {m: med(D, fn, m) for m, _ in MAIN}; best = min(meds, key=lambda m: max(meds[m], FLOOR))
            l1 = f"{fn.capitalize()}"; l2 = "\\hspace{1em}IQR"
            for m, _ in MAIN:
                l1 += " & " + cell(meds[m], m == best); q = iqr(D, fn, m)
                l2 += f" & $[{fmt(q[0],3,2)},{fmt(q[1],3,2)}]$"
            lines += [l1 + " \\\\", l2 + " \\\\"]
        open(f"t2_main_D{D}.tex", "w").write("\n".join(lines))

    # ---------------- variants / ablations ----------------
    VAR = [("ML-BJS free-probe", "free-probe"), ("ML-BJS reuse", "reuse"), ("ML-BJS probe/4", "probe/4"), (None, "train 0--4"), ("ML-BJS disjoint", "disjoint"),
           ("history-BJS low", "Hist low"), ("1/5sigma+p=0.2", "1/5$\\sigma$+0.2"), (None, "best-$\\sigma$ HC")]
    lines = []
    sig_star = {}
    for D in (2, 10):
        for fn in FUNCS:
            row = [f"{D}", fn.capitalize()]
            for m, lab in VAR:
                if m is not None:
                    row.append(cell(med(D, fn, m)))
                elif lab.startswith("train"):
                    ms = [med(D, fn, "ML-BJS")] + [med(D, fn, f"ML-BJS train{j}") for j in range(1, 5)]
                    row.append(f"${fmt(min(ms),2)}$--${fmt(max(ms),2)}$")
                else:
                    s = min(SIGS, key=lambda s: max(med(D, fn, hc(s)), FLOOR)); sig_star[(D, fn)] = s
                    row.append(f"{cell(med(D, fn, hc(s)))} ($\\sigma{{=}}{s:g}$)")
            lines.append(" & ".join(row) + " \\\\")
        if D == 2: lines.append("\\midrule")
    open("t4_variants.tex", "w").write("\n".join(lines))

    # ---------------- level 3 ----------------
    lines = []
    for D in (2, 10):
        for fn in FUNCS:
            cols = [("L3 grid", D == 2), ("L3 full+refine", D == 2), ("L3 matched+refine", True), ("p=0.0", True), ("p=0.2", True)]
            l1 = f"{D} & {fn.capitalize()}"; 
            for m, ok in cols:
                l1 += " & " + (cell(med(D, fn, m)) if ok else "---")
            lines.append(l1 + " \\\\")
        if D == 2: lines.append("\\midrule")
    open("t5_l3.tex", "w").write("\n".join(lines))

    # ---------------- tests ----------------
    def paired(D, fn, a, b):
        x, y = clip(get(D, fn, a)), clip(get(D, fn, b)); d = np.log10(x) - np.log10(y)
        try: p = float(wilcoxon(x, y).pvalue)
        except ValueError: p = 1.0
        return float(np.median(d)), p
    HYP = [("H1", "ML-BJS", "p=0.2"), ("H2", "ML-BJS", "ML-BJS free-probe"), ("H3", "history-BJS", "p=0.2"), ("H4", "1/5-rule", "p=0.2"),
           ("H5", "1/5-rule", "BEST_HC"), ("H6", "ML-BJS free-probe", "p=0.2"), ("H7", "ML-BJS reuse", "ML-BJS"), ("H8", "decay-p", "p=0.2")]
    tests = []
    for D in (2, 10):
        for fn in FUNCS:
            for h, a, b in HYP:
                bb = hc(sig_star[(D, fn)]) if b == "BEST_HC" else b
                d, p = paired(D, fn, a, bb); tests.append(dict(D=D, fn=fn, h=h, a=a, b=bb, d=d, p=p))
    def holm(ps):
        m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0
        for rank, i in enumerate(order): run = max(run, (m-rank)*ps[i]); adj[i] = min(1.0, run)
        return adj
    adj = holm([t["p"] for t in tests])
    for t, a in zip(tests, adj): t["holm"] = float(a)
    def pf(p): return "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0") if p < 1 else "1"
    lines = []
    for D in (2, 10):
        for fn in FUNCS:
            row = [f"{D}", fn.capitalize()]
            for h, _, _ in HYP:
                t = next(t for t in tests if t["D"] == D and t["fn"] == fn and t["h"] == h)
                row.append(f"{t['d']:+.1f} ({pf(t['holm'])}){'$^\\dagger$' if t['holm'] < 0.05 else ''}")
            lines.append(" & ".join(row) + " \\\\")
        if D == 2: lines.append("\\midrule")
    open("t6_tests.tex", "w").write("\n".join(lines))

    # ---------------- equivalence (TOST, factor 2) ----------------
    DELTA = math.log10(2)
    def _tost(D, fn, a, b):
        d = np.log10(clip(get(D, fn, a))) - np.log10(clip(get(D, fn, b)))
        p1 = wilcoxon(d + DELTA, alternative="greater").pvalue; p2 = wilcoxon(d - DELTA, alternative="less").pvalue
        return float(max(p1, p2))
    def tost(D, fn, a, b):
        try: return _tost(D, fn, a, b)
        except ValueError: return 1.0
    EQ = [("E1", "history-BJS", "p=0.2"), ("E2", "ML-BJS free-probe", "p=0.4")]
    eq = []
    for D in (2, 10):
        for fn in FUNCS:
            for e, a, b in EQ: eq.append(dict(D=D, fn=fn, e=e, p=tost(D, fn, a, b)))
    adjq = holm([t["p"] for t in eq])
    for t, a in zip(eq, adjq): t["holm"] = float(a)

    # ---------------- value-of-information table ----------------
    lines = []; value = {}
    for D in (2, 10):
        for fn in FUNCS:
            pstar = min(PS, key=lambda p: max(med(D, fn, f"p={p}"), FLOOR)); gap = max(med(D, fn, "p=0.2"), FLOOR)/max(med(D, fn, f"p={pstar}"), FLOOR)
            V = clip(np.array([med(D, fn, "ML-BJS free-probe")]))[0]/max(med(D, fn, "p=0.2"), FLOOR)
            P = med(D, fn, "ML-BJS")/max(med(D, fn, "ML-BJS free-probe"), FLOOR)
            e1 = next(t for t in eq if t["D"] == D and t["fn"] == fn and t["e"] == "E1")["holm"]
            e2 = next(t for t in eq if t["D"] == D and t["fn"] == fn and t["e"] == "E2")["holm"]
            value[(D, fn)] = dict(pstar=pstar, gap=gap, V=V, P=P, VP=V*P, e1=e1, e2=e2)
            lines.append(" & ".join([f"{D}", fn.capitalize(), f"{pstar:g}", rfmt(gap), rfmt(V), rfmt(P), rfmt(V*P), pf(e1) + ('$^\\ddagger$' if e1 < 0.05 else ''), pf(e2) + ('$^\\ddagger$' if e2 < 0.05 else '')]) + " \\\\")
        if D == 2: lines.append("\\midrule")
    open("t7_value.tex", "w").write("\n".join(lines))

    # ---------------- larger budget ----------------
    lines = []
    for D in (2, 10):
        for fn in FUNCS:
            row = [f"{D}", fn.capitalize()] + [cell(med(D, fn, "B+ " + m)) for m in ("p=0.2", "ML-BJS", "ML-BJS free-probe", "history-BJS", "1/5-rule")]
            Pl = med(D, fn, "B+ ML-BJS")/max(med(D, fn, "B+ ML-BJS free-probe"), FLOOR); row.append(rfmt(Pl)); row.append(rfmt(value[(D, fn)]["P"]))
            lines.append(" & ".join(row) + " \\\\")
        if D == 2: lines.append("\\midrule")
    open("t8_large.tex", "w").write("\n".join(lines))

    # ---------------- realised information ----------------
    if os.path.exists("logging_out.json"):
        L = json.load(open("logging_out.json")); lines = []
        for D in (2, 10):
            for fn in FUNCS:
                r = L[f"D{D}_{fn}"]
                lines.append(" & ".join([f"{D}", fn.capitalize(), f"{r['ml_mean_p']:.2f}", f"{r['ml_std_p']:.2f}", f"{r['ml_frac_psi0']:.2f}", f"{r['ml_H_psi_bits']:.2f}", f"{r['ml_H_p_bits']:.2f}",
                                         f"{r['hist_mean_p']:.2f}", f"{r['hist_std_p']:.2f}", f"{r['hist_H_p_bits']:.2f}"]) + " \\\\")
            if D == 2: lines.append("\\midrule")
        open("t9_real.tex", "w").write("\n".join(lines))

    # ---------------- success rates (f <= 1e-2) ----------------
    SUC = {}
    for D in (2, 10):
        for fn in FUNCS:
            for m in ["p=0.0", "p=0.2", "p=1.0", "ML-BJS", "ML-BJS free-probe", "history-BJS", "decay-p", "1/5-rule"]:
                SUC[(D, fn, m)] = int((get(D, fn, m) <= 1e-2).sum())

    # ---------------- success table (final fitness <= 1e-2) ----------------
    SM = ["p=0.0", "p=0.2", "p=1.0", "ML-BJS", "ML-BJS free-probe", "history-BJS", "decay-p", "1/5-rule", "1/5sigma+p=0.2", "L3 matched+refine"]
    lines = []
    for D in (2, 10):
        for fn in FUNCS:
            c = [int((get(D, fn, m) <= 1e-2).sum()) for m in SM]; mx = max(c)
            lines.append(f"{D} & {fn.capitalize()} & " + " & ".join((f"\\textbf{{{v}}}" if v == mx and mx > 0 else str(v)) for v in c) + " \\\\")
        if D == 2: lines.append("\\midrule")
    open("t10_success.tex", "w").write("\n".join(lines))

    # ---------------- figures ----------------
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.3))
    for i, D in enumerate((2, 10)):
        for j, fn in enumerate(FUNCS):
            ax = axes[i, j]; xs = range(len(PS))
            m_ = [max(med(D, fn, f"p={p}"), FLOOR) for p in PS]; q = np.array([iqr(D, fn, f"p={p}") for p in PS]); q = np.maximum(q, FLOOR)
            ax.fill_between(xs, q[:, 0], q[:, 1], alpha=0.25, lw=0); ax.plot(xs, m_, marker="o", ms=2, lw=0.9)
            ax.axhline(max(med(D, fn, "ML-BJS"), FLOOR), color="r", ls="--", lw=0.7)
            ax.set_yscale("log"); ax.set_xticks([0, 4, 8]); ax.set_xticklabels(["0", "0.2", "1"], fontsize=6); ax.tick_params(axis="y", labelsize=6)
            ax.set_title(f"{fn}, $D={D}$", fontsize=6.5, pad=2)
    for ax in axes[1]: ax.set_xlabel("$p$", fontsize=7)
    fig.tight_layout(pad=0.4); fig.savefig("fig_sweep2.pdf")

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    pts = [("Level 1 (constant)", 0, 0, "k"), ("counter $t$ (L2)", 0, math.log2(2000), "g"), ("history (L2)", 0, math.log2(21), "g"),
           ("probes $D{=}2$ (L2)", 8*2000/9, math.log2(9), "b"), ("probes $D{=}10$ (L2)", 8*10000/9, math.log2(9), "b"),
           ("grid $D{=}2$ (L3)", 1681, 64*1681, "r"), ("grid $D{=}10$ (L3)", 41.0**10, 64*41.0**10, "r")]
    for lab, x, y, c in pts: ax.scatter([x], [y], c=c, s=14); ax.annotate(lab, (x, y), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=1); ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("acquisition cost per run (FEs)", fontsize=7); ax.set_ylabel("nominal capacity $\\kappa$ (bits)", fontsize=7)
    ax.tick_params(labelsize=6); ax.set_xlim(-0.5, 1e19); ax.set_ylim(-0.5, 1e19)
    fig.tight_layout(pad=0.4); fig.savefig("fig_info.pdf")

    # ---------------- facts for the text ----------------
    F = {}
    sig = [t for t in tests if t["holm"] < 0.05]
    F["n_tests"] = len(tests); F["n_sig"] = len(sig)
    for h, _, _ in HYP:
        th = [t for t in tests if t["h"] == h]
        F[h] = dict(n_sig=sum(t["holm"] < 0.05 for t in th), worse=[(t["D"], t["fn"], round(t["d"], 2)) for t in th if t["holm"] < 0.05 and t["d"] > 0],
                    better=[(t["D"], t["fn"], round(t["d"], 2)) for t in th if t["holm"] < 0.05 and t["d"] < 0], nonsig=[(t["D"], t["fn"], round(t["d"], 2), round(t["holm"], 3)) for t in th if t["holm"] >= 0.05])
    F["equiv"] = {e: [(t["D"], t["fn"], round(t["holm"], 3)) for t in eq if t["e"] == e and t["holm"] < 0.05] for e, _, _ in EQ}
    F["pstar"] = {f"{D}-{fn}": (value[(D, fn)]["pstar"], round(float(value[(D, fn)]["gap"]), 2)) for D in (2, 10) for fn in FUNCS}
    F["V_P_VP"] = {f"{D}-{fn}": [round(float(value[(D, fn)][k]), 3) for k in ("V", "P", "VP")] for D in (2, 10) for fn in FUNCS}
    F["hc_vs_p02"] = {f"{D}-{fn}": round(med(D, fn, "p=0.0")/max(med(D, fn, "p=0.2"), FLOOR), 3) for D in (2, 10) for fn in FUNCS}
    F["rnd_vs_p02"] = {f"{D}-{fn}": round(med(D, fn, "p=1.0")/max(med(D, fn, "p=0.2"), FLOOR), 3) for D in (2, 10) for fn in FUNCS}
    F["sigma_star"] = {f"{D}-{fn}": s for (D, fn), s in sig_star.items()}
    F["success"] = {f"{D}-{fn}-{m}": v for (D, fn, m), v in SUC.items() if D == 2 or fn in ("sphere", "rosenbrock")}
    F["train_range"] = {f"{D}-{fn}": [min([med(D, fn, "ML-BJS")] + [med(D, fn, f"ML-BJS train{j}") for j in range(1, 5)]), max([med(D, fn, "ML-BJS")] + [med(D, fn, f"ML-BJS train{j}") for j in range(1, 5)])] for D in (2, 10) for fn in FUNCS}
    F["sigma_p02"] = {f"{D}-{fn}": [round(med(D, fn, f"p=0.2,sigma={s}")/med(D, fn, "p=0.2"), 2) for s in (0.03, 0.3)] for D in (2, 10) for fn in FUNCS}
    F["L3_vs_p02"] = {f"{D}-{fn}": {m: round(med(D, fn, m)/max(med(D, fn, "p=0.2"), FLOOR), 3) for m in ["L3 grid", "L3 full+refine", "L3 matched+refine"] if has(D, fn, m)} for D in (2, 10) for fn in FUNCS}
    json.dump(F, open("facts.json", "w"), indent=1, default=str)
    N = 2000; A = 10.24**2; formula = A*(1-2**(-1/N))/math.pi; v = get(2, "sphere", "p=1.0"); rs = np.random.default_rng(0)
    bs = [np.median(rs.choice(v, len(v))) for _ in range(5000)]
    print("random-search formula", formula, "median", np.median(v), "bootstrap CI", np.percentile(bs, [2.5, 97.5]))
    return F


def make_fig_info():
    """Figure 1 of the paper (was fig_info.py)."""
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    pts = [("L1: constant", 0, 0, "k", (6, -2)), ("L2: counter $t$\n(11 bits)", 0, math.log2(2000), "g", (6, 6)), ("L2: history\n(4.4 bits)", 0, math.log2(21), "g", (6, -14)),
           ("L2: probes, $D{=}2$", 8*2000/9, math.log2(9), "b", (4, -12)), ("L2: probes, $D{=}10$", 8*10000/9, math.log2(9), "b", (4, 7)),
           ("L3: grid, $D{=}2$", 1681, 64*1681, "r", (6, -3)), ("L3: grid, $D{=}10$", 41.0**10, 64*41.0**10, "r", (-78, -12))]
    for lab, x, y, c, off in pts:
        ax.scatter([x], [y], c=c, s=14, zorder=3); ax.annotate(lab, (x, y), fontsize=5.5, xytext=off, textcoords="offset points")
    ax.set_xscale("symlog", linthresh=1); ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks([0, 1e3, 1e6, 1e12, 1e18]); ax.set_xticklabels(["0", "$10^3$", "$10^6$", "$10^{12}$", "$10^{18}$"], fontsize=6)
    ax.set_yticks([0, 10, 1e3, 1e6, 1e12, 1e18]); ax.set_yticklabels(["0", "10", "$10^3$", "$10^6$", "$10^{12}$", "$10^{18}$"], fontsize=6)
    ax.set_xlim(-0.3, 1e20); ax.set_ylim(-0.3, 1e20); ax.grid(alpha=0.25, lw=0.4)
    ax.set_xlabel("acquisition cost over a run (FEs)", fontsize=7); ax.set_ylabel("nominal capacity $\\kappa$ (bits)", fontsize=7)
    fig.tight_layout(pad=0.4); fig.savefig("fig_info.pdf"); plt.close(fig)

def make_summary_table():
    """Compact main table of the paper: median best fitness of each policy divided by that of constant p = 0.2.
    Reads the raw CSVs and facts.json written by analyze(); writes t_summary.tex and summary_ratios.csv."""
    FLOOR_ = 1e-12
    data = {}
    for name in ("raw_D2.csv", "raw_D10.csv", "raw2_D2.csv", "raw2_D10.csv"):
        for r in csv.DictReader(open(name)):
            data.setdefault((int(r["D"]), r["function"], r["method"]), []).append(float(r["best"]))
    med_ = lambda D, f, m: max(float(np.median(data[(D, f, m)])), FLOOR_)
    facts = json.load(open("facts.json"))

    def mark(h, D, f):
        if any(D == a and f == b for a, b, _ in facts[h]["better"]): return r"$^{\ast}$"
        if any(D == a and f == b for a, b, _ in facts[h]["worse"]): return r"$^{\dagger}$"
        return ""

    def base_fmt(x):
        mant, ex = f"{x:.1e}".split("e"); ex = int(ex)
        return f"{x:.2g}" if -2 <= ex <= 2 else f"${mant}\\times10^{{{ex}}}$"

    def rfmt_(x):
        if x < 1e-3: return r"$<$0.001"
        if x >= 1000: return r"$>$1000"
        return f"{x:.0f}" if x >= 100 else f"{x:.2g}"

    cols = [("Best-$\\sigma$ HC", None, None), ("1/5 rule", "1/5-rule", "H4"), ("History", "history-BJS", "H3"),
            ("Decay", "decay-p", "H8"), ("ML-BJS", "ML-BJS", "H1"), ("Grid+ref.", "L3 matched+refine", None)]
    lines, table = [], []
    for D in (2, 10):
        for f in ("sphere", "rastrigin", "ackley", "griewank", "rosenbrock"):
            base = med_(D, f, "p=0.2")
            sigma = facts["sigma_star"][f"{D}-{f}"]
            row = [str(D), f.capitalize(), base_fmt(base)]
            rec = dict(D=D, function=f, p02_median=base)
            for lab, m, h in cols:
                if m is None:
                    m = "p=0.0" if sigma == 0.1 else f"p=0.0,sigma={sigma}"
                ratio = med_(D, f, m) / base
                s = rfmt_(ratio)
                s = f"\\textbf{{{s}}}" if ratio < 0.95 else s   # bold: median at least 5% lower than p = 0.2
                row.append(s + (mark(h, D, f) if h else ""))
                rec[lab] = ratio
            lines.append(" & ".join(row) + " \\\\")
            table.append(rec)
        if D == 2: lines.append("\\midrule")
    open("t_summary.tex", "w").write("\n".join(lines))
    import pandas as _pd
    _pd.DataFrame(table).to_csv("summary_ratios.csv", index=False)

# ============================================================================
# MAIN: run everything in the original order
# ============================================================================
def run_pool(fn, jobs, label):
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    res, t0 = [], time.time()
    step = max(1, len(jobs) // 5)
    with ctx.Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs), 1):
            res.append(r)
            if i % step == 0 or i == len(jobs):
                log(f"{label}: {i}/{len(jobs)} tasks")
    return res


def write_raw(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["D", "function", "method", "seed", "best", "FEs"]); w.writerows(rows)


def fetch_reference(name):
    local = os.path.join(REF_DIR, name) if REF_DIR else ""
    if local and os.path.exists(local):
        return local
    dest = os.path.join("reference", name)
    if not os.path.exists(dest):
        os.makedirs("reference", exist_ok=True)
        urllib.request.urlretrieve(REF_BASE + name, dest)
    return dest


def verify_against_published():
    """Compare regenerated raw results with the CSVs published on GitHub."""
    log("Checking regenerated results against the published raw CSVs ...")
    report = []
    for name in ("raw_D2.csv", "raw_D10.csv", "raw2_D2.csv", "raw2_D10.csv"):
        try:
            ref_path = fetch_reference(name)
        except Exception as e:
            report.append(f"{name}: could not download reference ({e.__class__.__name__}); skipped")
            continue
        def load(p):
            d = {}
            for r in csv.DictReader(open(p)):
                d[(r["function"], r["method"], int(r["seed"]))] = (float(r["best"]), int(r["FEs"]))
            return d
        a, b = load(name), load(ref_path)
        keys = set(a) | set(b)
        same = sum(1 for k in keys if k in a and k in b and a[k][1] == b[k][1] and
                   abs(a[k][0] - b[k][0]) <= 1e-9 * max(abs(b[k][0]), 1e-12) + 1e-15)
        report.append(f"{name}: {same}/{len(keys)} runs identical to the published results")
    for line in report:
        log(line)
    open("verification.txt", "w").write("\n".join(report))


def main():
    global SEEDS
    if QUICK:
        SEEDS = 2
    os.makedirs(OUT, exist_ok=True)
    os.chdir(OUT)
    log(f"workers={WORKERS}, instances per cell={SEEDS}, QUICK={QUICK}")
    log("numpy", np.__version__, "| scipy", __import__("scipy").__version__, "| scikit-learn", __import__("sklearn").__version__)

    # ---- Round 1 (H1-H4) ----
    for D in DIMS:
        table = train_phi(D)
        log(f"Round 1, D={D}: learned probe table {np.round(table, 3)}")
        jobs = [(D, n, s, table) for n in FUNCS for s in range(SEEDS)]
        rows = [r for chunk in run_pool(task_round1, jobs, f"round 1 D={D}") for r in chunk]
        write_raw(f"raw_D{D}.csv", rows)

    # ---- Round 2 (H5-H8, ablations, Level 3, larger budget) ----
    for D in DIMS:
        tabs = {"orig": train_phi(D)}
        for j in range(1, 5):
            tabs[j] = train_table(D, [(FUNCS["sphere"], 0.05), (FUNCS["rastrigin"], 0.8)], 12345 + D + 1000*j, j)[0]
        tabs["disj"] = train_table(D, [(zakharov, 0.05), (levy, 0.8)], 777 + D, 0)[0]
        jobs = [(D, n, s, tabs) for n in FUNCS for s in range(SEEDS)]
        rows = [r for chunk in run_pool(task_round2, jobs, f"round 2 D={D}") for r in chunk]
        write_raw(f"raw2_D{D}.csv", rows)

    # ---- realised-information diagnostics ----
    out = {f"train_D{D}": logging_train_record(D) for D in DIMS}
    for key, rec in run_pool(logging_cell, [(D, fn) for D in DIMS for fn in FUNCS], "diagnostics"):
        out[key] = rec
    json.dump(out, open("logging_out.json", "w"), indent=1)

    # ---- analysis, tables, figures ----
    log("Analysis ...")
    analyze()
    make_fig_info()
    make_summary_table()

    if VERIFY and not QUICK:
        verify_against_published()

    with zipfile.ZipFile("../results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(".")):
            if os.path.isfile(f):
                z.write(f)
    log("Finished. All outputs are in ./results and results.zip")
    os.chdir("..")
    try:
        from google.colab import files
        files.download("results.zip")
    except Exception:
        pass


if __name__ == "__main__":
    main()

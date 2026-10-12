"""
=============================================================================
 EXTENSION CELL: run the pipeline ONLY for NEW benchmark functions, append the
 results to the existing raw CSVs, and re-run the whole analysis on the
 combined dataset.   (Paste into a NEW Colab cell, run after the first script.)
=============================================================================
 * Works from the current folder if it already is ./results, or from ./results
   (or /content/results) otherwise.  Nothing is deleted or regenerated for the
   old functions.
 * Budgets, seeds, instance shifts, policy tables and random streams are
   identical to run_all_experiments.py.  The random stream of a function is
   keyed by its index in the ordered dict {old 5 functions, then NEW_FUNCS},
   i.e. the old functions keep indices 0-4 and the new ones get 5, 6, ...
   ==> DO NOT REORDER NEW_FUNCS after the first run (append new ones at the end).
 * Safe to re-run: finished functions are detected and skipped, new rows are
   staged in ./staging first and only then appended (header written only if a
   file is missing), and the original CSVs are backed up once to
   ./backup_before_new_funcs.
 * Runtime: same per-function cost as in the first script: roughly 13 min on Colab's
   2 cores for 5 new functions.
=============================================================================
"""
import os, sys, csv, json, math, time, shutil, hashlib, zipfile
import multiprocessing as mp
from collections import deque
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

WORKERS = os.cpu_count() or 1
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

# ----------------------------- constants (unchanged) -----------------------------
LO, HI, SIGMA, K = -5.12, 5.12, 0.1, 8
SEEDS, DIMS = 30, (2, 10)
BUDGET = lambda D: 1000 * D

# ---- the five ORIGINAL functions (needed for the ML training set, indices 0-4) ----
FUNCS_OLD = {
 "sphere":    lambda x: np.sum(x**2, axis=-1),
 "rastrigin": lambda x: 10*x.shape[-1] + np.sum(x**2 - 10*np.cos(2*np.pi*x), axis=-1),
 "ackley":    lambda x: (-20*np.exp(-0.2*np.sqrt(np.mean(x**2, axis=-1)))
                         - np.exp(np.mean(np.cos(2*np.pi*x), axis=-1)) + 20 + np.e),
 "griewank":  lambda x: (1 + np.sum(x**2, axis=-1)/4000
                         - np.prod(np.cos(x/np.sqrt(np.arange(1, x.shape[-1]+1))), axis=-1)),
 "rosenbrock":lambda x: np.sum(100*(x[...,1:]-x[...,:-1]**2)**2 + (1-x[...,:-1])**2, axis=-1),
}

# ---------------------------------- NEW functions ----------------------------------
# All are vectorised over the last axis, defined for any D >= 2, have global minimum 0
# at the origin (the pipeline then moves it to a random point in [-2,2]^D per instance,
# exactly as for the old functions), and are clamped at 0 against round-off.
def _nonneg(f):
    return lambda x: np.maximum(f(x), 0.0)

def _schwefel222(x):                       # Schwefel 2.22 (Schwefel 2.26 has its optimum outside the domain)
    a = np.abs(x); return np.sum(a, axis=-1) + np.prod(a, axis=-1)

def _weierstrass(x, a=0.5, b=3.0, kmax=20):
    k = np.arange(kmax + 1); ak, bk = a**k, b**k
    t = np.sum(ak*np.cos(2*np.pi*bk*(x[..., None] + 0.5)), axis=(-1, -2))
    return t - x.shape[-1]*np.sum(ak*np.cos(2*np.pi*bk*0.5))

def _happycat(x):                          # optimum (-1,..,-1) moved to the origin by y = x - 1
    y = x - 1.0; s = np.sum(y**2, axis=-1); D = x.shape[-1]
    return np.abs(s - D)**0.25 + (0.5*s + np.sum(y, axis=-1))/D + 0.5

def _salomon(x):
    r = np.sqrt(np.sum(x**2, axis=-1)); return 1 - np.cos(2*np.pi*r) + 0.1*r

def _schaffer2(x):                         # Schaffer N.2; for D>2 summed over consecutive pairs (D=2: exact N.2)
    a, b = x[..., :-1], x[..., 1:]
    return np.sum(0.5 + (np.sin(a**2 - b**2)**2 - 0.5)/(1 + 0.001*(a**2 + b**2))**2, axis=-1)

NEW_FUNCS = {                              # append only; never reorder (random streams depend on the position)
 "schwefel":   _nonneg(_schwefel222),
 "weierstrass":_nonneg(_weierstrass),
 "happycat":   _nonneg(_happycat),
 "salomon":    _nonneg(_salomon),
 "schaffer2":  _nonneg(_schaffer2),
}
DISPLAY = {"schwefel": "Schwefel 2.22", "happycat": "HappyCat", "schaffer2": "Schaffer N.2"}   # table labels
def L(fn): return DISPLAY.get(fn, fn.capitalize())

ALL_FUNCS = {**FUNCS_OLD, **NEW_FUNCS}
assert len(ALL_FUNCS) == len(FUNCS_OLD) + len(NEW_FUNCS), "name clash between old and new functions"
FIDX = {n: i for i, n in enumerate(ALL_FUNCS)}
assert [FIDX[n] for n in FUNCS_OLD] == [0, 1, 2, 3, 4]

def shifted(name, shift):
    f = ALL_FUNCS[name]
    return lambda x: f(x - shift)

class Counted:
    def __init__(self, f): self.f, self.n = f, 0
    def __call__(self, x): self.n += 1; return float(self.f(x))
    def many(self, X): self.n += len(X); return self.f(X)

# ------------------------------- Level-2 policies (unchanged) -------------------------------
def psi(fvec, x, fx, rng):
    probes = np.clip(x + rng.normal(0, SIGMA, (K, len(x))), LO, HI)
    return np.mean(fvec(probes) < fx)

def train_phi(D):
    rng = np.random.default_rng(12345 + D)
    X, y = [], []
    for name, lab in (("sphere", 0.05), ("rastrigin", 0.8)):
        for _ in range(300):
            x = rng.uniform(LO, HI, D)
            X.append([psi(FUNCS_OLD[name], x, FUNCS_OLD[name](x), rng)]); y.append(lab)
    rf = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
    return rf.predict([[j/K] for j in range(K+1)])

def train_table(D, fl, rng_seed, rf_state):
    rng = np.random.default_rng(rng_seed); X, y = [], []
    for fn, lab in fl:
        for _ in range(300):
            x = rng.uniform(LO, HI, D); X.append([psi(fn, x, float(fn(x)), rng)]); y.append(lab)
    rf = RandomForestRegressor(n_estimators=30, random_state=rf_state).fit(X, y)
    return rf.predict([[j/K] for j in range(K+1)])

class Policy:
    cost = 1
    def p(self, ctx): raise NotImplementedError
class Const(Policy):
    def __init__(self, p): self.v = p
    def p(self, ctx): return self.v
class History(Policy):
    def p(self, ctx):
        h = ctx["hist"]
        return 0.2 if not h else 0.05 + 0.5*(1 - np.mean(h))
class MLProbe(Policy):
    cost = 1 + K
    def __init__(self, table): self.t = table
    def p(self, ctx):
        z = psi(lambda X: ctx["f"].many(X), ctx["x"], ctx["fx"], ctx["rng"])
        return self.t[int(round(K*z))]
class MLFree(Policy):
    def __init__(self, table, raw): self.t, self.raw = table, raw
    def p(self, ctx):
        z = psi(self.raw, ctx["x"], ctx["fx"], ctx["rng"])
        return self.t[int(round(K*z))]
class Decay(Policy):
    def __init__(self, B, p0=0.5): self.B, self.p0 = B, p0
    def p(self, ctx): return self.p0*max(0.0, 1 - ctx["f"].n/self.B)
class HistLow(Policy):
    def p(self, ctx):
        h = ctx["hist"]; return 0.1 if not h else 0.02 + 0.2*(1 - np.mean(h))

# ------------------------------------ engines (unchanged) ------------------------------------
def search(f, D, B, policy, rng, sigma=SIGMA):
    x = rng.uniform(LO, HI, D); fx = f(x); hist = deque(maxlen=20)
    while f.n + policy.cost <= B:
        p = policy.p(dict(f=f, x=x, fx=fx, hist=hist, rng=rng))
        c = rng.uniform(LO, HI, D) if rng.random() < p else np.clip(x + rng.normal(0, sigma, D), LO, HI)
        fc = f(c); ok = fc < fx
        if ok: x, fx = c, fc
        hist.append(ok)
    return fx

def one_fifth(f, D, B, rng):
    x = rng.uniform(LO, HI, D); fx = f(x); s = SIGMA
    while f.n < B:
        c = np.clip(x + rng.normal(0, s, D), LO, HI); fc = f(c)
        if fc < fx: x, fx, s = c, fc, s*1.5
        else: s *= 1.5**-0.25
    return fx

def grid_table(fvec, D, n=41):
    ax = np.linspace(LO, HI, n)
    G = np.stack(np.meshgrid(*[ax]*D, indexing="ij"), -1)
    return float(fvec(G).min()), n**D

def search_adapt_sigma(f, D, B, pol, rng):
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

def search_ml_reuse(f, D, B, table, rng):
    x = rng.uniform(LO, HI, D); fx = f(x)
    while f.n + 1 + K <= B:
        P = np.clip(x + rng.normal(0, SIGMA, (K, D)), LO, HI); fp = f.many(P); j = int(np.argmin(fp))
        if fp[j] < fx: x, fx = P[j], float(fp[j]); continue
        c = _step(x, table[0], D, rng); fc = f(c)
        if fc < fx: x, fx = c, fc
    return fx

def search_ml_every(f, D, B, table, rng, m=4):
    x = rng.uniform(LO, HI, D); fx = f(x); p = 0.2; it = 0
    while True:
        need = (1 + K) if it % m == 0 else 1
        if f.n + need > B: break
        if it % m == 0: p = table[int(round(K*psi(lambda X: f.many(X), x, fx, rng)))]
        c = _step(x, p, D, rng); fc = f(c)
        if fc < fx: x, fx = c, fc
        it += 1
    return fx

def l3_refine(f, D, B, rng, n, centres):
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

def zakharov(x):
    s = np.sum(0.5*np.arange(1, x.shape[-1]+1)*x, axis=-1)
    return np.sum(x**2, axis=-1) + s**2 + s**4
def levy(x):
    w = 1 + (x-1)/4
    return (np.sin(np.pi*w[..., 0])**2
            + np.sum((w[..., :-1]-1)**2*(1+10*np.sin(np.pi*w[..., :-1]+1)**2), axis=-1)
            + (w[..., -1]-1)**2*(1+np.sin(2*np.pi*w[..., -1])**2))

# ---------------------------- Round 1 / Round 2 tasks (same streams) ----------------------------
PS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
SIG = (0.003, 0.01, 0.03, 0.3, 1.0)
LARGE = {2: 5, 10: 3}

def task_round1(args):
    D, name, seed, table = args
    fidx = FIDX[name]
    rs = np.random.default_rng([seed, D, fidx])
    shift = rs.uniform(-2, 2, D)
    raw = shifted(name, shift)
    B = BUDGET(D); rows = []
    def run(label, fn, **kw):
        f = Counted(raw); val = fn(f); rows.append((D, name, label, seed, val, f.n))
    for p in PS:
        lab = f"p={p}"; r = np.random.default_rng([seed, D, fidx, PS.index(p)])
        run(lab, lambda f: search(f, D, B, Const(p), r))
    for s in (0.03, 0.3):
        r = np.random.default_rng([seed, D, fidx, 100 + int(s*100)])
        run(f"p=0.2,sigma={s}", lambda f: search(f, D, B, Const(0.2), r, sigma=s))
    r = np.random.default_rng([seed, D, fidx, 200]); run("ML-BJS", lambda f: search(f, D, B, MLProbe(table), r))
    r = np.random.default_rng([seed, D, fidx, 201]); run("ML-BJS free-probe", lambda f: search(f, D, B, MLFree(table, raw), r))
    r = np.random.default_rng([seed, D, fidx, 202]); run("history-BJS", lambda f: search(f, D, B, History(), r))
    r = np.random.default_rng([seed, D, fidx, 203]); run("1/5-rule", lambda f: one_fifth(f, D, B, r))
    if D == 2:
        v, n = grid_table(raw, D); rows.append((D, name, "L3 grid", seed, v, n))
    return rows

def task_round2(args):
    D, name, seed, tabs = args
    fidx = FIDX[name]
    shift = np.random.default_rng([seed, D, fidx]).uniform(-2, 2, D)
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
    BL = B*LARGE[D]
    r = rng_next(); run("B+ p=0.2", lambda f, b: search(f, D, b, Const(0.2), r), BL)
    r = rng_next(); run("B+ ML-BJS", lambda f, b: search(f, D, b, MLProbe(t0), r), BL)
    r = rng_next(); run("B+ ML-BJS free-probe", lambda f, b: search(f, D, b, MLFree(t0, raw), r), BL)
    r = rng_next(); run("B+ history-BJS", lambda f, b: search(f, D, b, History(), r), BL)
    r = rng_next(); run("B+ 1/5-rule", lambda f, b: one_fifth(f, D, b, r), BL)
    return rows

def make_tabs(D):
    tabs = {"orig": train_phi(D)}
    for j in range(1, 5):
        tabs[j] = train_table(D, [(FUNCS_OLD["sphere"], 0.05), (FUNCS_OLD["rastrigin"], 0.8)], 12345 + D + 1000*j, j)
    tabs["disj"] = train_table(D, [(zakharov, 0.05), (levy, 0.8)], 777 + D, 0)
    return tabs

# ----------------------------- realised-information diagnostics -----------------------------
def entropy(vals):
    _, c = np.unique(vals, return_counts=True); p = c/c.sum(); return float(-(p*np.log2(p)).sum())

class Logged(Policy):
    def __init__(self, inner): self.inner, self.cost, self.ps = inner, inner.cost, []
    def p(self, ctx):
        v = self.inner.p(ctx); self.ps.append(v); return v

class MLProbeLog(MLProbe):
    def __init__(self, t): super().__init__(t); self.ps, self.zs = [], []
    def p(self, ctx):
        z = psi(lambda Xp: ctx["f"].many(Xp), ctx["x"], ctx["fx"], ctx["rng"])
        j = int(round(K*z)); self.zs.append(j); self.ps.append(self.t[j]); return self.t[j]

def logging_cell(args):
    D, fn = args
    table = train_phi(D); B = BUDGET(D); fidx = FIDX[fn]
    ml_z, ml_p, hi_p = [], [], []
    for seed in range(SEEDS):
        shift = np.random.default_rng([seed, D, fidx]).uniform(-2, 2, D); raw = shifted(fn, shift)
        r = np.random.default_rng([seed, D, fidx, 900])
        pol = MLProbeLog(table); f = Counted(raw); search(f, D, B, pol, r)
        ml_z.append(np.array(pol.zs)); ml_p.append(np.array(pol.ps))
        hist_pol = Logged(History()); f = Counted(raw); search(f, D, B, hist_pol, np.random.default_rng([seed, D, fidx, 901]))
        hi_p.append(np.array(hist_pol.ps))
    allp = np.concatenate(ml_p); zvis = np.concatenate(ml_z); hp = np.concatenate(hi_p)
    rec = dict(ml_mean_p=float(np.mean(allp)), ml_std_p=float(np.std(allp)), ml_H_p_bits=entropy(np.round(allp/0.05)*0.05),
               ml_iters_per_run=float(np.mean([len(a) for a in ml_p])), ml_H_psi_bits=entropy(zvis), ml_frac_psi0=float(np.mean(zvis == 0)),
               hist_mean_p=float(np.mean(hp)), hist_std_p=float(np.std(hp)), hist_H_p_bits=entropy(np.round(hp/0.05)*0.05),
               hist_mean_p_first_quarter=float(np.mean([a[:max(1, len(a)//4)].mean() for a in hi_p])),
               hist_mean_p_last_quarter=float(np.mean([a[-max(1, len(a)//4):].mean() for a in hi_p])))
    return f"D{D}_{fn}", rec

# =============================== ANALYSIS (dynamic function list) ===============================
BASE_ORDER = ["sphere", "rastrigin", "ackley", "griewank", "rosenbrock"]
RAW_FILES = ("raw_D2.csv", "raw_D10.csv", "raw2_D2.csv", "raw2_D10.csv")

def read_data():
    DATA, order = {}, []
    for fn_ in RAW_FILES:
        with open(fn_, newline="") as fh:
            for r in csv.DictReader(fh):
                DATA.setdefault((int(r["D"]), r["function"], r["method"]), {})[int(r["seed"])] = (float(r["best"]), int(r["FEs"]))
                if r["function"] not in order: order.append(r["function"])
    return DATA, order

def discover_funcs(DATA, order):
    """All functions present in the CSVs (old first, then new in file order) that are complete in every
    (D, method) cell of the reference function 'sphere'; incomplete ones are reported and skipped."""
    funcs = [f for f in BASE_ORDER if f in order] + [f for f in order if f not in BASE_ORDER]
    ref = {D: {m for (d, f, m) in DATA if d == D and f == "sphere"} for D in (2, 10)}
    ok = []
    for f in funcs:
        miss = [(D, m) for D in (2, 10) for m in ref[D] if (D, f, m) not in DATA]
        short = [(D, m, len(DATA[(D, f, m)])) for D in (2, 10) for m in ref[D] if (D, f, m) in DATA and len(DATA[(D, f, m)]) != SEEDS]
        if miss or short:
            print(f"WARNING: skipping '{f}' in the analysis (missing cells: {miss[:3]}..., wrong seed count: {short[:3]}...)")
        else:
            ok.append(f)
    return ok

def analyze():
    FLOOR = 1e-12
    DATA, order = read_data()
    FUNCS = discover_funcs(DATA, order)
    print("analysing functions:", FUNCS)
    def get(D, fn, m): return np.array([DATA[(D, fn, m)][s][0] for s in sorted(DATA[(D, fn, m)])])
    def clip(v): return np.maximum(v, FLOOR)
    def med(D, fn, m): return float(np.median(get(D, fn, m)))
    def iqr(D, fn, m): return np.percentile(get(D, fn, m), [25, 75])
    def has(D, fn, m): return (D, fn, m) in DATA
    PS_ = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    SIGS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    def hc(s): return "p=0.0" if s == 0.1 else f"p=0.0,sigma={s}"

    for (D, fn, m), d in DATA.items():                      # budget check
        B = 1000*D*({2: 5, 10: 3}[D] if m.startswith("B+") else 1)
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
        if r <= 0: return "$0$"
        def sci(v): e = int(math.floor(math.log10(v))); return f"${v/10**e:.1f}\\times10^{{{e}}}$"
        if r < 0.01 or r >= 1000: return sci(r)
        return f"{r:.2f}" if r < 10 else f"{r:.0f}"
    DAG, DDAG = "$^\\dagger$", "$^\\ddagger$"

    MAIN = [("p=0.0", "$p{=}0$"), ("p=0.2", "$p{=}0.2$"), ("p=1.0", "$p{=}1$"), ("ML-BJS", "ML-BJS"), ("history-BJS", "Hist-BJS"), ("decay-p", "Decay-$p$"), ("1/5-rule", "1/5-rule")]
    for D in (2, 10):
        lines = []
        for fn in FUNCS:
            meds = {m: med(D, fn, m) for m, _ in MAIN}; best = min(meds, key=lambda m: max(meds[m], FLOOR))
            l1 = L(fn); l2 = "\\hspace{1em}IQR"
            for m, _ in MAIN:
                l1 += " & " + cell(meds[m], m == best); q = iqr(D, fn, m)
                l2 += f" & $[{fmt(q[0],3,2)},{fmt(q[1],3,2)}]$"
            lines += [l1 + " \\\\", l2 + " \\\\"]
        open(f"t2_main_D{D}.tex", "w").write("\n".join(lines))

    VAR = [("ML-BJS free-probe", "free-probe"), ("ML-BJS reuse", "reuse"), ("ML-BJS probe/4", "probe/4"), (None, "train 0--4"), ("ML-BJS disjoint", "disjoint"),
           ("history-BJS low", "Hist low"), ("1/5sigma+p=0.2", "1/5$\\sigma$+0.2"), (None, "best-$\\sigma$ HC")]
    lines = []; sig_star = {}
    for D in (2, 10):
        for fn in FUNCS:
            row = [f"{D}", L(fn)]
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

    lines = []
    for D in (2, 10):
        for fn in FUNCS:
            cols = [("L3 grid", D == 2), ("L3 full+refine", D == 2), ("L3 matched+refine", True), ("p=0.0", True), ("p=0.2", True)]
            l1 = f"{D} & {L(fn)}"
            for m, ok in cols: l1 += " & " + (cell(med(D, fn, m)) if ok else "---")
            lines.append(l1 + " \\\\")
        if D == 2: lines.append("\\midrule")
    open("t5_l3.tex", "w").write("\n".join(lines))

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
        m = len(ps); order_ = np.argsort(ps); adj = np.empty(m); run = 0
        for rank, i in enumerate(order_): run = max(run, (m-rank)*ps[i]); adj[i] = min(1.0, run)
        return adj
    adj = holm([t["p"] for t in tests])          # Holm family = ALL tests over ALL functions
    for t, a in zip(tests, adj): t["holm"] = float(a)
    def pf(p): return "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0") if p < 1 else "1"
    lines = []
    for D in (2, 10):
        for fn in FUNCS:
            row = [f"{D}", L(fn)]
            for h, _, _ in HYP:
                t = next(t for t in tests if t["D"] == D and t["fn"] == fn and t["h"] == h)
                row.append(f"{t['d']:+.1f} ({pf(t['holm'])})" + (DAG if t["holm"] < 0.05 else ""))
            lines.append(" & ".join(row) + " \\\\")
        if D == 2: lines.append("\\midrule")
    open("t6_tests.tex", "w").write("\n".join(lines))

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

    lines = []; value = {}
    for D in (2, 10):
        for fn in FUNCS:
            pstar = min(PS_, key=lambda p: max(med(D, fn, f"p={p}"), FLOOR)); gap = max(med(D, fn, "p=0.2"), FLOOR)/max(med(D, fn, f"p={pstar}"), FLOOR)
            V = clip(np.array([med(D, fn, "ML-BJS free-probe")]))[0]/max(med(D, fn, "p=0.2"), FLOOR)
            P = med(D, fn, "ML-BJS")/max(med(D, fn, "ML-BJS free-probe"), FLOOR)
            e1 = next(t for t in eq if t["D"] == D and t["fn"] == fn and t["e"] == "E1")["holm"]
            e2 = next(t for t in eq if t["D"] == D and t["fn"] == fn and t["e"] == "E2")["holm"]
            value[(D, fn)] = dict(pstar=pstar, gap=gap, V=V, P=P, VP=V*P, e1=e1, e2=e2)
            lines.append(" & ".join([f"{D}", L(fn), f"{pstar:g}", rfmt(gap), rfmt(V), rfmt(P), rfmt(V*P), pf(e1) + (DDAG if e1 < 0.05 else ""), pf(e2) + (DDAG if e2 < 0.05 else "")]) + " \\\\")
        if D == 2: lines.append("\\midrule")
    open("t7_value.tex", "w").write("\n".join(lines))

    lines = []
    for D in (2, 10):
        for fn in FUNCS:
            row = [f"{D}", L(fn)] + [cell(med(D, fn, "B+ " + m)) for m in ("p=0.2", "ML-BJS", "ML-BJS free-probe", "history-BJS", "1/5-rule")]
            Pl = med(D, fn, "B+ ML-BJS")/max(med(D, fn, "B+ ML-BJS free-probe"), FLOOR); row.append(rfmt(Pl)); row.append(rfmt(value[(D, fn)]["P"]))
            lines.append(" & ".join(row) + " \\\\")
        if D == 2: lines.append("\\midrule")
    open("t8_large.tex", "w").write("\n".join(lines))

    if os.path.exists("logging_out.json"):
        LG = json.load(open("logging_out.json")); lines = []
        for D in (2, 10):
            for fn in FUNCS:
                if f"D{D}_{fn}" not in LG:
                    print(f"WARNING: no diagnostics for D={D}, {fn}; row omitted from t9_real.tex"); continue
                r = LG[f"D{D}_{fn}"]
                lines.append(" & ".join([f"{D}", L(fn), f"{r['ml_mean_p']:.2f}", f"{r['ml_std_p']:.2f}", f"{r['ml_frac_psi0']:.2f}", f"{r['ml_H_psi_bits']:.2f}", f"{r['ml_H_p_bits']:.2f}",
                                         f"{r['hist_mean_p']:.2f}", f"{r['hist_std_p']:.2f}", f"{r['hist_H_p_bits']:.2f}"]) + " \\\\")
            if D == 2: lines.append("\\midrule")
        open("t9_real.tex", "w").write("\n".join(lines))

    SUC = {}
    for D in (2, 10):
        for fn in FUNCS:
            for m in ["p=0.0", "p=0.2", "p=1.0", "ML-BJS", "ML-BJS free-probe", "history-BJS", "decay-p", "1/5-rule"]:
                SUC[(D, fn, m)] = int((get(D, fn, m) <= 1e-2).sum())
    SM = ["p=0.0", "p=0.2", "p=1.0", "ML-BJS", "ML-BJS free-probe", "history-BJS", "decay-p", "1/5-rule", "1/5sigma+p=0.2", "L3 matched+refine"]
    lines = []
    for D in (2, 10):
        for fn in FUNCS:
            c = [int((get(D, fn, m) <= 1e-2).sum()) for m in SM]; mx = max(c)
            lines.append(f"{D} & {L(fn)} & " + " & ".join((f"\\textbf{{{v}}}" if v == mx and mx > 0 else str(v)) for v in c) + " \\\\")
        if D == 2: lines.append("\\midrule")
    open("t10_success.tex", "w").write("\n".join(lines))

    # ---- figure: sweep over p, 5 panels per row, rows grow with the number of functions ----
    nF = len(FUNCS); ncol = 5; rpd = math.ceil(nF/ncol); nrow = 2*rpd
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 1.65*nrow), squeeze=False)
    for ax in axes.ravel(): ax.set_visible(False)
    for i, D in enumerate((2, 10)):
        for j, fn in enumerate(FUNCS):
            ax = axes[i*rpd + j//ncol, j % ncol]; ax.set_visible(True); xs = range(len(PS_))
            m_ = [max(med(D, fn, f"p={p}"), FLOOR) for p in PS_]; q = np.array([iqr(D, fn, f"p={p}") for p in PS_]); q = np.maximum(q, FLOOR)
            ax.fill_between(xs, q[:, 0], q[:, 1], alpha=0.25, lw=0); ax.plot(xs, m_, marker="o", ms=2, lw=0.9)
            ax.axhline(max(med(D, fn, "ML-BJS"), FLOOR), color="r", ls="--", lw=0.7)
            ax.set_yscale("log"); ax.set_xticks([0, 4, 8]); ax.set_xticklabels(["0", "0.2", "1"], fontsize=6); ax.tick_params(axis="y", labelsize=6)
            ax.set_title(f"{fn}, $D={D}$", fontsize=6.5, pad=2)
        for ax in axes[i*rpd + rpd - 1]:
            if ax.get_visible(): ax.set_xlabel("$p$", fontsize=7)
    fig.tight_layout(pad=0.4); fig.savefig("fig_sweep2.pdf"); plt.close(fig)

    # ---- facts ----
    F = {}
    F["functions"] = FUNCS
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
    F["success"] = {f"{D}-{fn}-{m}": v for (D, fn, m), v in SUC.items()}
    F["train_range"] = {f"{D}-{fn}": [min([med(D, fn, "ML-BJS")] + [med(D, fn, f"ML-BJS train{j}") for j in range(1, 5)]), max([med(D, fn, "ML-BJS")] + [med(D, fn, f"ML-BJS train{j}") for j in range(1, 5)])] for D in (2, 10) for fn in FUNCS}
    F["sigma_p02"] = {f"{D}-{fn}": [round(med(D, fn, f"p=0.2,sigma={s}")/max(med(D, fn, "p=0.2"), FLOOR), 2) for s in (0.03, 0.3)] for D in (2, 10) for fn in FUNCS}
    F["L3_vs_p02"] = {f"{D}-{fn}": {m: round(med(D, fn, m)/max(med(D, fn, "p=0.2"), FLOOR), 3) for m in ["L3 grid", "L3 full+refine", "L3 matched+refine"] if has(D, fn, m)} for D in (2, 10) for fn in FUNCS}
    json.dump(F, open("facts.json", "w"), indent=1, default=str)
    if has(2, "sphere", "p=1.0"):
        N = 2000; A = 10.24**2; formula = A*(1-2**(-1/N))/math.pi; v = get(2, "sphere", "p=1.0"); rs = np.random.default_rng(0)
        bs = [np.median(rs.choice(v, len(v))) for _ in range(5000)]
        print("random-search formula", formula, "median", np.median(v), "bootstrap CI", np.percentile(bs, [2.5, 97.5]))
    return F

def make_fig_info():
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
    FLOOR_ = 1e-12
    DATA, order = read_data(); funcs = discover_funcs(DATA, order)
    data = {k: [v[s][0] for s in sorted(v)] for k, v in DATA.items()}
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
        for f in funcs:
            base = med_(D, f, "p=0.2"); sigma = facts["sigma_star"][f"{D}-{f}"]
            row = [str(D), L(f), base_fmt(base)]; rec = dict(D=D, function=f, p02_median=base)
            for lab, m, h in cols:
                if m is None: m = "p=0.0" if sigma == 0.1 else f"p=0.0,sigma={sigma}"
                ratio = med_(D, f, m) / base; s = rfmt_(ratio)
                s = f"\\textbf{{{s}}}" if ratio < 0.95 else s
                row.append(s + (mark(h, D, f) if h else "")); rec[lab] = ratio
            lines.append(" & ".join(row) + " \\\\"); table.append(rec)
        if D == 2: lines.append("\\midrule")
    open("t_summary.tex", "w").write("\n".join(lines))
    pd.DataFrame(table).to_csv("summary_ratios.csv", index=False)

# ====================================== driver ======================================
def run_pool(fn, jobs, label):
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    res = []; step = max(1, len(jobs)//5)
    with ctx.Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs), 1):
            res.append(r)
            if i % step == 0 or i == len(jobs): log(f"{label}: {i}/{len(jobs)} tasks")
    return res

HEADER = ["D", "function", "method", "seed", "best", "FEs"]

def write_stage(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(HEADER); w.writerows(rows)
    os.replace(tmp, path)

def functions_in(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0: return set()
    with open(path, newline="") as fh: return {r["function"] for r in csv.DictReader(fh)}

def append_rows(target, staged):
    """Append staged rows (for functions not yet present in target) in 'a' mode; header only if target is new/empty."""
    have = functions_in(target)
    with open(staged, newline="") as fh:
        rd = csv.reader(fh); next(rd); rows = [r for r in rd if r[1] not in have]
    if not rows:
        log(f"{target}: nothing to append"); return 0
    new_file = (not os.path.exists(target)) or os.path.getsize(target) == 0
    add_nl = False
    if not new_file:
        with open(target, "rb") as fh: fh.seek(-1, os.SEEK_END); add_nl = fh.read(1) not in (b"\n", b"\r")
    with open(target, "a", newline="") as fh:
        if add_nl: fh.write("\r\n")
        w = csv.writer(fh)
        if new_file: w.writerow(HEADER)
        w.writerows(rows)
    log(f"{target}: appended {len(rows)} rows")
    return len(rows)

def locate_results():
    if os.path.exists("raw_D2.csv"): return os.getcwd()
    for c in ("results", "/content/results"):
        if os.path.exists(os.path.join(c, "raw_D2.csv")):
            os.chdir(c); return os.getcwd()
    raise FileNotFoundError("raw_D2.csv not found in the current folder, ./results or /content/results")

def sanity_check_engine():
    """Re-run one old (function, seed) and compare with the stored CSV: confirms that this cell's engine and the
    installed library versions reproduce the original streams."""
    table = train_phi(2)
    new = task_round1((2, "sphere", 0, table))
    ref = {}
    with open("raw_D2.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["function"] == "sphere" and int(r["seed"]) == 0: ref[r["method"]] = (float(r["best"]), int(r["FEs"]))
    same = sum(1 for (_, _, m, _, v, n) in new if m in ref and ref[m][1] == n and abs(ref[m][0] - v) <= 1e-9*max(abs(ref[m][0]), 1e-12) + 1e-15)
    log(f"engine check on old data (sphere, D=2, seed 0): {same}/{len(new)} runs identical to the stored results")
    if same != len(new): log("WARNING: mismatch - library versions differ from the original run; new rows may not be stream-compatible")

def run_extension():
    start_dir = os.getcwd()
    rdir = locate_results(); log("working in", rdir)
    log("numpy", np.__version__, "| scipy", __import__("scipy").__version__, "| scikit-learn", __import__("sklearn").__version__)
    sanity_check_engine()

    # which new functions still need computing (missing from at least one raw file or from the diagnostics)?
    lg_have = set(json.load(open("logging_out.json")).keys()) if os.path.exists("logging_out.json") else set()
    def todo(fn):
        return (any(fn not in functions_in(p) for p in RAW_FILES)
                or any(f"D{D}_{fn}" not in lg_have for D in DIMS))
    to_run = [fn for fn in NEW_FUNCS if todo(fn)]
    log("new functions to run:", to_run if to_run else "none (already in the CSVs)")

    if to_run:
        tag = hashlib.md5(",".join(to_run).encode()).hexdigest()[:8]
        st = lambda name: os.path.join("staging", f"{tag}_{name}")
        for D in DIMS:                                              # Round 1
            if not os.path.exists(st(f"raw_D{D}.csv")):
                table = train_phi(D); log(f"Round 1, D={D}: learned probe table {np.round(table, 3)}")
                jobs = [(D, n, s, table) for n in to_run for s in range(SEEDS)]
                rows = [r for chunk in run_pool(task_round1, jobs, f"round 1 D={D}") for r in chunk]
                write_stage(st(f"raw_D{D}.csv"), rows)
        for D in DIMS:                                              # Round 2
            if not os.path.exists(st(f"raw2_D{D}.csv")):
                tabs = make_tabs(D)
                jobs = [(D, n, s, tabs) for n in to_run for s in range(SEEDS)]
                rows = [r for chunk in run_pool(task_round2, jobs, f"round 2 D={D}") for r in chunk]
                write_stage(st(f"raw2_D{D}.csv"), rows)
        lg_path = st("logging.json")                                # diagnostics
        if not os.path.exists(lg_path):
            out = dict(run_pool(logging_cell, [(D, fn) for D in DIMS for fn in to_run], "diagnostics"))
            json.dump(out, open(lg_path, "w"), indent=1)

        # backup once, then append
        if not os.path.exists("backup_before_new_funcs"):
            os.makedirs("backup_before_new_funcs")
            for p in RAW_FILES + ("logging_out.json",):
                if os.path.exists(p): shutil.copy(p, "backup_before_new_funcs")
            log("backed up the original CSV/JSON files to ./backup_before_new_funcs")
        for D in DIMS:
            append_rows(f"raw_D{D}.csv", st(f"raw_D{D}.csv"))
            append_rows(f"raw2_D{D}.csv", st(f"raw2_D{D}.csv"))
        LG = json.load(open("logging_out.json")) if os.path.exists("logging_out.json") else {}
        LG.update(json.load(open(lg_path)))
        json.dump(LG, open("logging_out.json", "w"), indent=1)

    log("Re-running analysis on the combined data ...")
    analyze(); make_fig_info(); make_summary_table()

    with zipfile.ZipFile("../results_extended.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(".")):
            if os.path.isfile(f): z.write(f)
    log("Finished. Updated tables (t*.tex), facts.json, fig_sweep2.pdf, fig_info.pdf are in", os.getcwd(), "and ../results_extended.zip")
    os.chdir(start_dir)
    try:
        from google.colab import files
        files.download("results_extended.zip")
    except Exception:
        pass

if __name__ == "__main__":
    run_extension()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tai_experiments_v2.py  --  evidence-generating experiments for the BJS / ML-BJS paper (TAI revision)
=================================================================================================
Run in Google Colab:   !python tai_experiments_v2.py        (or paste into one cell)
Smoke test (minutes):  set QUICK = True.   DO NOT read numbers from a QUICK run.

RELATION TO THE GITHUB REPOSITORY (matinkhah/three-level-heuristics)
--------------------------------------------------------------------
The synthetic suite (EXP-B) reuses the repository's *exact* conventions, so that its numbers are
the same numbers as in the published CSVs, not a second, different set:
  * instance shifts:  default_rng([seed, D, fidx]).uniform(-2, 2, D) in raw coordinates, seeds 0..29
  * method streams:   default_rng([seed, D, fidx, key]) with the repository's keys
                      (p=0.05:2, p=0.2:4, p=0.5:5, p=1:6, ML-BJS:200, free:201, history:202,
                       reuse:311, training replicates 1..4: 313..316)
  * learned table:    train_phi / train_table (30 trees, random_state=0, default_rng(12345+D), 300 pts/function)
  * engines/policies: copied from run_all_experiments.py (including the repository's probe-reuse semantics)
  * function order:   old five, then Schwefel2.22, Weierstrass, HappyCat, Salomon, SchafferN2 (indices 0..9)
After EXP-B the script CHECKS itself against results_extended.zip (published raw CSVs) and prints how
many runs are bit-identical.  If that check does not say "identical" for all rows, do not use EXP-B numbers
until the cause is found (library versions are the usual cause).

New in this script (not in the repository): the non-learned rule on the same probe signal, a budget sweep
(0.1B .. 2B), a real-ML task (MLP hyper-parameter optimisation), Optuna TPE, a small-budget D=5 comparison,
and extended diagnostics.  Their streams use new keys (>= 1001) so they never collide with the repository's.

WHAT IT RUNS
------------
EXP-B  Synthetic suite: 10 shifted functions, D = 2 and 10, B = 1000 D, 30 paired instances.
       Learned controller (ML) and NON-LEARNED rule on the same probe signal, each in three variants:
         charged (ch)   probes are charged to B                     (budget-fair)
         free           probes evaluated but not charged             (counterfactual)
         reuse          an improving probe becomes the incumbent     (repository semantics)
       plus rs (p=1), c0.05 / c0.2 (baseline) / c0.5, history-BJS, and the ML controller retrained on 4 other
       training streams (ml_ch_t1..t4).  Holm-corrected paired Wilcoxon tests (one family per question),
       price/value ratios, budget sweep with break-even, check on the 8 functions not used in training.
EXP-A  Real ML task: MLP on scikit-learn digits, 4 hyper-parameters (lr, L2, width, batch size),
       B = 60 training runs, 12 instances = different random train/validation splits.
       rs, c0.05, c0.2, c0.5, history, ML charged / reuse, rule charged / reuse, Optuna TPE.
       Reports validation error, held-out TEST error (fixed test set), and anytime curves.
EXP-C  Small-budget comparison with Optuna TPE: D = 5, B = 200, 10 shifted instances per function.
DIAG   Realised information R vs nominal capacity kappa, probe-state visitation in search vs in training,
       mutual information of the probe feature with the training label, controller training cost vs B.

ACCOUNTING (state exactly like this in the paper)
-------------------------------------------------
* every objective evaluation made during search is charged to B: candidates AND probes ('ch' variants);
* 'free' variants evaluate probes without charging them (counterfactual, not budget-fair);
* controller TRAINING evaluations are NOT charged to B but ARE counted and reported (b_learn);
* decision compute is not charged; wall-clock is measured and reported;  no energy is measured.
* EXP-B budget sweep: runs use 2B evaluations and the best-so-far at b = 0.1B..2B is read from the trace.
  Because every engine is a deterministic function of its random stream, this equals a separate run with
  budget b (a probe-reuse move is time-stamped at the full cost of its iteration to keep that exact).

OUTPUT:  ./tai_results/ (CSVs, tests, JSON) and tai_results.zip.  The last thing printed is the block between
=====BEGIN PASTE BLOCK===== and =====END PASTE BLOCK=====.  Paste that block back into the chat.
"""

# ----------------------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------------------
QUICK = False            # True = tiny smoke test; False = full run

import os
CFG = dict(
    OUT="tai_results",
    WORKERS=None,                              # None = all cores; results do not depend on this
    # EXP-B synthetic suite
    SEEDS=30, DIMS=(2, 10), B_PER_DIM=1000,
    SWEEP=(0.1, 0.25, 0.5, 1.0, 2.0),          # budget multipliers; max(SWEEP) sets the run length
    TRAIN_REPS=4,                              # extra training streams for the learned table (keys 313..316)
    # EXP-A real ML task
    HPO_INST=12, HPO_B=60, HPO_SIGMA=0.08, HPO_EPOCHS=25,
    HPO_RUN_FREE=False,                        # free-probe variants cost real training runs; off
    HPO_TPE=True, HPO_GP=False,                # GP sampler needs torch; slow; UNTESTED; off by default
    # EXP-C small-budget BO comparison
    BO_SYN=True, BO_TPE=True, BO_DIM=5, BO_B=200, BO_INST=10,
    TPE_STARTUP=10,
    # statistics / verification
    ALPHA=0.05,
    VERIFY=True,
    REF_ZIP=os.environ.get("REF_ZIP", ""),     # optional local path of results_extended.zip
    REF_URL="https://raw.githubusercontent.com/matinkhah/three-level-heuristics/main/results_extended.zip",
)
if QUICK:
    CFG.update(SEEDS=3, HPO_INST=2, HPO_B=12, HPO_EPOCHS=10, BO_INST=2, BO_B=40)

# ----------------------------------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------------------------------
import sys, io, csv, json, math, time, shutil, subprocess, warnings, zipfile, urllib.request
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import multiprocessing as mp
from collections import deque
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

OUT = CFG["OUT"]
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

_OPT = None
def _optuna():
    """Import Optuna lazily (installs it if missing)."""
    global _OPT
    if _OPT is None:
        try:
            import optuna
        except Exception:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna"], check=False)
            import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        _OPT = optuna
    return _OPT

# ----------------------------------------------------------------------------------
# 2. BENCHMARKS (repository definitions; vectorised over the last axis; global minimum 0)
# ----------------------------------------------------------------------------------
LO, HI, SIGMA, K = -5.12, 5.12, 0.1, 8
def _nonneg(f): return lambda x: np.maximum(f(x), 0.0)
def _schwefel222(x):
    a = np.abs(x); return np.sum(a, axis=-1) + np.prod(a, axis=-1)
def _weierstrass(x, a=0.5, b=3.0, kmax=20):
    k = np.arange(kmax + 1); ak, bk = a**k, b**k
    t = np.sum(ak*np.cos(2*np.pi*bk*(x[..., None] + 0.5)), axis=(-1, -2))
    return t - x.shape[-1]*np.sum(ak*np.cos(2*np.pi*bk*0.5))
def _happycat(x):
    y = x - 1.0; s = np.sum(y**2, axis=-1); D = x.shape[-1]
    return np.abs(s - D)**0.25 + (0.5*s + np.sum(y, axis=-1))/D + 0.5
def _salomon(x):
    r = np.sqrt(np.sum(x**2, axis=-1)); return 1 - np.cos(2*np.pi*r) + 0.1*r
def _schaffer2(x):
    a, b = x[..., :-1], x[..., 1:]
    return np.sum(0.5 + (np.sin(a**2 - b**2)**2 - 0.5)/(1 + 0.001*(a**2 + b**2))**2, axis=-1)

FUNCS = {   # ORDER MATTERS: the index is part of every random stream key
 "sphere":     lambda x: np.sum(x**2, axis=-1),
 "rastrigin":  lambda x: 10*x.shape[-1] + np.sum(x**2 - 10*np.cos(2*np.pi*x), axis=-1),
 "ackley":     lambda x: (-20*np.exp(-0.2*np.sqrt(np.mean(x**2, axis=-1)))
                          - np.exp(np.mean(np.cos(2*np.pi*x), axis=-1)) + 20 + np.e),
 "griewank":   lambda x: (1 + np.sum(x**2, axis=-1)/4000
                          - np.prod(np.cos(x/np.sqrt(np.arange(1, x.shape[-1]+1))), axis=-1)),
 "rosenbrock": lambda x: np.sum(100*(x[..., 1:]-x[..., :-1]**2)**2 + (1-x[..., :-1])**2, axis=-1),
 "schwefel":    _nonneg(_schwefel222),
 "weierstrass": _nonneg(_weierstrass),
 "happycat":    _nonneg(_happycat),
 "salomon":     _nonneg(_salomon),
 "schaffer2":   _nonneg(_schaffer2),
}
FNAMES = list(FUNCS); FIDX = {n: i for i, n in enumerate(FNAMES)}
TRAIN_NAMES = ("sphere", "rastrigin")
TRAIN_FL = [(FUNCS["sphere"], 0.05), (FUNCS["rastrigin"], 0.8)]
DISPLAY = {"schwefel": "Schwefel2.22", "happycat": "HappyCat", "schaffer2": "SchafferN2",
           "weierstrass": "Weierstrass", "salomon": "Salomon"}
def disp(n): return DISPLAY.get(n, n.capitalize())

def shifted(name, shift):
    f = FUNCS[name]; return lambda x: f(x - shift)

# ----------------------------------------------------------------------------------
# 3. ENGINE AND POLICIES (repository code, generalised only by an explicit search box)
# ----------------------------------------------------------------------------------
class Counted:
    def __init__(self, f, fm=None): self.f, self.fm, self.n = f, (fm or f), 0
    def __call__(self, x): self.n += 1; return float(self.f(x))
    def many(self, X): self.n += len(X); return self.fm(X)

def psi(fvec, x, fx, rng, box):
    lo, hi, sg = box
    probes = np.clip(x + rng.normal(0, sg, (K, len(x))), lo, hi)
    return np.mean(fvec(probes) < fx)

RULE_TABLE = 0.05 + 0.75*(1 - np.arange(K + 1)/K)     # NON-LEARNED rule on the same probe signal

class Policy:
    cost = 1
class Const(Policy):
    def __init__(self, p): self.v = p
    def p(self, ctx): return self.v
class History(Policy):
    def __init__(self): self.ps, self.ns = [], []
    def p(self, ctx):
        h = ctx["hist"]
        v = 0.2 if not h else 0.05 + 0.5*(1 - np.mean(h))
        self.ps.append(float(v)); self.ns.append(ctx["f"].n); return v
class Probe(Policy):
    """charged (default): the K probes are counted; free: probes evaluated by free_fvec, not counted."""
    def __init__(self, table, free_fvec=None):
        self.t = np.asarray(table); self.free = free_fvec
        self.cost = 1 if free_fvec is not None else 1 + K
        self.zs, self.ps, self.ns = [], [], []
    def p(self, ctx):
        fv = self.free if self.free is not None else (lambda X: ctx["f"].many(X))
        self.ns.append(ctx["f"].n)
        z = psi(fv, ctx["x"], ctx["fx"], ctx["rng"], ctx["box"])
        j = int(round(K*z)); v = self.t[j]
        self.zs.append(j); self.ps.append(float(v)); return v

def search(f, D, B, policy, rng, box, tr):
    lo, hi, sg = box
    x = rng.uniform(lo, hi, D); fx = f(x); hist = deque(maxlen=20); tr.append((f.n, fx))
    while f.n + policy.cost <= B:
        p = policy.p(dict(f=f, x=x, fx=fx, hist=hist, rng=rng, box=box))
        c = rng.uniform(lo, hi, D) if rng.random() < p else np.clip(x + rng.normal(0, sg, D), lo, hi)
        fc = f(c); ok = fc < fx
        if ok: x, fx = c, fc; tr.append((f.n, fx))
        hist.append(ok)
    return fx, x

def _step(x, p, D, rng, box):
    lo, hi, sg = box
    return rng.uniform(lo, hi, D) if rng.random() < p else np.clip(x + rng.normal(0, sg, D), lo, hi)

def search_reuse(f, D, B, table, rng, box, tr):
    """Repository semantics: an improving probe is accepted as the move (the candidate step is skipped);
    if no probe improves, a normal step is taken with p = table[0]."""
    lo, hi, sg = box
    x = rng.uniform(lo, hi, D); fx = f(x); tr.append((f.n, fx))
    while f.n + 1 + K <= B:
        P = np.clip(x + rng.normal(0, sg, (K, D)), lo, hi); fp = f.many(P); j = int(np.argmin(fp))
        if fp[j] < fx:
            x, fx = P[j], float(fp[j])
            tr.append((f.n + 1, fx))     # time-stamped at the full iteration cost (see module docstring)
            continue
        c = _step(x, table[0], D, rng, box); fc = f(c)
        if fc < fx: x, fx = c, fc; tr.append((f.n, fx))
    return fx, x

def run_tpe(fscalar, D, B, lo, hi, seed, tr, startup, sampler="tpe"):
    optuna = _optuna()
    if sampler == "tpe":
        smp = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(startup, max(1, B // 4)))
    else:
        smp = optuna.samplers.GPSampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=smp)
    cnt, best = [0], [np.inf]
    def obj(trial):
        u = np.array([trial.suggest_float(f"x{i}", lo, hi) for i in range(D)])
        v = float(fscalar(u)); cnt[0] += 1
        if v < best[0]: best[0] = v; tr.append((cnt[0], v))
        return v
    study.optimize(obj, n_trials=B, show_progress_bar=False)
    bp = study.best_params
    return best[0], np.array([bp[f"x{i}"] for i in range(D)]), cnt[0]

def value_at(tr, b):
    ns = np.fromiter((e[0] for e in tr), dtype=float)
    i = np.searchsorted(ns, b, side="right") - 1
    return tr[i][1] if i >= 0 else float("nan")

# method keys: the repository's keys where a method exists there, new keys (>= 1001) otherwise
KEY = {"rs": 6, "c0.05": 2, "c0.2": 4, "c0.5": 5, "hist": 202,
       "ml_ch": 200, "ml_free": 201, "ml_reuse": 311,
       "ml_ch_t1": 313, "ml_ch_t2": 314, "ml_ch_t3": 315, "ml_ch_t4": 316,
       "rule_ch": 1001, "rule_free": 1002, "rule_reuse": 1003, "tpe": 1010, "gp": 1011}

def run_method(m, f, D, B, rng, box, tabs, raw_vec, tr):
    """returns (fx, x, policy_or_None)"""
    if m == "rs":              pol = Const(1.0)
    elif m.startswith("c"):    pol = Const(float(m[1:]))
    elif m == "hist":          pol = History()
    else:
        parts = m.split("_"); fam, mode = parts[0], parts[1]
        rep = int(parts[2][1:]) if len(parts) > 2 else None
        table = tabs["rule"] if fam == "rule" else (tabs[rep] if rep else tabs["orig"])
        if mode == "reuse":
            fx, x = search_reuse(f, D, B, table, rng, box, tr); return fx, x, None
        pol = Probe(table, free_fvec=(raw_vec if mode == "free" else None))
    fx, x = search(f, D, B, pol, rng, box, tr)
    return fx, x, pol

# ----------------------------------------------------------------------------------
# 4. LEARNED CONTROLLER (repository training procedure) AND ITS DIAGNOSTICS
# ----------------------------------------------------------------------------------
def entropy(vals):
    vals = np.asarray(vals)
    if vals.size == 0: return float("nan")
    _, c = np.unique(vals, return_counts=True); p = c/c.sum(); return float(-(p*np.log2(p)).sum()) + 0.0

def mutual_info(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ja = {v: i for i, v in enumerate(np.unique(a))}; jb = {v: i for i, v in enumerate(np.unique(b))}
    M = np.zeros((len(ja), len(jb)))
    for u, v in zip(a, b): M[ja[u], jb[v]] += 1
    M /= M.sum(); pa = M.sum(1, keepdims=True); pb = M.sum(0, keepdims=True)
    nz = M > 0; return float((M[nz]*np.log2(M[nz]/(pa@pb)[nz])).sum())

def train_table(D, fl, rng_seed, rf_state, sigma=SIGMA):
    rng = np.random.default_rng(rng_seed); X, y = [], []
    for fn, lab in fl:
        for _ in range(300):
            x = rng.uniform(LO, HI, D); X.append([psi(fn, x, float(fn(x)), rng, (LO, HI, sigma))]); y.append(lab)
    rf = RandomForestRegressor(n_estimators=30, random_state=rf_state).fit(X, y)
    return rf.predict([[j/K] for j in range(K + 1)]), np.array(X)[:, 0], np.array(y)

def make_tabs(D, sigma_train=SIGMA, reps=0):
    tab, X, y = train_table(D, TRAIN_FL, 12345 + D, 0, sigma_train)      # == repository train_phi(D) when sigma=0.1
    tabs = {"orig": tab, "rule": RULE_TABLE.copy()}
    for j in range(1, reps + 1):
        tabs[j] = train_table(D, TRAIN_FL, 12345 + D + 1000*j, j, sigma_train)[0]
    zt = np.round(X*K).astype(int)
    info = dict(D=D, sigma_train=sigma_train, table=[float(t) for t in tab],
                train_state=(np.bincount(zt, minlength=K + 1)/len(zt)).tolist(),
                H_feature_bits=entropy(zt), MI_feature_label_bits=mutual_info(zt, (y > 0.4).astype(int)),
                frac_state0=float((zt == 0).mean()), b_learn_FEs=int(2*300*(K + 1)), kappa_bits=float(np.log2(K + 1)))
    return tabs, info

# ----------------------------------------------------------------------------------
# 5. SYNTHETIC TASKS (EXP-B and EXP-C)
# ----------------------------------------------------------------------------------
B_METHODS = ["rs", "c0.05", "c0.2", "c0.5", "hist", "ml_ch", "ml_free", "ml_reuse",
             "rule_ch", "rule_free", "rule_reuse"]
def b_methods(reps): return B_METHODS + [f"ml_ch_t{j}" for j in range(1, reps + 1)]

def synth_task(args):
    D, name, seed, B, tabs, methods, mults, startup = args
    fidx = FIDX[name]
    shift = np.random.default_rng([seed, D, fidx]).uniform(-2, 2, D)
    raw = shifted(name, shift); box = (LO, HI, SIGMA)
    Bmax = int(round(B*max(mults))); rows, diags = [], []
    for m in methods:
        rng = np.random.default_rng([seed, D, fidx, KEY[m]])
        tr = []; pol = None; t0 = time.perf_counter()
        if m in ("tpe", "gp"):
            fx, x, _ = run_tpe(lambda u: raw(u), D, Bmax, LO, HI, int(rng.integers(1 << 30)), tr, startup, m)
        else:
            f = Counted(raw)
            fx, x, pol = run_method(m, f, D, Bmax, rng, box, tabs, raw, tr)
        wall = time.perf_counter() - t0
        row = dict(D=D, func=name, method=m, seed=seed, B=B, wall_s=wall)
        for fr in mults: row[f"v{fr}"] = value_at(tr, int(round(B*fr)))
        rows.append(row)
        if pol is not None and getattr(pol, "ps", None):
            ps = np.array(pol.ps); ns = np.array(pol.ns); keep = (ns + pol.cost) <= B      # only iterations inside B
            if keep.any():
                dg = dict(D=D, func=name, method=m, seed=seed,
                          pb=np.bincount(np.round(ps[keep]/0.05).astype(int), minlength=21).tolist(),
                          pmean=float(ps[keep].mean()), psd=float(ps[keep].std()))
                if hasattr(pol, "zs") and pol.zs:
                    dg["z"] = np.bincount(np.array(pol.zs)[keep], minlength=K + 1).tolist()
                diags.append(dg)
    return rows, diags

# ----------------------------------------------------------------------------------
# 6. REAL ML TASK (EXP-A)
# ----------------------------------------------------------------------------------
_DIG = None
def _digits():
    global _DIG
    if _DIG is None:
        X, y = load_digits(return_X_y=True)
        perm = np.random.RandomState(12345).permutation(len(y))
        _DIG = (X, y, perm[:360], perm[360:])            # fixed held-out TEST set; the rest is the pool for splits
    return _DIG

def hpo_setup(inst, epochs):
    X, y, te, pool = _digits()
    rs = np.random.RandomState(1000 + inst)               # the instance = a random train/validation split
    pp = rs.permutation(pool); tr, va = pp[:700], pp[700:1100]
    sc = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
    ytr, yva, yte = y[tr], y[va], y[te]
    def fit(u):
        lr = 10**(-4 + 3*u[0]); alpha = 10**(-6 + 5*u[1])
        width = int(round(8*2**(5*u[2]))); bs = int(round(16*2**(4*u[3])))
        clf = MLPClassifier(hidden_layer_sizes=(width,), alpha=alpha, learning_rate_init=lr,
                            batch_size=bs, max_iter=epochs, random_state=inst)   # fixed init => noise-free objective
        clf.fit(Xtr, ytr); return clf
    val = lambda u: 1.0 - fit(u).score(Xva, yva)
    test = lambda u: 1.0 - fit(u).score(Xte, yte)
    vec = lambda U: np.array([val(u) for u in U])
    return val, test, vec

def hpo_task(args):
    inst, m, B, tabs, sigma, epochs, startup = args
    D = 4; box = (0.0, 1.0, sigma)
    val, test, vec = hpo_setup(inst, epochs)
    rng = np.random.default_rng([inst, D, 4242, KEY[m]])
    tr = []; pol = None; t0 = time.perf_counter(); f = Counted(val, vec)
    if m in ("tpe", "gp"):
        fx, x, evals = run_tpe(val, D, B, 0.0, 1.0, int(rng.integers(1 << 30)), tr, startup, m)
    else:
        fx, x, pol = run_method(m, f, D, B, rng, box, tabs, vec, tr); evals = f.n
    wall = time.perf_counter() - t0
    row = dict(inst=inst, method=m, wall_s=wall, evals=evals, val=value_at(tr, B), test=test(x))
    for b in range(5, B + 1, 5): row[f"a{b}"] = value_at(tr, b)
    dg = None
    if pol is not None and getattr(pol, "ps", None):
        dg = dict(method=m, pb=np.bincount(np.round(np.array(pol.ps)/0.05).astype(int), minlength=21).tolist(),
                  pmean=float(np.mean(pol.ps)), psd=float(np.std(pol.ps)))
        if getattr(pol, "zs", None): dg["z"] = np.bincount(pol.zs, minlength=K + 1).tolist()
    return row, dg

# ----------------------------------------------------------------------------------
# 7. STATISTICS
# ----------------------------------------------------------------------------------
def holm(pv):
    p = np.asarray(pv, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank)*p[i]); adj[i] = min(1.0, run)
    return adj

def wil(x, y):
    try: return float(stats.wilcoxon(x, y).pvalue)
    except ValueError: return 1.0

def paired_table(wide, pairs, floor):
    """wide: {setting: DataFrame(index=instance, columns=method)}; pairs: [(A, B, family)].
    Wilcoxon on floored raw values (as in the repository); effect = median log10(A/B), positive => A worse.
    Holm is applied within each family."""
    recs = []
    for st, W in wide.items():
        for A, B_, fam in pairs:
            if A not in W.columns or B_ not in W.columns: continue
            a = np.maximum(W[A].values.astype(float), floor); b = np.maximum(W[B_].values.astype(float), floor)
            recs.append(dict(setting=st, A=A, B=B_, fam=fam, effect=float(np.median(np.log10(a) - np.log10(b))),
                             p=wil(a, b), medA=float(np.median(a)), medB=float(np.median(b)),
                             a_wins=int((a < b).sum()), b_wins=int((b < a).sum())))
    T = pd.DataFrame(recs)
    if len(T):
        T["p_holm"] = np.nan
        for fam, g in T.groupby("fam"): T.loc[g.index, "p_holm"] = holm(g["p"].values)
    return T

def sig_counts(T, alpha):
    out = []
    for (fam, A, B_), g in T.groupby(["fam", "A", "B"], sort=False):
        sg = g["p_holm"] < alpha
        out.append(dict(fam=fam, A=A, B=B_, n=len(g), n_sig=int(sg.sum()),
                        A_sig_better=int(((g.effect < 0) & sg).sum()), A_sig_worse=int(((g.effect > 0) & sg).sum()),
                        med_effect=round(float(g.effect.median()), 3)))
    return pd.DataFrame(out)

def g3(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "nan"
    return f"{x:.3g}"

# ----------------------------------------------------------------------------------
# 8. VERIFICATION AGAINST THE PUBLISHED RAW CSVs
# ----------------------------------------------------------------------------------
VERIFY_MAP = {"rs": "p=1.0", "c0.05": "p=0.05", "c0.2": "p=0.2", "c0.5": "p=0.5", "hist": "history-BJS",
              "ml_ch": "ML-BJS", "ml_free": "ML-BJS free-probe", "ml_reuse": "ML-BJS reuse",
              "ml_ch_t1": "ML-BJS train1", "ml_ch_t2": "ML-BJS train2",
              "ml_ch_t3": "ML-BJS train3", "ml_ch_t4": "ML-BJS train4"}

def verify_against_published(DB):
    try:
        if CFG["REF_ZIP"] and os.path.exists(CFG["REF_ZIP"]): raw = open(CFG["REF_ZIP"], "rb").read()
        else: raw = urllib.request.urlopen(CFG["REF_URL"], timeout=60).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except Exception as e:
        return [f"could not obtain the published results ({e.__class__.__name__}: {e}); verification SKIPPED"], None
    ref = {}
    for nm in ("raw_D2.csv", "raw_D10.csv", "raw2_D2.csv", "raw2_D10.csv"):
        names = [n for n in zf.namelist() if n.endswith(nm)]
        if not names: continue
        for r in csv.DictReader(io.TextIOWrapper(zf.open(names[0]))):
            ref[(int(r["D"]), r["function"], r["method"], int(r["seed"]))] = float(r["best"])
    inv = {v: k for k, v in VERIFY_MAP.items()}
    tot = same = 0; bad = []
    for _, r in DB.iterrows():
        lab = VERIFY_MAP.get(r["method"])
        if lab is None: continue
        k = (int(r["D"]), r["func"], lab, int(r["seed"]))
        if k not in ref: continue
        tot += 1; v = float(r["v1.0"]); w = ref[k]
        if abs(v - w) <= 1e-9*max(abs(w), 1e-12) + 1e-15: same += 1
        elif len(bad) < 5: bad.append((k, v, w))
    lines = [f"{same}/{tot} runs bit-identical to the published results "
             f"(methods compared: {sorted(VERIFY_MAP)})"]
    if tot == 0: lines.append("WARNING: no overlapping rows found; check REF_ZIP / seeds")
    for b in bad: lines.append(f"  MISMATCH {b[0]}: this run {b[1]!r} vs published {b[2]!r}")
    return lines, (same, tot)

# ----------------------------------------------------------------------------------
# 9. MAIN
# ----------------------------------------------------------------------------------
def run_pool(fn, jobs, label, workers):
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    res, step = [], max(1, len(jobs)//5)
    with ctx.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=1), 1):
            res.append(r)
            if i % step == 0 or i == len(jobs): log(f"{label}: {i}/{len(jobs)} tasks")
    return res

def diag_lines(DG, dims, tinfo, budget_of, label):
    """DG: list of dicts (D, method, pb, [z], pmean, psd)."""
    out = []
    kap = {"ml_ch": np.log2(K + 1), "rule_ch": np.log2(K + 1), "hist": np.log2(21)}
    for D in dims:
        for m in ("ml_ch", "rule_ch", "hist"):
            g = [x for x in DG if x["D"] == D and x["method"] == m]
            if not g: continue
            pb = np.sum([x["pb"] for x in g], axis=0)
            Rb = entropy(np.repeat(np.arange(len(pb)), pb.astype(int)))
            s = f"{label} D={D} {m}: R={Rb:.2f} bits (kappa={kap[m]:.2f}); mean p={np.mean([x['pmean'] for x in g]):.3f}, sd p={np.mean([x['psd'] for x in g]):.3f}"
            if "z" in g[0]:
                z = np.sum([x["z"] for x in g], axis=0); sd = z/z.sum()
                s += f"; search state dist={[round(float(v), 3) for v in sd]}; frac state0 (search)={sd[0]:.3f}"
                if D in tinfo:
                    td = np.array(tinfo[D]["train_state"])
                    s += f"; train state dist={[round(float(v), 3) for v in td]}; frac state0 (train)={td[0]:.3f}; TV(search,train)={0.5*np.abs(sd-td).sum():.3f}"
            out.append(s)
    return out

def main():
    import platform
    os.makedirs(OUT, exist_ok=True)
    W = CFG["WORKERS"] or (os.cpu_count() or 1)
    alpha = CFG["ALPHA"]
    paste = []; P = paste.append
    P(f"CONFIG: {json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items() if k != 'REF_URL'})}  QUICK={QUICK}")
    P(f"VERSIONS: python {platform.python_version()} numpy {np.__version__} scipy {__import__('scipy').__version__} "
      f"sklearn {__import__('sklearn').__version__} pandas {pd.__version__}; workers={W}")
    if CFG["HPO_TPE"] or (CFG["BO_SYN"] and CFG["BO_TPE"]):
        try:
            _optuna(); import optuna as _o; P(f"optuna {_o.__version__}")
        except Exception as e:
            log("WARNING: optuna unavailable; TPE comparisons are SKIPPED:", e)
            CFG["HPO_TPE"] = False; CFG["BO_TPE"] = False; P("optuna UNAVAILABLE: TPE rows are missing")
    dims, reps, SW = CFG["DIMS"], CFG["TRAIN_REPS"], CFG["SWEEP"]

    # ---- controllers -------------------------------------------------------------
    log("training controllers ...")
    tabs, tinfo = {}, {}
    for D in sorted(set(dims) | ({CFG["BO_DIM"]} if CFG["BO_SYN"] else set())):
        tabs[D], tinfo[D] = make_tabs(D, SIGMA, reps)
    tabs_hpo, tinfo_hpo = make_tabs(4, CFG["HPO_SIGMA"]*(HI - LO), 0)     # same RELATIVE step as the HPO search
    json.dump({"synthetic": tinfo, "hpo": tinfo_hpo}, open(f"{OUT}/controller_training.json", "w"), indent=1, default=float)
    P("\n## CONTROLLER TRAINING (repository procedure: Sphere target p=0.05, Rastrigin target p=0.8; 300 uniform points each; NOT charged to B)")
    for D, inf in list(tinfo.items()) + [(4, tinfo_hpo)]:
        P(f"D={D}: kappa={inf['kappa_bits']:.2f} bits; train feature entropy={inf['H_feature_bits']:.2f} bits; "
          f"MI(feature,label)={inf['MI_feature_label_bits']:.3f} bits; b_learn={inf['b_learn_FEs']} FEs; "
          f"table p(psi=0..8)={[round(t, 3) for t in inf['table']]}; train state dist={[round(v, 3) for v in inf['train_state']]}")
    P(f"non-learned rule table p(psi=0..8)={[round(float(t), 3) for t in RULE_TABLE]}  (p = 0.05 + 0.75 (1 - psi); b_learn = 0)")

    # ---- EXP-B ---------------------------------------------------------------------
    log("EXP-B synthetic suite ...")
    mB = b_methods(reps)
    jobs = [(D, fn, s, 1000*D, tabs[D], mB, SW, CFG["TPE_STARTUP"]) for D in dims for fn in FNAMES for s in range(CFG["SEEDS"])]
    res = run_pool(synth_task, jobs, "EXP-B", W)
    DB = pd.DataFrame([r for rr, _ in res for r in rr]); DB.to_csv(f"{OUT}/expB_synthetic.csv", index=False)
    DGB = [x for _, dd in res for x in dd]
    json.dump(DGB, open(f"{OUT}/expB_diagnostics.json", "w"))

    P("\n## VERIFICATION of EXP-B against the published raw CSVs (results_extended.zip)")
    if CFG["VERIFY"]:
        vl, vr = verify_against_published(DB)
        for l in vl: P(l); log(l)
    else: P("skipped (VERIFY=False)")

    FLOOR = 1e-12
    wideB = {}
    for (fn, D), g in DB.groupby(["func", "D"]):
        wideB[f"{disp(fn)}|D{D}"] = g.pivot(index="seed", columns="method", values="v1.0")
    base = "c0.2"
    pairsB = ([(m, base, "F1 learned/baselines") for m in ["rs", "c0.05", "c0.5", "hist", "ml_ch", "ml_free", "ml_reuse"]] +
              [("ml_ch", "ml_free", "F1 learned/baselines"), ("ml_reuse", "ml_ch", "F1 learned/baselines")] +
              [(m, base, "F2 rule") for m in ["rule_ch", "rule_free", "rule_reuse"]] +
              [("rule_ch", "rule_free", "F2 rule"), ("rule_reuse", "rule_ch", "F2 rule")] +
              [("ml_ch", "rule_ch", "F3 learned vs rule"), ("ml_free", "rule_free", "F3 learned vs rule"),
               ("ml_reuse", "rule_reuse", "F3 learned vs rule")] +
              [(f"ml_ch_t{j}", base, "F4 training replicates") for j in range(1, reps + 1)])
    TB = paired_table(wideB, pairsB, FLOOR); TB.to_csv(f"{OUT}/expB_tests.csv", index=False)
    SB = sig_counts(TB, alpha)
    P(f"\n## EXP-B: SYNTHETIC SUITE ({len(wideB)} settings, {CFG['SEEDS']} paired instances; Holm within family; alpha={alpha}; "
      f"effect = median log10 ratio A/B, positive => A worse; sig counts are over settings)")
    P("family | A | B | n_sig/n | A_sig_better | A_sig_worse | median_effect_log10")
    for _, r in SB.iterrows(): P(f"{r.fam} | {r.A} | {r.B} | {r.n_sig}/{r.n} | {r.A_sig_better} | {r.A_sig_worse} | {r.med_effect}")

    P("\n### Price P = med(charged)/med(free), value V = med(free)/med(c0.2), net VP = med(charged)/med(c0.2); ml | rule (medians floored at 1e-12)")
    P("setting | P_ml | V_ml | VP_ml | P_rule | V_rule | VP_rule")
    nV = {"ml": 0, "rule": 0}; nVP = {"ml": 0, "rule": 0}
    for st, Wd in wideB.items():
        med = Wd.median().clip(lower=FLOOR); row = [st]
        for fam in ("ml", "rule"):
            Pp = med[f"{fam}_ch"]/med[f"{fam}_free"]; V = med[f"{fam}_free"]/med[base]; VP = med[f"{fam}_ch"]/med[base]
            nV[fam] += int(V < 1); nVP[fam] += int(VP < 1); row += [g3(Pp), g3(V), g3(VP)]
        P(" | ".join(row))
    P(f"settings with V<1 (free beats baseline): ml {nV['ml']}/{len(wideB)}, rule {nV['rule']}/{len(wideB)}; "
      f"settings with VP<1 (charged beats baseline): ml {nVP['ml']}/{len(wideB)}, rule {nVP['rule']}/{len(wideB)}")

    P("\n### Median final fitness per setting")
    cols = ["rs", "c0.05", "c0.2", "c0.5", "hist", "ml_ch", "ml_free", "ml_reuse", "rule_ch", "rule_free", "rule_reuse"]
    P("setting | " + " | ".join(cols))
    for st, Wd in wideB.items():
        med = Wd.median(); P(f"{st} | " + " | ".join(g3(med[c]) for c in cols))

    P("\n### Budget sweep / break-even: median over settings of the per-setting median log10(method / c0.2); negative = better than baseline")
    methods_sw = ["rs", "hist", "ml_ch", "ml_free", "ml_reuse", "rule_ch", "rule_free", "rule_reuse"]
    P("D | method | " + " | ".join(f"{fr}B" for fr in SW) + " | first multiplier with median<0")
    for Dsel in list(dims) + ["all"]:
        for m in methods_sw:
            vals = []
            for fr in SW:
                per = []
                for (fn, D), g in DB.groupby(["func", "D"]):
                    if Dsel != "all" and D != Dsel: continue
                    Wd = g.pivot(index="seed", columns="method", values=f"v{fr}").clip(lower=FLOOR)
                    per.append(float(np.median(np.log10(Wd[m]/Wd[base]))))
                vals.append(float(np.median(per)))
            first = next((f"{fr}B" for fr, v in zip(SW, vals) if v < 0), "none")
            P(f"{Dsel} | {m} | " + " | ".join(f"{v:+.3f}" for v in vals) + f" | {first}")

    P("\n### Transfer check: tests restricted to the 8 functions NOT used in training (excl. Sphere, Rastrigin); Holm p-values from the full families")
    held = TB[~TB.setting.str.startswith(("Sphere|", "Rastrigin|"))]
    SH = sig_counts(held, alpha)
    P("family | A | B | n_sig/n | A_sig_better | A_sig_worse | median_effect_log10")
    for _, r in SH[SH.B == base].iterrows(): P(f"{r.fam} | {r.A} | {r.B} | {r.n_sig}/{r.n} | {r.A_sig_better} | {r.A_sig_worse} | {r.med_effect}")
    for _, r in SH[SH.fam.str.startswith("F3")].iterrows(): P(f"{r.fam} | {r.A} | {r.B} | {r.n_sig}/{r.n} | {r.A_sig_better} | {r.A_sig_worse} | {r.med_effect}")

    P("\n### Training-stream sensitivity (F4): ml_ch retrained on 4 other training streams, tested against c0.2; and median final-fitness spread")
    ml_cols = ["ml_ch"] + [f"ml_ch_t{j}" for j in range(1, reps + 1)]
    spread = []
    for st, Wd in wideB.items():
        meds = np.array([max(Wd[c].median(), FLOOR) for c in ml_cols]); spread.append(meds.max()/meds.min())
    P(f"max/min of the 5 training-stream medians over settings: median={np.median(spread):.2f}, max={np.max(spread):.2f} "
      f"(setting {list(wideB)[int(np.argmax(spread))]})")
    for j in range(0, reps + 1):
        m = "ml_ch" if j == 0 else f"ml_ch_t{j}"
        if j == 0: g = TB[(TB.A == "ml_ch") & (TB.B == base) & (TB.fam == "F1 learned/baselines")]
        else:      g = TB[(TB.A == m) & (TB.B == base)]
        # note: j=0 uses the F1 Holm; j>=1 uses F4 Holm
        P(f"{m} vs c0.2: sig_worse={int(((g.effect > 0) & (g.p_holm < alpha)).sum())}, "
          f"sig_better={int(((g.effect < 0) & (g.p_holm < alpha)).sum())} of {len(g)} settings")

    P("\n### DIAGNOSTICS (EXP-B, pooled over runs, iterations inside B): realised information R vs nominal capacity kappa; probe-state visitation in search vs training")
    for l in diag_lines(DGB, dims, tinfo, lambda D: 1000*D, "EXP-B"): P(l)
    P("controller training cost: " + "; ".join(f"D={D}: b_learn={tinfo[D]['b_learn_FEs']} FEs = {tinfo[D]['b_learn_FEs']/(1000*D):.2f} x B" for D in dims))

    P("\n### Mean wall-clock seconds per run (EXP-B, run length 2B), by method and D")
    wc = DB.groupby(["D", "method"]).wall_s.mean().unstack(0)
    for m in wc.index: P(f"{m} | " + " | ".join(f"D={d}: {wc.loc[m, d]:.3f}" for d in wc.columns))

    # ---- EXP-A ---------------------------------------------------------------------
    log("EXP-A real ML task ...")
    hm = ["rs", "c0.05", "c0.2", "c0.5", "hist", "ml_ch", "ml_reuse", "rule_ch", "rule_reuse"]
    if CFG["HPO_RUN_FREE"]: hm += ["ml_free", "rule_free"]
    if CFG["HPO_TPE"]: hm.append("tpe")
    if CFG["HPO_GP"]: hm.append("gp")
    HB = CFG["HPO_B"]
    jobsA = [(i, m, HB, tabs_hpo, CFG["HPO_SIGMA"], CFG["HPO_EPOCHS"], CFG["TPE_STARTUP"]) for i in range(CFG["HPO_INST"]) for m in hm]
    resA = run_pool(hpo_task, jobsA, "EXP-A", W)
    DA = pd.DataFrame([r for r, _ in resA]); DA.to_csv(f"{OUT}/expA_hpo.csv", index=False)
    DGA = [dict(g, D=4) for _, g in resA if g]
    json.dump(DGA, open(f"{OUT}/expA_diagnostics.json", "w"))
    assert DA.evals.max() <= HB, "budget exceeded in EXP-A"
    wA = {"MLP-digits": DA.pivot(index="inst", columns="method", values="val")}
    wT = {"MLP-digits(test)": DA.pivot(index="inst", columns="method", values="test")}
    refs = ["c0.2", "rs"] + (["tpe"] if CFG["HPO_TPE"] else []) + (["gp"] if CFG["HPO_GP"] else [])
    pairsA = [(m, b, "A1 vs references") for m in hm for b in refs if m != b]
    pairsA += [("ml_ch", "rule_ch", "A2 learned vs rule"), ("ml_reuse", "rule_reuse", "A2 learned vs rule"),
               ("ml_reuse", "ml_ch", "A3 reuse"), ("rule_reuse", "rule_ch", "A3 reuse")]
    TA = paired_table(wA, pairsA, 1e-3); TT = paired_table(wT, pairsA, 1e-3)
    TA.to_csv(f"{OUT}/expA_tests_val.csv", index=False); TT.to_csv(f"{OUT}/expA_tests_test.csv", index=False)
    P(f"\n## EXP-A: REAL ML TASK (MLP on scikit-learn digits; 4 hyper-parameters: lr, L2, width, batch; B={HB} training runs; "
      f"{CFG['HPO_INST']} instances = random train/val splits; epochs={CFG['HPO_EPOCHS']}; search step sigma={CFG['HPO_SIGMA']} of the unit box)")
    P("method | val_err median [IQR] | test_err median [IQR] | wall_s mean")
    q = lambda s: f"{np.median(s):.4f} [{np.percentile(s, 25):.4f},{np.percentile(s, 75):.4f}]"
    for m in hm:
        g = DA[DA.method == m]; P(f"{m} | {q(g.val)} | {q(g.test)} | {g.wall_s.mean():.1f}")
    chk = sorted(set([b for b in range(10, HB + 1, 10)] + [HB]))
    P("\nAnytime median validation error (budget = number of training runs)")
    P("method | " + " | ".join(f"b={b}" for b in chk))
    for m in hm:
        g = DA[DA.method == m]
        P(f"{m} | " + " | ".join(f"{np.median(g['val' if b == HB else f'a{b}']):.4f}" if ('val' if b == HB else f'a{b}') in g else "na" for b in chk))
    P("\nPaired Wilcoxon on VALIDATION error (Holm within family); effect = median log10 ratio A/B, positive => A worse")
    P("family | A | B | effect | p_holm | A_wins | B_wins")
    for _, r in TA.iterrows(): P(f"{r.fam} | {r.A} | {r.B} | {r.effect:+.3f} | {r.p_holm:.3g} | {r.a_wins} | {r.b_wins}")
    P("\nSame on held-out TEST error (final incumbent retrained once), pairs against references only")
    for _, r in TT[TT.fam.str.startswith("A1")].iterrows(): P(f"{r.A} | {r.B} | {r.effect:+.3f} | {r.p_holm:.3g} | {r.a_wins} | {r.b_wins}")
    P("\nEXP-A diagnostics of the probe controllers (pooled)")
    for l in diag_lines(DGA, [4], {4: tinfo_hpo}, lambda D: HB, "EXP-A"): P(l)
    P(f"Training cost of the learned controller b_learn={tinfo_hpo['b_learn_FEs']} FEs on Sphere/Rastrigin (zero HPO runs). "
      f"Learning it on the HPO objective itself would need {tinfo_hpo['b_learn_FEs']} training runs = {tinfo_hpo['b_learn_FEs']/HB:.0f} x the budget B={HB}.")
    P(f"ML-BJS charged completes {(HB - 1)//(1 + K)} iterations within B={HB} (probe cost {K} runs per step).")

    # ---- EXP-C ---------------------------------------------------------------------
    if CFG["BO_SYN"]:
        log("EXP-C small-budget comparison ...")
        D, B = CFG["BO_DIM"], CFG["BO_B"]
        mC = ["rs", "c0.2", "hist", "ml_ch", "ml_reuse", "rule_ch", "rule_reuse"] + (["tpe"] if CFG["BO_TPE"] else [])
        jobsC = [(D, fn, s, B, tabs[D], mC, (1.0,), CFG["TPE_STARTUP"]) for fn in FNAMES for s in range(CFG["BO_INST"])]
        resC = run_pool(synth_task, jobsC, "EXP-C", W)
        DC = pd.DataFrame([r for rr, _ in resC for r in rr]); DC.to_csv(f"{OUT}/expC_bo_synth.csv", index=False)
        wC = {disp(fn): g.pivot(index="seed", columns="method", values="v1.0") for fn, g in DC.groupby("func")}
        refC = ["c0.2"] + (["tpe"] if "tpe" in mC else [])
        pairsC = [(m, b, "C1 vs references") for m in mC for b in refC if m != b]
        TC = paired_table(wC, pairsC, FLOOR); TC.to_csv(f"{OUT}/expC_tests.csv", index=False)
        SC = sig_counts(TC, alpha)
        P(f"\n## EXP-C: SMALL BUDGET (D={D}, B={B}, {CFG['BO_INST']} instances, {len(wC)} functions, {len(TC)} tests, Holm within family)")
        P("A | B | n_sig/n | A_sig_better | A_sig_worse | median_effect_log10")
        for _, r in SC.iterrows(): P(f"{r.A} | {r.B} | {r.n_sig}/{r.n} | {r.A_sig_better} | {r.A_sig_worse} | {r.med_effect}")
        P("median final fitness per function: function | " + " | ".join(mC))
        for st, Wd in wC.items():
            med = Wd.median(); P(f"{st} | " + " | ".join(g3(med[c]) for c in mC))
        P("mean wall-clock s/run: " + ", ".join(f"{m}={DC[DC.method == m].wall_s.mean():.2f}" for m in mC))

    # ---- save + zip -----------------------------------------------------------------
    P(f"\nTotal runtime: {(time.time()-T0)/60:.1f} min")
    txt = "\n".join(paste)
    open(f"{OUT}/PASTE_BLOCK.txt", "w").write(txt)
    shutil.make_archive("tai_results", "zip", OUT)
    print("\n=====BEGIN PASTE BLOCK=====")
    print(txt)
    print("=====END PASTE BLOCK=====")
    print("\nFiles in", OUT, "and tai_results.zip")
    try:
        from google.colab import files  # type: ignore
        files.download("tai_results.zip")
    except Exception:
        pass

if __name__ == "__main__":
    main()

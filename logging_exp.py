"""Realised-information diagnostics (same 30 instances, same budgets as the main runs)."""
import json, sys
import numpy as np
from collections import deque
import experiments as E
from experiments import FUNCS, shifted, Counted, psi, Const, History, Policy, LO, HI, SIGMA, K, SEEDS, BUDGET, train_phi
import experiments2 as X

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

class MLProbeLog(E.MLProbe):
    def __init__(self, t): super().__init__(t); self.ps, self.zs = [], []
    def p(self, ctx):
        z = psi(lambda Xp: ctx["f"].many(Xp), ctx["x"], ctx["fx"], ctx["rng"])
        j = int(round(E.K*z)); self.zs.append(j); self.ps.append(self.t[j]); return self.t[j]

def search_xy(f, D, B, policy, rng):   # engine copy that also returns start and final points
    x = rng.uniform(LO, HI, D); x0 = x.copy(); fx = f(x); hist = deque(maxlen=20)
    while f.n + policy.cost <= B:
        p = policy.p(dict(f=f, x=x, fx=fx, hist=hist, rng=rng))
        c = rng.uniform(LO, HI, D) if rng.random() < p else np.clip(x + rng.normal(0, SIGMA, D), LO, HI)
        fc = f(c); ok = fc < fx
        if ok: x, fx = c, fc
        hist.append(ok)
    return fx, x0, x

out = {}
for D in (2, 10):
    table, Xtr, ytr = X.train_table(D, [(FUNCS["sphere"], 0.05), (FUNCS["rastrigin"], 0.8)], 12345 + D, 0)
    assert np.allclose(table, train_phi(D))                      # same table as the main runs
    zt = np.round(Xtr*K).astype(int)
    out[f"train_D{D}"] = dict(table=table.tolist(), mean_label=float(ytr.mean()), mean_abs_dev_from_mean_label=float(np.abs(table - ytr.mean()).mean()),
                              H_psi_bits=entropy(zt), MI_psi_label_bits=mutual_info(zt, (ytr > 0.4).astype(int)),
                              frac_psi0=float((zt == 0).mean()))
    for fn in FUNCS:
        B = BUDGET(D); fidx = list(FUNCS).index(fn)
        ml_z, ml_p, hi_p, ml_mean, hi_mean = [], [], [], [], []
        same_hc, same_p2 = [], []
        for seed in range(SEEDS):
            shift = np.random.default_rng([seed, D, fidx]).uniform(-2, 2, D); raw = shifted(fn, shift)
            r = np.random.default_rng([seed, D, fidx, 900])
            pol = MLProbeLog(table); f = Counted(raw); E.search(f, D, B, pol, r)
            ml_z.append(np.array(pol.zs)); ml_p.append(np.array(pol.ps))
            hist_pol = Logged(History()); f = Counted(raw); E.search(f, D, B, hist_pol, np.random.default_rng([seed, D, fidx, 901]))
            hi_p.append(np.array(hist_pol.ps))
            if fn == "rastrigin":
                for lab, p, lst in (("hc", 0.0, same_hc), ("p02", 0.2, same_p2)):
                    f = Counted(raw); fx, x0, xf = search_xy(f, D, B, Const(p), np.random.default_rng([seed, D, fidx, 902]))
                    lst.append(bool(np.all(np.rint(xf - shift) == np.rint(x0 - shift))))
        # psi values visited: recompute from p via the table (table entries are distinct up to rounding)
        allp = np.concatenate(ml_p); zvis = np.concatenate(ml_z)
        hp = np.concatenate(hi_p)
        rec = dict(ml_mean_p=float(np.mean(allp)), ml_std_p=float(np.std(allp)), ml_H_p_bits=entropy(np.round(allp/0.05)*0.05),
                   ml_iters_per_run=float(np.mean([len(a) for a in ml_p])), ml_H_psi_bits=entropy(zvis), ml_frac_psi0=float(np.mean(zvis == 0)),
                   hist_mean_p=float(np.mean(hp)), hist_std_p=float(np.std(hp)), hist_H_p_bits=entropy(np.round(hp/0.05)*0.05),
                   hist_mean_p_first_quarter=float(np.mean([a[:max(1, len(a)//4)].mean() for a in hi_p])),
                   hist_mean_p_last_quarter=float(np.mean([a[-max(1, len(a)//4):].mean() for a in hi_p])))
        if fn == "rastrigin": rec.update(frac_same_basin_hc=float(np.mean(same_hc)), frac_same_basin_p02=float(np.mean(same_p2)))
        out[f"D{D}_{fn}"] = rec
        print(D, fn, {k: round(v, 3) for k, v in rec.items()}, flush=True)
json.dump(out, open("logging_out.json", "w"), indent=1)

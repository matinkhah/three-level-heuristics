import csv, json, math, os
import numpy as np
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
def tost(D, fn, a, b):
    d = np.log10(clip(get(D, fn, a))) - np.log10(clip(get(D, fn, b)))
    p1 = wilcoxon(d + DELTA, alternative="greater").pvalue; p2 = wilcoxon(d - DELTA, alternative="less").pvalue
    return float(max(p1, p2))
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
print(json.dumps(F, default=str)[:6000])

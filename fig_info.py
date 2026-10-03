import math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
fig.tight_layout(pad=0.4); fig.savefig("fig_info.pdf")

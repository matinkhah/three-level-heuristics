# three-level-heuristics

Code, raw results and figure scripts for the paper **"Budget-Aware Evaluation of Information-Driven Adaptive Search"**.

The paper asks how adaptive search controllers should be evaluated when the information they use costs objective evaluations drawn from the same budget as the search itself. It studies this in one controlled engine, Bernoulli jump--step search (BJS), by charging every probe and grid evaluation to a common budget and comparing each charged controller with a free-information counterfactual and a fixed baseline.

This repository reproduces every experiment, statistical test, table and figure in the paper. The conclusions apply to the tested engine, controllers, benchmarks and budgets, as stated in the paper's Limitations.

## What is studied

| Item | Setting |
|---|---|
| Engine | BJS: with probability `p` a uniform global proposal, otherwise a Gaussian step of size `sigma`; elitist acceptance |
| Benchmarks | 10 shifted functions (Ackley, Griewank, HappyCat, Rastrigin, Rosenbrock, Salomon, Schaffer N.2, Schwefel 2.22, Sphere, Weierstrass), domain [-5.12, 5.12]^D |
| Dimensions | D = 2 and D = 10 (main suite); D = 4 (tuning task), D = 5 (small-budget check) |
| Budget | B = 1000 D function evaluations (FEs), shared by search and information acquisition |
| Instances | 30 paired instances per setting (same shifts and random streams across methods) |
| Statistics | Paired Wilcoxon tests with Holm correction; equivalence tests where "no difference" is claimed |

Policies compared (paper Table II): constant `p` (including 0.05, 0.2, 0.5 and a grid), decay-`p`, History-BJS, one-fifth success rule (controls `sigma`), ML-BJS (learned table from the share of 8 random probes that improve), and grid-plus-refinement. Probe controllers are run in **charged**, **free** (diagnostic counterfactual, not deployable) and **reuse** (an improving probe is accepted as a move) variants.

Offline training of the learned controller is **not** charged to the online budget; this convention favors the learned policy and is stated in the paper. Only FEs are counted, not wall-clock time or energy.

## Repository layout

| File | Role |
|---|---|
| `experiments.py` | Round 1 (hypotheses H1--H4): main suite, writes `raw_D2.csv`, `raw_D10.csv` |
| `experiments2.py` | Round 2 (hypotheses H5--H8, exploratory): writes `raw2_D2.csv`, `raw2_D10.csv` |
| `logging_exp.py` | Realized control-output entropy and probe-state diagnostics, writes `logging_out.json` |
| `analysis2.py` | Tables (`t*.tex`), tests, `facts.json`, `fig_sweep2.pdf` |
| `fig_info.py` | Capacity versus acquisition-cost figure, writes `fig_info.pdf` |
| `tai_experiments_v2.py` | Extended checks (paper section "Extended Checks"): single self-contained script |
| `raw*.csv`, `logging_out.json` | Raw results behind the main tables |
| `fig_info.pdf`, `fig_sweep2.pdf` | Figures used in the paper |
| `archive/` | Older drafts and scripts, not needed for reproduction |

## Requirements

Python 3 with `numpy`, `scipy`, `pandas`. The extended script additionally needs `scikit-learn` (digits dataset and MLP) and `optuna` (TPE baseline). The extended run was tested in Google Colab with Python 3.13, numpy 2.1.3, scipy 1.16.3, scikit-learn 1.6.1, pandas 2.2.3 and Optuna 5.0.0 (the versions the script prints at start-up). A different NumPy or SciPy version can change random streams; see "Verification" below.

```
pip install numpy scipy pandas scikit-learn optuna
```

## Reproducing the main experiments (single CPU, about 25 minutes in total)

```
python experiments.py 2 10      # round 1 (H1-H4): raw_D2.csv, raw_D10.csv
python experiments2.py 2        # round 2 (H5-H8): raw2_D2.csv
python experiments2.py 10       # round 2: raw2_D10.csv
python logging_exp.py           # diagnostics: logging_out.json
python analysis2.py             # tables (t*.tex), tests, facts.json, fig_sweep2.pdf
python fig_info.py              # fig_info.pdf
```

The `t*.tex` table bodies produced by `analysis2.py` were pasted into the corresponding tables of the LaTeX source.

## Reproducing the extended checks (about 50 minutes on 2 CPU cores)

```
python tai_experiments_v2.py
```

Settings are in the `CONFIG` block at the top of the script; a `QUICK` flag gives a roughly one-minute smoke test whose numbers should not be interpreted. The script runs four studies and prints one report at the end (it also writes to the folder given by `OUT`, default `tai_results`):

| Study | Content | Paper location |
|---|---|---|
| EXP-B | Synthetic suite at D = 2 and 10, 30 paired instances; charged, free and reuse variants of the learned controller and of an untrained linear rule `p = 0.05 + 0.75(1 - psi)`; four retrained controllers; Holm-corrected tests within question families; transfer check on the 8 functions not used in training; budget sweep 0.1B to 2B; diagnostics | Tables on extended checks, budget-sweep text |
| EXP-A | Neural-network tuning: MLP on scikit-learn digits, 4 hyperparameters (learning rate, L2, width, batch size), B = 60 training runs of 25 epochs, 12 random train/validation splits; compared with random search, fixed `p`, History-BJS, the controllers and Optuna TPE | Tuning-task table |
| EXP-C | Small budget, D = 5, B = 200, 10 instances per function, against TPE (descriptive only: 10 instances cannot reach significance after Holm correction over 140 tests) | Small-budget table |
| Verification | Reruns of the synthetic suite are compared with the published raw results | See below |

### Verification against the published raw results

To check the synthetic rerun, set `REF_ZIP` in `CONFIG` to the path of the archive of published raw CSVs (`results_extended.zip`). The report then prints `N/N runs bit-identical`. In the reported run, all 7200 of 7200 runs were bit-identical, so the extended synthetic results use exactly the streams, engines and tables of the main experiments. If the check does not print `N/N`, do not use the synthetic numbers; the usual cause is a library-version difference.

## Mapping to the paper

| Paper element | Produced by |
|---|---|
| Main results, tests, Level 3 variants (Sections on results) | `experiments.py`, `experiments2.py`, `analysis2.py` |
| Realized control-output entropy, probe-state diagnostics | `logging_exp.py` and the diagnostics block of `tai_experiments_v2.py` |
| Capacity versus cost figure | `fig_info.py` |
| Extended checks: retrained controllers, untrained rule, transfer, budget sweep, tuning task, small budget | `tai_experiments_v2.py` |

The tests in the main analysis form one Holm family of 160 tests; the extended checks are exploratory and use Holm correction within each question family (learned controller against baselines, rule against baselines, learned against rule, training replicates). Effects are median log10 ratios, positive when the first method is worse.

## Notes and limits

- Results concern one single-solution engine, ten shifted synthetic functions, one small tuning task and the budgets tested; they are not claims about learned optimizers in general.
- The best fixed-`sigma` hill climber and best constant `p` are chosen in-sample, which favors those references.
- Offline training used 5400 FEs per dimension (2.7B at D = 2, 0.54B at D = 10; about 90B on the tuning objective) and is not charged.
- The Gaussian-process Bayesian-optimization option in the extended script is off and untested.

## Citation

If you use this code, please cite the paper above. Citation details will be added when the paper is published.

# Information Capacity and Acquisition Cost in Heuristic Search: A Three-Level Cybernetic Taxonomy

[![Python Version](https://shields.io)](https://python.org)
[![License: MIT](https://shields.io)](https://opensource.org)

This repository contains the official, **one-file reproducible experimental pipeline** for the paper *"Information State and Evaluation Cost in Single-Solution Heuristic Search: A Three-Level Taxonomy"*. 

Rather than treating heuristic search blindly, this framework models the allocation of computational budgets between **state exploration** and **online learning** as a cybernetic feedback control problem. We introduce a rigorous three-level taxonomy to classify how much online information an algorithm acquires—and at what exact fitness evaluation (FE) cost.

---

## 🔬 Framework & Taxonomy Overview

The architecture systematically categorizes feedback control policies in heuristic search engines based on their nominal capacity (κ) and acquisition cost:

*   **Level 1 (Static/Constant):** Zero-capacity policies (p=const) serving as baseline information states.
*   **Level 2 (Online Feedback Loops):** Dynamic adjustments using zero-cost historical indicators (rejection rates, success rules) or budget-charged Machine Learning (ML) probe-based feedback policies (Random Forest regression).
*   **Level 3 (Global Landscape Tabulation):** High-cost acquisition models that tabulate macroscopic landscape properties via grid mapping combined with localized refinement.

---

## 🛠️ Repository Structure & Pipeline

The pipeline is completely integrated into highly optimized, deterministic scripts designed for **100% exact numerical replication** (preserving all random streams, instance shifts, and seeds):

*   `run_all_experiments.py`: The baseline pipeline executing the framework over the first 5 classic benchmark landscapes (Sphere, Rastrigin, Ackley, Griewank, Rosenbrock).
*   `extend_new_functions.py`: The extension pipeline that expands validation over 5 additional complex, deceptive, and non-differentiable landscapes (Schwefel 2.22, Weierstrass, HappyCat, Salomon, Schaffer N.2) across dimensions D=2 and D=10.

The scripts automate the entire workflow:
\[\text{Data Generation} \longrightarrow \text{Wilcoxon/TOST Statistical Analysis} \longrightarrow \text{LaTeX Table Compilation (t*.tex)} \longrightarrow \text{PDF Visualization}\]

---

## 🚀 How to Run (Google Colab / Local Python >= 3.9)

No external heavy-duty dependencies are required. Standard scientific computing packages (`numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`) are sufficient.

### 1. Fast Verification (Smoke Test)
To verify script sanity and environment compatibility in under **2 minutes**, open the script, set the quick flag to true, and run:
```python
QUICK = True
```
```bash
python run_all_experiments.py
```

### 2. Full Production Run
To reproduce the complete suite of raw data, LaTeX tables, and figures from scratch (~13–25 minutes depending on CPU cores):
```python
QUICK = False
```
```bash
python run_all_experiments.py
```

### 3. Pipeline Extension & Re-Analysis
To seamlessly append the 5 new complex benchmark functions to your pre-existing data matrices, run the extension cell script. It verifies your existing environment streams against stored benchmarks, runs the new functions safely via staged environments, and automatically recalculates the global statistical metrics:
```bash
python extend_new_functions.py
```

---

## 📊 Outputs & Generated Artifacts

All outputs are structured systematically within the generated `./results` directory and archived into an auto-downloadable `results_extended.zip`:

*   `raw_D*.csv` / `raw2_D*.csv`: Compiled raw fitness and evaluation cost arrays per seed.
*   `facts.json`: Centralized data store containing all computed bootstrap confidence intervals, exact p-values, and descriptive metrics.
*   `t2_main_D*.tex` to `t10_success.tex`: Polished, publication-ready LaTeX table bodies.
*   `fig_sweep2.pdf`: Log-scaled performance profiles showing structural behavior under varying control limits.
*   `fig_info.pdf`: Systemic mapping of nominal information capacity against physical acquisition cost.

---

## 📜 Citation & License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```bibtex
@article{matinkhah2026information,
  title={Information Capacity and Acquisition Cost in Heuristic Search: A Three-Level Cybernetic Taxonomy},
  author={Matinkhah, M. A.},
  journal={IEEE Transactions on Cybernetics (Under Review)},
  year={2026}
}
```

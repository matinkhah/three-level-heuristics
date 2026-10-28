# Budget-Aware Evaluation of Information-Driven Adaptive Search


This repository contains the official, self-contained implementation pipeline for the paper "Budget-Aware Evaluation of Information-Driven Adaptive Search".
Rather than treating metaheuristics blindly, this framework models adaptive parameter control as a closed-loop system where information-gathering measurements compete directly with optimization steps for the exact same resource: objective function evaluations (FEs).

🔬 Framework & Core Methodology

The code evaluates adaptive search under a strict common-budget constraint, introducing metrics to separate the optimization value of a controller's decisions from the opportunity cost of acquiring its information:
* Nominal Information Capacity (κ): The maximum bits of variety an information state can theoretically represent.
* Acquisition Cost (a): The number of FEs consumed purely to build an information state.
* Realized Control-Output Entropy (R): The empirical Shannon entropy of the quantized parameters produced along search trajectories.
* Free-Information Counterfactuals: A diagnostic mechanism that runs adaptive policies without charging their probes to isolate the cost penalty (P) from behavioral value (V).

Three-Level Information Taxonomy

1. Level 1 (Uninformed): Baseline states with zero capacity (κ = 0), acting as static references (e.g., constant jump probability p).
2. Level 2 (Compressed State): Dynamic feedback using zero-cost historical markers (rejection rates, success counters) or budget-charged Machine Learning (ML) probe-based feedback.
3. Level 3 (Exhaustive Tables): High-capacity spatial structures that tabulate macroscopic landscape properties (e.g., coordinate grids) followed by localized refinement.

💻 Repository Structure & Codebase

The implementation is written cleanly without heavy framework overhead to prioritize deterministic replication across random streams, seeds, and coordinate offsets:
* run_all_experiments.py: Implements the baseline pipeline over the initial 5 benchmark problems (Sphere, Rastrigin, Ackley, Griewank, Rosenbrock).
* extend_new_functions.py: Extends validation to 5 additional complex, non-differentiable, and highly multimodal landscapes (Schwefel 2.22, Weierstrass, HappyCat, Salomon, Schaffer N.2).
* tai_experiments_v2.py: The unified, production-grade experiment runner. It includes the complete synthetic suite (\(D \in \{2, 10\}\)), the baseline repository checks, an embedded budget sweep (\(0.1B \dots 2B\)), a small-budget comparison with Optuna TPE (D=5, B=200), and a real-world machine learning workload (hyperparameter optimization of a Multi-Layer Perceptron on scikit-learn digits).

🚀 Execution Guide (Google Colab / Local Machine)


Prerequisites

The script compiles using standard mathematical and data manipulation utilities. Ensure you have the core scientific stack installed:
bash
pip install numpy scipy scikit-learn pandas optuna
Use code with caution.

1. Verification & Smoke Test

To confirm environment compatibility and verify your environment handles the data collection loop safely, toggle the QUICK flag to True at the top of the script and run:
bash
python tai_experiments_v2.py
Use code with caution.
(This completes in under 2 minutes with reduced sample seeds).

2. Full Reproduction Suite

To generate the complete dataset, descriptive statistics, and diagnostics used in the paper:
1. Open tai_experiments_v2.py.
2. Ensure QUICK = False is set.
3. Set your target multiprocessing count (e.g., WORKERS = 2 for a standard free Colab instance).
4. Run the file:
bash
python tai_experiments_v2.py
Use code with caution.
(Takes approximately 15–25 minutes depending on your CPU configuration).

📊 Outputs & Validation Artifacts

All results are compiled locally inside the auto-created ./tai_results/ directory and packed cleanly into tai_results.zip:
* expB_synthetic.csv: Compiled raw records containing global evaluation trajectories across seeds and function landscapes.
* expA_hpo.csv: Raw execution metrics mapping error performance profiles during the real ML workload training tasks.
* expB_tests.csv / expA_tests_val.csv: Multi-family Holm-corrected paired Wilcoxon signed-rank matrices.
* controller_training.json: Information log detailing training labels, mutual information profiles, and base entropy tracking measurements.

Bit-Identical Integrity Check

When running tai_experiments_v2.py, the pipeline automatically down-streams the raw reference zip (results_extended.zip) from GitHub and performs a cell-by-cell row check. A successful run will output:
text
3600/3600 runs bit-identical to the published results
Use code with caution.
This guarantees that your hardware, runtime environment, and dependency distributions match the original published experimental logs with absolute numerical precision.

📜 Citation & License

This project is licensed under the terms of the MIT License.
If you use this evaluation framework, the taxonomy, or the data generation pipelines in your research, please cite the work as follows:
```bibtex
@article{matinkhah2026budget,
  title={Budget-Aware Evaluation of Information-Driven Adaptive Search},
  author={Matinkhah, M. A.},
  journal={IEEE Transactions on Artificial Intelligence (Under Review)},
  year={2026}
}
```

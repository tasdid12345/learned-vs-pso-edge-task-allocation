# LightGBM allocator vs fixed-budget PSO (edge micro-cluster simulation)

Question: can a LightGBM allocator trained offline on long-run PSO solutions match or beat a fixed-budget PSO (20 particles x 30 iterations) when allocating tasks to edge nodes on a shifted workload (heavier tasks, slower nodes)?

Run `python analysis.py`. It writes `results.json` and `figures/relative_objective.png`.

Setup (seed 42, CPU only, seeded numpy simulation, 300 train / 100 shifted test instances):
- Baseline: standard discrete PSO, 20 particles, 30 iterations (unchanged).
- Improved: LightGBM pairwise task-node classifier trained on long-PSO assignments (config and greedy load weight chosen by 3-fold CV on training instances only). At decision time, the probabilities are decoded greedily with a few load weights, the best candidate by objective is kept, and it is refined with steepest-descent single-task moves. The total number of objective evaluations is capped below the baseline PSO's (620), so the improved method does not get more search budget than the baseline.
- Metric: mean objective relative to a long-run PSO reference (lower is better). The test set is used only once, for final evaluation.

## Results (latest run)

On a simulated edge micro-cluster allocation test set (100 shifted instances), my LightGBM allocator scored 0.998 versus 1.067 for a fixed-budget discrete PSO baseline on mean objective relative to a long-run PSO reference, a small gain at lower decision time.

| Metric | Value |
|---|---|
| baseline_score | 1.067 |
| improved_score | 0.9976 |
| baseline_score_std | 0.007483 |
| improved_score_std | 0.0005543 |
| improvement | 0.06987 |
| baseline_miss_rate | 0.4 |
| improved_miss_rate | 0.3478 |
| baseline_miss_rate_std | 0.009908 |
| improved_miss_rate_std | 0.001434 |
| baseline_decision_ms | 3.861 |
| improved_decision_ms | 2.381 |
| decision_speedup_x | 1.622 |
| baseline_mean_objective | 63.12 |
| improved_mean_objective | 59.33 |
| reference_mean_objective | 59.48 |
| reference_miss_rate | 0.353 |
| offline_label_and_train_seconds | 16.36 |
| greedy_lambda | 2 |
| baseline_objective_evals_per_instance | 620 |
| improved_objective_evals_per_instance | 288.3 |

Run on 2026-10-23. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulation only; objective weights and shift are my own choices. Rows are instances (each has 20 tasks x ~5 nodes). The reference is a single long PSO run, not a proven optimum, so ratios below 1 are possible. The improved method's total timing (features, prediction, decoding, local refinement) is included in decision time but not offline training/labelling. The local refinement uses the same objective function as the PSO, with fewer objective evaluations per instance than the baseline (see metrics). Hyper-parameters and greedy load weight were chosen by CV on training instances only; the shifted test set was used once. PSO miss rate is the fraction of tasks missing deadlines.

## Reference

This project is motivated by the following paper:

> Y Alhaizaey, J Singer (2022). *Optimizing heterogeneous task allocation for edge compute micro clusters using PSO metaheuristic*. 2022 Seventh International. https://ieeexplore.ieee.org/abstract/document/10062755/

Paper author: Prof. Jeremy Singer, University of Glasgow (jeremy.singer@glasgow.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

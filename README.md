# LightGBM allocator vs fixed-budget PSO (edge micro-cluster simulation)

Question: can a LightGBM allocator trained offline on long-run PSO solutions match or beat a fixed-budget PSO (20 particles x 30 iterations) when allocating tasks to edge nodes on a shifted workload (heavier tasks, slower nodes)?

Run `python analysis.py`. It writes `results.json` and `figures/relative_objective.png`.

Setup (seed 42, CPU only, seeded numpy simulation, 300 train / 100 shifted test instances):
- Baseline: standard discrete PSO, 20 particles, 30 iterations (unchanged).
- Improved: LightGBM pairwise task-node classifier trained on long-PSO assignments (config and greedy load weight chosen by 3-fold CV on training instances only). At decision time, the probabilities are decoded greedily with a few load weights, the best candidate by objective is kept, and it is refined with steepest-descent single-task moves. The total number of objective evaluations is capped below the baseline PSO's (620), so the improved method does not get more search budget than the baseline.
- Metric: mean objective relative to a long-run PSO reference (lower is better). The test set is used only once, for final evaluation.

## Reference

This project is motivated by the following paper:

> Y Alhaizaey, J Singer (2022). *Optimizing heterogeneous task allocation for edge compute micro clusters using PSO metaheuristic*. 2022 Seventh International. https://ieeexplore.ieee.org/abstract/document/10062755/

Paper author: Prof. Jeremy Singer, University of Glasgow (jeremy.singer@glasgow.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

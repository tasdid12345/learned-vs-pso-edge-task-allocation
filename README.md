# LightGBM allocator vs fixed-budget PSO for edge micro-cluster task allocation

## Motivation
The paper "Optimizing heterogeneous task allocation for edge compute micro clusters using PSO metaheuristic" (2022) uses particle swarm optimization to assign heterogeneous tasks to edge nodes. PSO needs many iterations at decision time. This experiment asks whether a learned allocator, trained offline on long-run PSO solutions, can match it at a much lower decision cost.

## Question
Can a LightGBM allocator trained offline on long-run PSO solutions match or beat a fixed-budget PSO at allocating heterogeneous tasks to edge micro-cluster nodes, at much lower decision cost, on shifted workloads?

## Dataset
A seeded numpy simulation, so nothing is downloaded. Each instance has about 5 heterogeneous nodes (speed, memory, power) and 20 tasks (compute, memory, data size, deadline). The objective is a weighted sum of makespan, energy and deadline-miss and memory-overflow penalties. Training instances come from a base distribution. The 100 test instances are shifted to heavier tasks and slower nodes. This is **not** the paper's dataset.

## Baseline vs improved
- **Baseline:** a standard discrete PSO (positions rounded to node indices, 20 particles, 30 iterations). The paper's exact settings are unknown. It is not claimed to be the paper's exact method.
- **Improved:** a LightGBM classifier on task-node pair features. It imitates assignments from a long-budget PSO (50 particles, 200 iterations). A load-aware greedy pass decodes its output. Hyper-parameters and the greedy load weight are chosen by cross-validation on the training instances only.
- Both use the same evaluation function and the same test instances. Randomized parts run with seeds 42, 43 and 44.
- Main metric: mean objective relative to the long-run PSO reference (lower is better). Secondary metrics: deadline-miss rate and decision time per instance.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> Y Alhaizaey, J Singer (2022). *Optimizing heterogeneous task allocation for edge compute micro clusters using PSO metaheuristic*. 2022 Seventh International. https://ieeexplore.ieee.org/abstract/document/10062755/

Paper author: Prof. Jeremy Singer, University of Glasgow (jeremy.singer@glasgow.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

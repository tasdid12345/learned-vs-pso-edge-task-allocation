"""LightGBM allocator imitating long-run PSO vs fixed-budget PSO on shifted edge workloads (seeded simulation)."""
import json, os, sys, time
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [42, 43, 44]
T = 20                      # tasks per instance
N_TRAIN, N_TEST = 300, 100  # instances (test is 25% of all)
BASE_P, BASE_IT = 20, 30
BASE_EVALS = BASE_P * (BASE_IT + 1)   # objective evaluations used by baseline PSO
rng0 = np.random.default_rng(42)


def make_instance(rng, shift):
    """Random edge micro-cluster (4-6 nodes) and 20 tasks; shift = heavier tasks, slower nodes."""
    N = int(rng.integers(4, 7))
    speed = rng.uniform(2, 10, N); mem = rng.uniform(4, 16, N)
    power = 3 + 1.2 * speed * rng.uniform(0.8, 1.5, N)
    c = rng.uniform(2, 30, T); m = rng.uniform(0.2, 3, T); data = rng.uniform(1, 50, T)
    d = c / 6 + data / 50 + rng.uniform(4, 24, T)   # absolute deadlines (not shifted)
    if shift:
        speed = speed * 0.7; c = c * 1.5; m = m * 1.3; data = data * 1.5
    t = c[:, None] / speed[None, :] + data[:, None] / 50.0  # exec + transfer time
    return dict(speed=speed, mem=mem, power=power, c=c, m=m, d=d, t=t, N=N)


def evaluate(inst, A):
    """Vectorised objective for assignments A (P,T): makespan + 0.01*energy + misses + 5*mem overflow."""
    N = inst["N"]; ar = np.arange(N)
    tt = inst["t"][np.arange(T)[None, :], A]
    oh = A[:, :, None] == ar
    cum = np.cumsum(oh * tt[:, :, None], axis=1)            # tasks run in index order per node
    comp = np.take_along_axis(cum, A[:, :, None], 2)[..., 0]  # completion time of each task
    busy = cum[:, -1, :]
    misses = (comp > inst["d"]).sum(1)
    over = np.maximum((oh * inst["m"][None, :, None]).sum(1) - inst["mem"], 0).sum(1)
    obj = busy.max(1) + 0.01 * (busy * inst["power"]).sum(1) + 1.0 * misses + 5.0 * over
    return obj, misses


def pso(inst, P, iters, seed, w=0.7, c1=1.5, c2=1.5):
    """Standard discrete PSO: continuous positions rounded to node indices."""
    N = inst["N"]; rng = np.random.default_rng(seed)
    dec = lambda x: np.clip(np.rint(x), 0, N - 1).astype(int)
    pos = rng.uniform(-0.5, N - 0.5, (P, T)); vel = rng.uniform(-1, 1, (P, T))
    pb = pos.copy(); pbv, _ = evaluate(inst, dec(pos))
    g = pb[pbv.argmin()].copy(); gv = pbv.min()
    for _ in range(iters):
        vel = w * vel + c1 * rng.random((P, T)) * (pb - pos) + c2 * rng.random((P, T)) * (g - pos)
        pos = np.clip(pos + vel, -0.5, N - 0.5)
        v, _ = evaluate(inst, dec(pos))
        imp = v < pbv; pb[imp] = pos[imp]; pbv[imp] = v[imp]
        if pbv.min() < gv: gv = pbv.min(); g = pb[pbv.argmin()].copy()
    return dec(g)


def features(inst):
    """Scale-free task-node pair features (rows: task-major, node-minor)."""
    t, N = inst["t"], inst["N"]
    e = t * inst["power"][None, :]
    f = [t / t.min(1, keepdims=True), t / inst["d"][:, None], e / e.min(1, keepdims=True),
         np.tile(inst["speed"] / inst["speed"].mean(), (T, 1)),
         np.tile(inst["power"] / inst["power"].mean(), (T, 1)),
         np.tile(inst["mem"] / inst["mem"].mean(), (T, 1)),
         inst["m"][:, None] / inst["mem"][None, :],
         np.tile(inst["c"] / inst["c"].mean(), (N, 1)).T,
         np.tile(inst["d"] / inst["d"].mean(), (N, 1)).T,
         np.tile(np.argsort(np.argsort(-inst["speed"])) / N, (T, 1))]
    return np.stack([x.ravel() for x in f], 1)


def decode(inst, P, lam):
    """Load-aware greedy decode of per-pair probabilities (tasks in index order)."""
    t, d, m, mem = inst["t"], inst["d"], inst["m"], inst["mem"]
    P = P / P.sum(1, keepdims=True); scale = t.min(1).sum() / inst["N"]
    load = np.zeros(inst["N"]); mu = np.zeros(inst["N"]); A = np.empty(T, int)
    for i in range(T):
        fin = load + t[i]
        s = np.log(P[i] + 1e-6) - lam * fin / scale - 1.5 * (fin > d[i]) - 3.0 * (mu + m[i] > mem)
        j = int(s.argmax()); A[i] = j; load[j] += t[i, j]; mu[j] += m[i]
    return A


def allocate(inst, P, lams, max_evals):
    """Model-based allocator: decode model probabilities with a few load weights, keep the best by
    the objective, then steepest-descent single-task moves. Total objective evaluations are capped
    BELOW the number the baseline PSO uses (max_evals)."""
    N = inst["N"]
    cands = np.array([decode(inst, P, l) for l in lams])
    v, _ = evaluate(inst, cands); used = len(cands)
    A = cands[v.argmin()].copy(); best = v.min()
    ti = np.repeat(np.arange(T), N)
    nj = np.tile(np.arange(N), T)
    while True:
        keep = nj != A[ti]
        ii, jj = ti[keep], nj[keep]
        if used + len(ii) > max_evals:
            break
        nb = np.repeat(A[None], len(ii), 0); nb[np.arange(len(ii)), ii] = jj
        nv, _ = evaluate(inst, nb); used += len(ii)
        k = nv.argmin()
        if nv[k] < best - 1e-9:
            best = nv[k]; A = nb[k].copy()
        else:
            break
    return A, used


def lgb_model(cfg, seed):
    return lgb.LGBMClassifier(learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=seed, n_jobs=1, verbose=-1, **cfg)


# ---------------- data ----------------
tr_inst = [make_instance(rng0, False) for _ in range(N_TRAIN)]
te_inst = [make_instance(rng0, True) for _ in range(N_TEST)]
t0 = time.perf_counter()
print("Labelling training instances with long PSO (50 particles x 200 iters)...")
tr_A = [pso(x, 50, 200, seed=int(i) + 42) for i, x in enumerate(tr_inst)]
Xs = [features(x) for x in tr_inst]
ys = [(A[:, None] == np.arange(x["N"])[None, :]).ravel().astype(int) for A, x in zip(tr_A, tr_inst)]
stack = lambda idx: (np.vstack([Xs[i] for i in idx]), np.concatenate([ys[i] for i in idx]))

# ---------------- CV on training set only ----------------
print("Cross-validating LightGBM config and greedy load weight on training set...")
grid = [dict(num_leaves=7, n_estimators=150), dict(num_leaves=15, n_estimators=300)]
best = None
for cfg in grid:
    oof, lls = [None] * N_TRAIN, []
    for a, b in KFold(3, shuffle=True, random_state=42).split(np.arange(N_TRAIN)):
        mdl = lgb_model(cfg, 42).fit(*stack(a))
        Xv, yv = stack(b); lls.append(log_loss(yv, mdl.predict_proba(Xv)[:, 1]))
        for i in b: oof[i] = mdl.predict_proba(Xs[i])[:, 1].reshape(T, -1)
    print(f"  cfg {cfg}: CV logloss {np.mean(lls):.4f}")
    if best is None or np.mean(lls) < best[0]: best = (np.mean(lls), cfg, oof)
_, cfg, oof = best
lam_scores = {lam: np.mean([evaluate(x, decode(x, oof[i], lam)[None])[0][0] for i, x in enumerate(tr_inst)])
              for lam in [0, 0.5, 1, 2, 4, 8]}
lam = min(lam_scores, key=lam_scores.get)
lams = sorted({float(lam), 1.0, 4.0})
print(f"  chosen cfg {cfg}, lambda {lam}, decode lambdas {lams}")
models = [lgb_model(cfg, s).fit(*stack(range(N_TRAIN))) for s in SEEDS]
offline_s = time.perf_counter() - t0
print(f"Offline labelling+training took {offline_s:.1f}s")

# ---------------- test (used once) ----------------
print("Evaluating on shifted test set...")
ref = np.array([evaluate(x, pso(x, 50, 200, 42 + 1000 + i)[None])[0][0] for i, x in enumerate(te_inst)])
ref_miss = np.mean([evaluate(x, pso(x, 50, 200, 42 + 1000 + i)[None])[1][0] for i, x in enumerate(te_inst)]) / T
offs = np.concatenate([[0], np.cumsum([T * x["N"] for x in te_inst])])
res = {"base": [], "imp": []}
evals_used = []
for s, mdl in zip(SEEDS, models):
    # baseline: fixed-budget PSO (20 particles, 30 iterations)
    t1 = time.perf_counter()
    As = [pso(x, BASE_P, BASE_IT, s * 1000 + i) for i, x in enumerate(te_inst)]
    tb = (time.perf_counter() - t1) / N_TEST
    ev = [evaluate(x, A[None]) for x, A in zip(te_inst, As)]
    res["base"].append((np.mean(np.array([e[0][0] for e in ev]) / ref), np.mean([e[1][0] for e in ev]) / T, tb * 1000,
                        np.mean([e[0][0] for e in ev])))
    # improved: features + LightGBM + multi-lambda greedy decode + capped local refinement (timed end to end)
    t1 = time.perf_counter()
    Xt = np.vstack([features(x) for x in te_inst]); pr = mdl.predict_proba(Xt)[:, 1]
    outs = [allocate(x, pr[offs[i]:offs[i + 1]].reshape(T, -1), lams, BASE_EVALS - 1)
            for i, x in enumerate(te_inst)]
    As = [o[0] for o in outs]; evals_used.append(np.mean([o[1] for o in outs]))
    ti = (time.perf_counter() - t1) / N_TEST
    ev = [evaluate(x, A[None]) for x, A in zip(te_inst, As)]
    res["imp"].append((np.mean(np.array([e[0][0] for e in ev]) / ref), np.mean([e[1][0] for e in ev]) / T, ti * 1000,
                       np.mean([e[0][0] for e in ev])))
    print(f"  seed {s}: baseline {res['base'][-1][0]:.4f}  improved {res['imp'][-1][0]:.4f}")

B, I = np.array(res["base"]), np.array(res["imp"])
metrics = {"baseline_score": B[:, 0].mean(), "improved_score": I[:, 0].mean(),
           "baseline_score_std": B[:, 0].std(), "improved_score_std": I[:, 0].std(),
           "improvement": B[:, 0].mean() - I[:, 0].mean(),
           "baseline_miss_rate": B[:, 1].mean(), "improved_miss_rate": I[:, 1].mean(),
           "baseline_miss_rate_std": B[:, 1].std(), "improved_miss_rate_std": I[:, 1].std(),
           "baseline_decision_ms": B[:, 2].mean(), "improved_decision_ms": I[:, 2].mean(),
           "decision_speedup_x": B[:, 2].mean() / I[:, 2].mean(),
           "baseline_mean_objective": B[:, 3].mean(), "improved_mean_objective": I[:, 3].mean(),
           "reference_mean_objective": ref.mean(), "reference_miss_rate": ref_miss,
           "offline_label_and_train_seconds": offline_s, "greedy_lambda": float(lam),
           "baseline_objective_evals_per_instance": float(BASE_EVALS),
           "improved_objective_evals_per_instance": float(np.mean(evals_used))}
metrics = {k: float(v) for k, v in metrics.items()}
if not all(np.isfinite(v) for v in metrics.values()):
    sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
plt.bar(["PSO 20x30", "LightGBM allocator"], [metrics["baseline_score"], metrics["improved_score"]],
        yerr=[metrics["baseline_score_std"], metrics["improved_score_std"]], color=["gray", "tab:blue"])
plt.axhline(1.0, color="k", ls="--", lw=1); plt.ylabel("Mean objective / long-PSO reference (lower better)")
plt.tight_layout(); plt.savefig("figures/relative_objective.png", dpi=120)

out = {"question": "Can a LightGBM allocator trained offline on long-run PSO solutions match or beat a fixed-budget PSO at allocating heterogeneous tasks to edge micro-cluster nodes, at much lower decision cost, on shifted workloads?",
       "method": "Seeded numpy simulation; baseline discrete PSO (20 particles, 30 iters) vs LightGBM pairwise classifier imitating long PSO (50 particles, 200 iters) with load-aware greedy decoding (several load weights, best by objective) followed by a capped single-task-move local refinement; evaluated on a shifted (heavier tasks, slower nodes) test set.",
       "dataset": {"name": "Seeded numpy edge micro-cluster allocation simulation (instances)", "source_url": "N/A (simulated)",
                   "n_rows": N_TRAIN + N_TEST, "n_train": N_TRAIN, "n_test": N_TEST, "same_as_paper": False,
                   "baseline_is_papers_method": False},
       "metric_name": "Mean objective relative to long-run PSO reference (lower is better)", "higher_is_better": False,
       "baseline": "Standard discrete PSO, 20 particles, 30 iterations, w=0.7, c1=c2=1.5 (paper's settings unknown); seeds 42-44.",
       "improved": "LightGBM classifier on scale-free task-node features trained on long-run PSO assignments, decoded by load-aware greedy passes and refined by a local search whose total objective evaluations are capped below the baseline PSO's; seeds 42-44.",
       "metrics": metrics,
       "notes": "Simulation only; objective weights and shift are my own choices. Rows are instances (each has 20 tasks x ~5 nodes). The reference is a single long PSO run, not a proven optimum, so ratios below 1 are possible. The improved method's total timing (features, prediction, decoding, local refinement) is included in decision time but not offline training/labelling. The local refinement uses the same objective function as the PSO, with fewer objective evaluations per instance than the baseline (see metrics). Hyper-parameters and greedy load weight were chosen by CV on training instances only; the shifted test set was used once. PSO miss rate is the fraction of tasks missing deadlines."}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(metrics, indent=2))

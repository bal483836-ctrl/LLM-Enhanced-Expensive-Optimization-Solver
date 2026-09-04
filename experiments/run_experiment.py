"""
Experiment harness.

Runs every method on every test problem across several seeds under the SAME
300-FE budget, then saves:
  * results/summary.csv        -- final best value (mean +/- std) per method
  * results/llm_cost.csv       -- LLM scoring cost per method (Idea 1 evidence)
  * results/curves_<fn>.csv    -- mean best-so-far curve per method
  * results/convergence.png    -- convergence plots (one panel per function)

Methods
-------
  random            : uniform random search (lower bar)
  de                : standard DE/rand/1/bin (traditional EA baseline)
  llm_ea_full       : BOTH ideas (two-stage funnel + budget-aware scheduler)
  llm_ea_single     : ablation of Idea 1 (no coarse screen; large model only)
  llm_ea_no_sched   : ablation of Idea 2 (fixed finalists, no jump-out)

Usage:  python -m experiments.run_experiment [--seeds 5] [--dim 10] [--fes 300]
Set env LLM_API_KEY (+ optional LLM_BASE_URL) and pass --real to use a real
LLM instead of the offline mock.
"""

from __future__ import annotations

import argparse
import os
import csv
import numpy as np

from src.problems import make_problem, ALL_PROBLEMS
from src.evaluator import Evaluator
from src.llm_client import MockLLMClient, ApiLLMClient
from src.solver import LLMEASolver
from src.baselines import random_search, differential_evolution


METHODS = ["random", "de", "llm_ea_full", "llm_ea_single", "llm_ea_no_sched"]


def make_llm(real, seed):
    if real:
        return ApiLLMClient(
            small_model=os.environ.get("LLM_SMALL_MODEL", "gpt-3.5-turbo"),
            large_model=os.environ.get("LLM_LARGE_MODEL", "gpt-4o-mini"),
        )
    return MockLLMClient(seed=seed)


def run_one(method, fn, dim, fes, seed, real):
    prob = make_problem(fn, dim=dim, seed=seed)
    ev = Evaluator(prob, max_fes=fes)
    cost = {}
    if method == "random":
        random_search(ev, seed=seed)
    elif method == "de":
        differential_evolution(ev, seed=seed)
    else:
        llm = make_llm(real, seed)
        # in --real mode print each generation so progress is visible live
        kwargs = dict(seed=seed, verbose=real)
        if method == "llm_ea_single":
            kwargs["single_tier"] = True
        elif method == "llm_ea_no_sched":
            kwargs["use_scheduler"] = False
        LLMEASolver(llm, **kwargs).solve(ev)
        cost = llm.cost_summary()
    # pad/truncate curve to exactly `fes` for aligned averaging
    curve = np.asarray(ev.best_curve, dtype=float)
    if len(curve) < fes:
        curve = np.concatenate([curve, np.full(fes - len(curve), curve[-1])])
    return ev.best_y, curve[:fes], cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--fes", type=int, default=300)
    ap.add_argument("--real", action="store_true", help="use a real LLM API")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--problems", default="",
                    help="comma-separated subset, e.g. 'sphere'. Default: all.")
    ap.add_argument("--methods", default="",
                    help="comma-separated subset, e.g. 'llm_ea_full'. Default: all.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seeds = list(range(args.seeds))

    # allow running a small subset (essential for slow/costly --real demos)
    problems = [p.strip() for p in args.problems.split(",") if p.strip()] or ALL_PROBLEMS
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or METHODS

    # results[fn][method] = {"finals": [...], "curves": [...], "cost": {...}}
    results = {fn: {m: {"finals": [], "curves": [], "cost": {}} for m in methods}
               for fn in problems}

    total = len(problems) * len(methods) * len(seeds)
    done = 0
    for fn in problems:
        for m in methods:
            for s in seeds:
                done += 1
                # progress line so --real runs don't look frozen during slow API calls
                print(f"  [{done}/{total}] running {fn} / {m} / seed {s} ...", flush=True)
                best, curve, cost = run_one(m, fn, args.dim, args.fes, s, args.real)
                results[fn][m]["finals"].append(best)
                results[fn][m]["curves"].append(curve)
                if cost:
                    for k, v in cost.items():
                        results[fn][m]["cost"][k] = results[fn][m]["cost"].get(k, 0) + v
            fm = np.mean(results[fn][m]["finals"])
            fs = np.std(results[fn][m]["finals"])
            print(f"{fn:11s} {m:16s} final = {fm:.4e} +/- {fs:.2e}")

    _write_summary(results, args, seeds)
    _write_costs(results, args, seeds)
    _write_curves(results, args)
    _plot(results, args)
    print(f"\nSaved results to ./{args.outdir}/")


def _write_summary(results, args, seeds):
    path = os.path.join(args.outdir, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "method", "final_mean", "final_std", "n_seeds"])
        for fn in results:
            for m in results[fn]:
                fin = results[fn][m]["finals"]
                w.writerow([fn, m, f"{np.mean(fin):.6e}", f"{np.std(fin):.6e}", len(seeds)])


def _write_costs(results, args, seeds):
    path = os.path.join(args.outdir, "llm_cost.csv")
    n = len(seeds)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "method", "avg_small_scored", "avg_large_scored",
                    "avg_large_calls"])
        for fn in results:
            for m in results[fn]:
                c = results[fn][m]["cost"]
                if not c:
                    continue
                w.writerow([fn, m,
                            f"{c.get('llm_scored_small', 0)/n:.1f}",
                            f"{c.get('llm_scored_large', 0)/n:.1f}",
                            f"{c.get('llm_calls_large', 0)/n:.1f}"])


def _write_curves(results, args):
    for fn in results:
        path = os.path.join(args.outdir, f"curves_{fn}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fes"] + list(next(iter(results.values())).keys()))
            mean_curves = {m: np.mean(results[fn][m]["curves"], axis=0) for m in results[fn]}
            T = len(next(iter(mean_curves.values())))
            for t in range(T):
                w.writerow([t + 1] + [f"{mean_curves[m][t]:.6e}" for m in results[fn]])


def _plot(results, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping plot: {e})")
        return
    problems = list(results.keys())
    fig, axes = plt.subplots(1, len(problems), figsize=(5 * len(problems), 4))
    if len(problems) == 1:
        axes = [axes]
    colors = {"random": "#888888", "de": "#1f77b4", "llm_ea_full": "#d62728",
              "llm_ea_single": "#2ca02c", "llm_ea_no_sched": "#ff7f0e"}
    for ax, fn in zip(axes, problems):
        for m in results[fn]:
            mean = np.mean(results[fn][m]["curves"], axis=0)
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, label=m, color=colors.get(m), lw=1.8)
        ax.set_yscale("log")
        ax.set_title(fn)
        ax.set_xlabel("real evaluations (FEs)")
        ax.set_ylabel("best-so-far f (log)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Convergence under {args.fes}-FE budget (dim={args.dim}, {args.seeds} seeds)")
    fig.tight_layout()
    out = os.path.join(args.outdir, "convergence.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

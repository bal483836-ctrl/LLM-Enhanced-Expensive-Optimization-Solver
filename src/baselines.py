"""
Baselines to judge whether the LLM actually adds value, all under the same
budget-controlled Evaluator.

  * random_search : pure uniform sampling (the lower bar to beat).
  * differential_evolution : a standard self-implemented DE/rand/1/bin,
                             one real evaluation per trial (no surrogate).
"""

from __future__ import annotations

import numpy as np

from .ea import de_rand_1, binomial_crossover, latin_hypercube


def random_search(evaluator, seed=0):
    rng = np.random.default_rng(seed)
    prob = evaluator.problem
    while evaluator.can_eval():
        evaluator.evaluate(rng.uniform(prob.lb, prob.ub))
    return evaluator.best_x, evaluator.best_y


def differential_evolution(evaluator, n_pop=12, F=0.6, CR=0.9, seed=0):
    rng = np.random.default_rng(seed)
    prob = evaluator.problem
    lb, ub = prob.lb, prob.ub

    # init
    pop = latin_hypercube(n_pop, lb, ub, rng)
    fit = np.array([evaluator.evaluate(x) for x in pop if evaluator.can_eval()])
    pop = pop[: len(fit)]

    while evaluator.can_eval():
        mutants = de_rand_1(pop, F, rng)
        for i in range(len(pop)):
            if not evaluator.can_eval():
                break
            trial = np.clip(binomial_crossover(pop[i], mutants[i], CR, rng), lb, ub)
            ft = evaluator.evaluate(trial)
            if ft < fit[i]:  # greedy selection
                pop[i] = trial
                fit[i] = ft
    return evaluator.best_x, evaluator.best_y

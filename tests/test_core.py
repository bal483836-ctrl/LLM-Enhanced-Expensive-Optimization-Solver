"""
Unit tests for the core self-implemented logic.

Run with:  python -m pytest tests/ -q     (or)     python tests/test_core.py
These check the parts that MUST be correct: the budget cap, the two-stage
funnel shrinking the pool, and the scheduler's decisions.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems import make_problem
from src.evaluator import Evaluator, BudgetExceeded
from src.llm_client import MockLLMClient
from src.surrogate import TwoStageSurrogate
from src.scheduler import BudgetAwareScheduler
from src.solver import LLMEASolver


def test_budget_is_hard_capped():
    prob = make_problem("sphere", dim=5, seed=1)
    ev = Evaluator(prob, max_fes=50)
    for _ in range(50):
        ev.evaluate(prob.random_point(np.random.default_rng(0)))
    assert ev.n_fes == 50
    try:
        ev.evaluate(prob.x_opt)
        assert False, "should have raised BudgetExceeded"
    except BudgetExceeded:
        pass
    print("ok: budget hard cap")


def test_solver_never_exceeds_budget():
    for fn in ["sphere", "rosenbrock", "rastrigin"]:
        prob = make_problem(fn, dim=8, seed=2)
        ev = Evaluator(prob, max_fes=120)
        LLMEASolver(MockLLMClient(seed=0), seed=0).solve(ev)
        assert ev.n_fes <= 120, f"{fn}: used {ev.n_fes} > 120"
        assert len(ev.best_curve) == ev.n_fes
    print("ok: solver respects budget")


def test_funnel_shrinks_pool():
    prob = make_problem("sphere", dim=6, seed=3)
    rng = np.random.default_rng(0)
    ctx_X = np.array([prob.random_point(rng) for _ in range(20)])
    ctx_y = np.array([prob(x) for x in ctx_X])
    pool = np.array([prob.random_point(rng) for _ in range(50)])
    surr = TwoStageSurrogate(MockLLMClient(seed=0), n_shortlist=10, n_final=3)
    idx, info = surr.select(pool, ctx_X, ctx_y)
    assert len(idx) == 3
    assert info["pool_size"] == 50 and info["shortlist_size"] == 10
    # single-tier ablation skips the coarse screen (no small-model calls)
    llm = MockLLMClient(seed=0)
    surr2 = TwoStageSurrogate(llm, n_shortlist=10, n_final=3, single_tier=True)
    surr2.select(pool, ctx_X, ctx_y)
    assert llm.n_calls_small == 0 and llm.n_calls_large == 1
    print("ok: two-stage funnel shrinks pool and single-tier skips small model")


def test_scheduler_budget_regimes_and_jumpout():
    sch = BudgetAwareScheduler()
    pop = np.random.default_rng(0).uniform(-5, 5, size=(12, 5))
    lb, ub = np.full(5, -5.0), np.full(5, 5.0)
    # budget-rich -> more finalists; budget-poor -> fewer
    d_rich = sch.update(10.0, pop, lb, ub, budget_frac=0.1)
    d_poor = sch.update(10.0, pop, lb, ub, budget_frac=0.9)
    assert d_rich.n_final >= d_poor.n_final
    # collapsed population (all identical) -> low diversity -> jump-out
    collapsed = np.ones((12, 5))
    d = sch.update(10.0, collapsed, lb, ub, budget_frac=0.5)
    assert d.inject_jumpout
    print("ok: scheduler budget regimes and jump-out trigger")


if __name__ == "__main__":
    test_budget_is_hard_capped()
    test_solver_never_exceeds_budget()
    test_funnel_shrinks_pool()
    test_scheduler_budget_regimes_and_jumpout()
    print("\nALL TESTS PASSED")

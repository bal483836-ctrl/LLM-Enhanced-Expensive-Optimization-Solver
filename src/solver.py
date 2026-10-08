"""
The LLM+EA expensive-optimization solver.

Puts the two ideas together in one loop:

  Idea 1 (two-stage hierarchical surrogate): each generation the EA makes a
          large candidate pool; a cheap small-model coarse screen + an
          accurate large-model fine judge pick only a few finalists to
          really evaluate.
  Idea 2 (budget-aware scheduler): a scheduler watches diversity /
          stagnation / remaining budget and decides how many finalists to
          evaluate and whether to inject LLM "jump-out" solutions.

Everything routes real evaluations through the Evaluator so the 300-FE cap
is strictly enforced.
"""

from __future__ import annotations

import numpy as np

from .ea import latin_hypercube, generate_pool, select_diverse
from .surrogate import TwoStageSurrogate
from .scheduler import BudgetAwareScheduler, Decision


class LLMEASolver:
    def __init__(
        self,
        llm,
        n_init=20,
        n_pop=12,
        pool_size=50,
        n_shortlist=10,
        ctx_size=15,
        n_jumpout=6,
        scheduler=None,
        seed=0,
        verbose=False,
        use_scheduler=True,
        single_tier=False,
        fixed_n_final=3,
        use_funnel=True,
    ):
        self.llm = llm
        self.n_init = n_init
        self.n_pop = n_pop
        self.pool_size = pool_size
        self.ctx_size = ctx_size
        self.n_jumpout = n_jumpout
        # use_scheduler=False is an ablation of Idea 2: fixed finalists, no
        # jump-out injection. single_tier=True is an ablation of Idea 1.
        self.use_scheduler = use_scheduler
        self.single_tier = single_tier
        self.fixed_n_final = fixed_n_final
        # use_funnel=False is the "AI only as scheduler" variant: the two-stage
        # LLM funnel is switched off and finalists are drawn from the pool
        # without any model scoring, so the LLM is used only for jump-out.
        self.use_funnel = use_funnel
        self.surrogate = TwoStageSurrogate(
            llm, n_shortlist=n_shortlist, n_final=fixed_n_final,
            single_tier=single_tier,
        )
        self.scheduler = scheduler or BudgetAwareScheduler()
        self.seed = seed
        self.verbose = verbose
        self.log = []  # per-generation records for analysis

    def _context(self, evaluator):
        """Best `ctx_size` evaluated points, used as LLM in-context examples."""
        X, y = evaluator.history_arrays()
        order = np.argsort(y)[: self.ctx_size]
        return X[order], y[order]

    def _population(self, evaluator):
        """Current population: good-and-spread `n_pop` evaluated points.

        Uses diversity-preserving selection so the population does not
        collapse to near-duplicates, keeping the scheduler's diversity
        signal informative.
        """
        X, y = evaluator.history_arrays()
        prob = evaluator.problem
        idx = select_diverse(X, y, self.n_pop, prob.lb, prob.ub)
        return X[idx], y[idx]

    def solve(self, evaluator):
        rng = np.random.default_rng(self.seed)
        prob = evaluator.problem
        lb, ub = prob.lb, prob.ub

        # ---- initial design ----
        init = latin_hypercube(self.n_init, lb, ub, rng)
        evaluator.evaluate_batch(init)

        gen = 0
        while evaluator.can_eval():
            gen += 1
            ctx_X, ctx_y = self._context(evaluator)
            pop_X, pop_y = self._population(evaluator)
            budget_frac = evaluator.n_fes / evaluator.max_fes

            # ---- Idea 2: scheduler decides the plan for this generation ----
            if self.use_scheduler:
                decision = self.scheduler.update(
                    best_now=evaluator.best_y, pop=pop_X, lb=lb, ub=ub,
                    budget_frac=budget_frac,
                )
            else:
                # ablation: fixed finalists, no jump-out
                decision = Decision(
                    n_final=self.fixed_n_final, inject_jumpout=False,
                    reason="no-scheduler(fixed)",
                )

            # ---- EA builds a large candidate pool ----
            pool = generate_pool(pop_X, self.pool_size, lb, ub, rng)

            # ---- optional LLM "jump-out" injection ----
            if decision.inject_jumpout:
                jump = self.llm.propose(ctx_X, ctx_y, lb, ub, self.n_jumpout, rng)
                pool = np.vstack([pool, np.clip(jump, lb, ub)])

            # ---- Idea 1: two-stage funnel picks the finalists ----
            n_final = min(decision.n_final, evaluator.remaining)
            if self.use_funnel:
                finalists_idx, info = self.surrogate.select(
                    pool, ctx_X, ctx_y, n_final=n_final
                )
            else:
                # AI-only-as-scheduler: no model scoring; draw finalists at random
                k = max(1, min(n_final, len(pool)))
                finalists_idx = rng.choice(len(pool), k, replace=False)
                info = {"pool_size": len(pool), "shortlist_size": len(pool), "n_final": k}

            # ---- spend real evaluations on the finalists ----
            before = evaluator.best_y
            evaluator.evaluate_batch(pool[finalists_idx])
            gain = before - evaluator.best_y

            self.log.append({
                "gen": gen, "fes": evaluator.n_fes, "best": evaluator.best_y,
                "n_final": len(finalists_idx), "jumpout": decision.inject_jumpout,
                "pool": len(pool), "gain": gain, "reason": decision.reason,
            })
            if self.verbose:
                print(f"gen {gen:3d} | fes {evaluator.n_fes:3d} | best {evaluator.best_y:.4e} "
                      f"| final {len(finalists_idx)} | {decision.reason}", flush=True)

        return evaluator.best_x, evaluator.best_y

"""
Idea 2: Budget-aware scheduler.

Instead of a fixed rule like "call the LLM every 3 generations", a scheduler
watches the search state each generation and decides *what to do* and *how
hard to spend*. It treats LLM help as an emergency measure, not a routine.

Signals it monitors
-------------------
* diversity   : mean pairwise distance in the population, normalized by the
                box diagonal. Low  -> population collapsing (premature
                convergence risk).
* stagnation  : number of consecutive generations with no meaningful
                improvement of the best value.
* budget_frac : fraction of the real-evaluation budget already spent.

Decisions it returns each generation (a `Decision`)
---------------------------------------------------
* n_final       : how many finalists the two-stage funnel should really
                  evaluate this generation. Early / budget-rich -> more
                  (build history); late / budget-poor -> fewer (be frugal,
                  trust the surrogate more).
* inject_jumpout: whether to ask the LLM for exploratory "jump-out" solutions
                  this generation (triggered by low diversity OR stagnation).
* reason        : human-readable explanation, logged for analysis.

This module is pure logic (no LLM, no evaluation) so it is trivial to unit
test and reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Decision:
    n_final: int
    inject_jumpout: bool
    reason: str


@dataclass
class BudgetAwareScheduler:
    # thresholds
    div_low: float = 0.05          # below this normalized diversity -> collapsing
    stagnation_patience: int = 5   # generations w/o improvement -> stuck
    improve_tol: float = 1e-6      # relative improvement considered "real"
    # funnel finalist counts for the three budget regimes
    n_final_rich: int = 4          # budget_frac < rich_thr
    n_final_mid: int = 3
    n_final_poor: int = 2          # budget_frac > poor_thr
    rich_thr: float = 0.4
    poor_thr: float = 0.75
    # internal state
    _best: float = field(default=float("inf"), repr=False)
    _stagnation: int = field(default=0, repr=False)

    def _diversity(self, pop: np.ndarray, lb, ub) -> float:
        if len(pop) < 2:
            return 0.0
        # mean pairwise distance, normalized by the box diagonal
        diag = np.linalg.norm(np.asarray(ub) - np.asarray(lb)) + 1e-12
        n = len(pop)
        total, cnt = 0.0, 0
        for i in range(n):
            d = np.linalg.norm(pop[i + 1:] - pop[i], axis=1)
            total += d.sum()
            cnt += len(d)
        return (total / cnt) / diag if cnt else 0.0

    def update(self, best_now: float, pop: np.ndarray, lb, ub, budget_frac: float) -> Decision:
        # ---- stagnation tracking ----
        if self._best == float("inf"):
            improved = True
        else:
            rel = (self._best - best_now) / (abs(self._best) + 1e-12)
            improved = rel > self.improve_tol
        if improved:
            self._stagnation = 0
            self._best = min(self._best, best_now)
        else:
            self._stagnation += 1

        diversity = self._diversity(np.asarray(pop), lb, ub)

        # ---- how many finalists to really evaluate (budget regime) ----
        if budget_frac < self.rich_thr:
            n_final = self.n_final_rich
            regime = "rich"
        elif budget_frac > self.poor_thr:
            n_final = self.n_final_poor
            regime = "poor"
        else:
            n_final = self.n_final_mid
            regime = "mid"

        # ---- jump-out trigger: collapsing diversity OR stagnation ----
        low_div = diversity < self.div_low
        stuck = self._stagnation >= self.stagnation_patience
        inject = low_div or stuck

        reasons = [f"budget={budget_frac:.2f}({regime})", f"div={diversity:.3f}"]
        if low_div:
            reasons.append("LOW-DIVERSITY->jumpout")
        if stuck:
            reasons.append(f"STAGNATION({self._stagnation})->jumpout")
        if inject:
            self._stagnation = 0  # reset after intervening

        return Decision(n_final=n_final, inject_jumpout=inject, reason=", ".join(reasons))

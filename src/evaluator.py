"""
Budget-controlled evaluator.

The single most important object in expensive optimization: it wraps the
true objective function and enforces a hard cap on the number of *real*
function evaluations (FEs). Every method in this project must go through
this object so all comparisons happen under the exact same budget.

It also records the full history of evaluated points and the best-so-far
value after each evaluation, which we use to draw convergence curves.
"""

from __future__ import annotations

import numpy as np


class BudgetExceeded(Exception):
    """Raised when a real evaluation is attempted after the budget is spent."""


class Evaluator:
    def __init__(self, problem, max_fes: int):
        self.problem = problem
        self.max_fes = int(max_fes)
        self.n_fes = 0
        # history of every real evaluation
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        # best-so-far value recorded after each real evaluation
        self.best_curve: list[float] = []
        self.best_x: np.ndarray | None = None
        self.best_y: float = float("inf")

    @property
    def remaining(self) -> int:
        return self.max_fes - self.n_fes

    def can_eval(self, k: int = 1) -> bool:
        return self.remaining >= k

    def evaluate(self, x: np.ndarray) -> float:
        """Spend one FE. Clips x into the box first."""
        if self.n_fes >= self.max_fes:
            raise BudgetExceeded(
                f"budget of {self.max_fes} FEs already spent"
            )
        x = self.problem.clip(np.asarray(x, dtype=float))
        val = float(self.problem(x))
        self.n_fes += 1
        self.X.append(x.copy())
        self.y.append(val)
        if val < self.best_y:
            self.best_y = val
            self.best_x = x.copy()
        self.best_curve.append(self.best_y)
        return val

    def evaluate_batch(self, xs) -> list[float]:
        """Evaluate a list of points, stopping if the budget runs out."""
        out = []
        for x in xs:
            if not self.can_eval():
                break
            out.append(self.evaluate(x))
        return out

    def history_arrays(self):
        """Return (X, y) as numpy arrays of everything evaluated so far."""
        if not self.X:
            return np.zeros((0, self.problem.dim)), np.zeros((0,))
        return np.asarray(self.X), np.asarray(self.y)

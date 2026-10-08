"""
BBOB-style single-objective test problems for expensive optimization.

We implement a small, self-contained subset of BBOB-like functions
(Sphere, Rosenbrock, Rastrigin). Each function can be given a random
optimum shift so that the optimum is NOT at the origin. This matters for
LLM-based methods: if the optimum sat at a "nice" point like the origin,
an LLM might guess it from prior knowledge instead of actually searching,
which would inflate the results ("memorization leakage").

All problems are minimization problems with a known optimal value of 0.0
(achieved at the shifted optimum).
"""

from __future__ import annotations

import numpy as np


class Problem:
    """A box-constrained minimization problem.

    Attributes
    ----------
    name : str
        Human readable name.
    dim : int
        Number of decision variables.
    lb, ub : np.ndarray
        Lower / upper bounds (shape (dim,)).
    x_opt : np.ndarray
        Location of the global optimum (shape (dim,)).
    f_opt : float
        Objective value at the optimum (always 0.0 here).
    """

    def __init__(self, name: str, dim: int, lb, ub, x_opt, f_opt: float = 0.0):
        self.name = name
        self.dim = dim
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.x_opt = np.asarray(x_opt, dtype=float)
        self.f_opt = float(f_opt)

    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Clip a point into the box constraints."""
        return np.clip(x, self.lb, self.ub)

    def random_point(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lb, self.ub)


class Sphere(Problem):
    """f(x) = sum((x - x_opt)^2). Convex, unimodal, easy."""

    def __call__(self, x: np.ndarray) -> float:
        z = np.asarray(x, dtype=float) - self.x_opt
        return float(np.sum(z * z))


class Rosenbrock(Problem):
    """Rosenbrock ("banana") valley. Unimodal but ill-conditioned.

    f(x) = sum_i [100 (z_{i+1} - z_i^2)^2 + (z_i - 1)^2], where z = x - x_opt + 1
    so that the global optimum value is 0 at x = x_opt.
    """

    def __call__(self, x: np.ndarray) -> float:
        z = np.asarray(x, dtype=float) - self.x_opt + 1.0
        return float(
            np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1.0) ** 2)
        )


class Rastrigin(Problem):
    """Rastrigin. Highly multimodal (many local optima), hard.

    f(x) = 10 d + sum(z^2 - 10 cos(2 pi z)),  z = x - x_opt
    """

    def __call__(self, x: np.ndarray) -> float:
        z = np.asarray(x, dtype=float) - self.x_opt
        d = z.shape[0]
        return float(10.0 * d + np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z)))


def make_problem(name: str, dim: int = 10, seed: int = 0) -> Problem:
    """Factory that builds a shifted problem instance.

    The optimum is placed at a random location inside the middle 80% of the
    box, deterministically derived from `seed`, so every run/method on the
    same seed sees the same instance.
    """
    rng = np.random.default_rng(1000 + seed)
    lb = np.full(dim, -5.0)
    ub = np.full(dim, 5.0)
    # keep the optimum away from the boundary
    x_opt = rng.uniform(lb + 1.0, ub - 1.0)

    name = name.lower()
    if name == "sphere":
        return Sphere("Sphere", dim, lb, ub, x_opt)
    if name == "rosenbrock":
        return Rosenbrock("Rosenbrock", dim, lb, ub, x_opt)
    if name == "rastrigin":
        return Rastrigin("Rastrigin", dim, lb, ub, x_opt)
    raise ValueError(f"unknown problem: {name}")


ALL_PROBLEMS = ["sphere", "rosenbrock", "rastrigin"]

"""
Self-implemented evolutionary operators (continuous search).

Nothing here is copied from a library; these are plain-numpy implementations
of standard operators used to turn the current population into a large pool
of candidate offspring. The two-stage LLM surrogate then decides which few
of those candidates are worth a real evaluation.
"""

from __future__ import annotations

import numpy as np


def latin_hypercube(n, lb, ub, rng):
    """Latin Hypercube sampling for a well-spread initial design."""
    lb = np.asarray(lb); ub = np.asarray(ub)
    d = lb.shape[0]
    cut = np.linspace(0, 1, n + 1)
    pts = np.empty((n, d))
    for j in range(d):
        u = rng.uniform(size=n)
        pts[:, j] = cut[:n] + u * (cut[1:] - cut[:n])
        rng.shuffle(pts[:, j])
    return lb + pts * (ub - lb)


def de_rand_1(pop, F, rng):
    """DE/rand/1 mutant vector for each individual."""
    n = len(pop)
    mutants = np.empty_like(pop)
    for i in range(n):
        idxs = [j for j in range(n) if j != i]
        r1, r2, r3 = pop[rng.choice(idxs, 3, replace=False)]
        mutants[i] = r1 + F * (r2 - r3)
    return mutants


def binomial_crossover(target, mutant, CR, rng):
    d = target.shape[0]
    mask = rng.uniform(size=d) < CR
    if not mask.any():
        mask[rng.integers(d)] = True
    return np.where(mask, mutant, target)


def blend_crossover(a, b, rng, alpha=0.5):
    """BLX-alpha crossover between two parents."""
    lo = np.minimum(a, b) - alpha * np.abs(a - b)
    hi = np.maximum(a, b) + alpha * np.abs(a - b)
    return rng.uniform(lo, hi)


def gaussian_mutation(x, lb, ub, rng, scale=0.1):
    step = scale * (np.asarray(ub) - np.asarray(lb)) * rng.standard_normal(x.shape[0])
    return x + step


def select_diverse(X, y, k, lb, ub, min_frac=0.02):
    """Pick `k` good-and-spread points (diversity-preserving selection).

    Greedily takes the best points by fitness but skips any that sit within
    `min_frac` of the box diagonal from an already-chosen point. This stops
    the working population from collapsing to a cluster of near-duplicates,
    which is what makes the scheduler's diversity signal meaningful instead
    of always reading ~0.
    """
    X = np.asarray(X); y = np.asarray(y, dtype=float)
    lb = np.asarray(lb); ub = np.asarray(ub)
    diag = np.linalg.norm(ub - lb) + 1e-12
    min_dist = min_frac * diag
    order = np.argsort(y)
    chosen = []
    for idx in order:
        if len(chosen) >= k:
            break
        if all(np.linalg.norm(X[idx] - X[c]) > min_dist for c in chosen):
            chosen.append(idx)
    # if spacing left us short, top up with the next-best remaining points
    if len(chosen) < k:
        for idx in order:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= k:
                break
    return np.asarray(chosen[:k])


def generate_pool(pop, pool_size, lb, ub, rng, F=0.6, CR=0.9):
    """Turn the population into a diverse pool of `pool_size` candidates.

    Mixes DE/rand/1 + binomial crossover, BLX crossover, and Gaussian
    mutation so the pool covers both exploitative and exploratory moves.
    All candidates are clipped into the box.
    """
    pop = np.asarray(pop)
    lb = np.asarray(lb); ub = np.asarray(ub)
    n = len(pop)
    pool = []
    mutants = de_rand_1(pop, F, rng)
    while len(pool) < pool_size:
        i = len(pool) % n
        mode = rng.integers(3)
        if mode == 0:  # DE
            child = binomial_crossover(pop[i], mutants[i], CR, rng)
        elif mode == 1:  # BLX between two random parents
            a, b = pop[rng.choice(n, 2, replace=False)]
            child = blend_crossover(a, b, rng)
        else:  # Gaussian mutation of a parent
            child = gaussian_mutation(pop[i], lb, ub, rng, scale=0.15)
        pool.append(np.clip(child, lb, ub))
    return np.asarray(pool)

"""
Idea 1: Two-stage hierarchical LLM surrogate (coarse screen -> fine judge).

A traditional EA produces MANY candidate offspring per generation, but under
an expensive budget we can only afford a FEW real evaluations. This module
decides *which* candidates deserve a real evaluation, using a cost-aware
funnel of two LLM tiers:

    pool (e.g. 50)  --[small model: cheap, noisy]-->  shortlist (e.g. 10)
    shortlist (10)  --[large model: pricey, sharp]-->  finalists (e.g. 3)

Only the finalists are really evaluated. Total LLM cost is
    small x 50  +  large x 10
which is far cheaper than
    large x 50
while losing little ranking accuracy, because the small model is good enough
to throw out the obviously bad candidates and the large model only refines
the survivors.

The funnel returns the indices of the chosen finalists plus a small log dict
for analysis.
"""

from __future__ import annotations

import numpy as np


class TwoStageSurrogate:
    def __init__(self, llm, n_shortlist=10, n_final=3, single_tier=False):
        self.llm = llm
        self.n_shortlist = int(n_shortlist)
        self.n_final = int(n_final)
        # single_tier=True is an ablation of Idea 1: skip the coarse screen and
        # score the whole pool with the large (expensive) model directly.
        self.single_tier = bool(single_tier)

    def select(self, pool, ctx_X, ctx_y, n_final=None):
        """Run the coarse->fine funnel over `pool`.

        Parameters
        ----------
        pool : (m, d) array of candidate solutions
        ctx_X, ctx_y : evaluated history used as in-context examples
        n_final : override the number of finalists (the scheduler uses this
                  to spend fewer/more real FEs depending on budget state)

        Returns
        -------
        finalists_idx : indices into `pool` chosen for real evaluation
        info : dict with intermediate rankings for logging
        """
        pool = np.atleast_2d(pool)
        m = len(pool)
        n_final = self.n_final if n_final is None else int(n_final)
        n_final = max(1, min(n_final, m))
        n_shortlist = max(n_final, min(self.n_shortlist, m))

        # ---- Stage 1: coarse screen with the SMALL (cheap) model ----------
        if self.single_tier:
            # ablation: no coarse screen; the whole pool goes to the large model
            shortlist_idx = np.arange(m)
        elif m <= n_shortlist:
            shortlist_idx = np.arange(m)
        else:
            coarse = self.llm.score(pool, ctx_X, ctx_y, tier="small")
            shortlist_idx = np.argsort(coarse)[:n_shortlist]

        # ---- Stage 2: fine judge with the LARGE (accurate) model ----------
        if len(shortlist_idx) <= n_final:
            finalists_idx = shortlist_idx
            fine = None
        else:
            fine = self.llm.score(
                pool[shortlist_idx], ctx_X, ctx_y, tier="large"
            )
            finalists_local = np.argsort(fine)[:n_final]
            finalists_idx = shortlist_idx[finalists_local]

        info = {
            "pool_size": m,
            "shortlist_size": len(shortlist_idx),
            "n_final": len(finalists_idx),
        }
        return np.asarray(finalists_idx), info

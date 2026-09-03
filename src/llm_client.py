"""
LLM client abstraction.

The solver never talks to a concrete LLM directly; it talks to this
interface. That lets us swap between:

  * ApiLLMClient  -- a real OpenAI-compatible endpoint (e.g. aihubmix,
                     OpenAI, DeepSeek). Works with any base_url + api_key.
                     Two "tiers" map to two model names: a cheap/fast
                     small model and an accurate/expensive large model.

  * MockLLMClient -- a fully offline simulation used for development and
                     for reproducible experiments without an API key.
                     It emulates "LLM-as-surrogate" behavior by fitting a
                     tiny internal regressor on the in-context (x, f(x))
                     pairs and adding tier-dependent noise: the small tier
                     is noisier (cheap but rough), the large tier is more
                     accurate (expensive but sharp). This is exactly the
                     accuracy/cost trade-off the two-stage funnel exploits.

Both clients expose the SAME two operations used by the solver:

  score(candidates, ctx_X, ctx_y, tier)  -> predicted objective per candidate
                                            (LOWER = better; minimization)
  propose(ctx_X, ctx_y, lb, ub, n, rng)  -> n new exploratory solutions

Every LLM call is counted (`n_calls_small`, `n_calls_large`) so the report
can compare LLM cost across methods.
"""

from __future__ import annotations

import json
import os
import re
import numpy as np


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------
class LLMClient:
    def __init__(self):
        self.n_calls_small = 0
        self.n_calls_large = 0
        self.n_scored_small = 0
        self.n_scored_large = 0

    def _count(self, tier: str, n: int):
        if tier == "small":
            self.n_calls_small += 1
            self.n_scored_small += n
        else:
            self.n_calls_large += 1
            self.n_scored_large += n

    def score(self, candidates, ctx_X, ctx_y, tier):
        raise NotImplementedError

    def propose(self, ctx_X, ctx_y, lb, ub, n, rng):
        raise NotImplementedError

    def cost_summary(self) -> dict:
        return {
            "llm_calls_small": self.n_calls_small,
            "llm_calls_large": self.n_calls_large,
            "llm_scored_small": self.n_scored_small,
            "llm_scored_large": self.n_scored_large,
        }


# --------------------------------------------------------------------------
# Offline mock (used for testing / reproducible experiments)
# --------------------------------------------------------------------------
class MockLLMClient(LLMClient):
    """Simulates a hierarchy of LLM surrogates without any network.

    Parameters
    ----------
    small_noise, large_noise : float
        Relative noise levels for the two tiers. small_noise > large_noise
        encodes "the small model is rougher than the large model".
    k : int
        Neighbors used by the internal inverse-distance regressor.
    """

    def __init__(self, small_noise=0.35, large_noise=0.10, k=5, seed=0):
        super().__init__()
        self.small_noise = float(small_noise)
        self.large_noise = float(large_noise)
        self.k = int(k)
        self._rng = np.random.default_rng(seed)

    def _predict(self, candidates, ctx_X, ctx_y):
        """Inverse-distance-weighted k-NN regression on the context."""
        candidates = np.atleast_2d(candidates)
        if len(ctx_X) == 0:
            return np.zeros(len(candidates))
        preds = np.empty(len(candidates))
        k = min(self.k, len(ctx_X))
        for i, c in enumerate(candidates):
            d = np.linalg.norm(ctx_X - c, axis=1)
            idx = np.argsort(d)[:k]
            dk = d[idx]
            if dk[0] < 1e-12:  # candidate coincides with a known point
                preds[i] = ctx_y[idx[0]]
                continue
            w = 1.0 / (dk ** 2 + 1e-12)
            preds[i] = np.sum(w * ctx_y[idx]) / np.sum(w)
        return preds

    def score(self, candidates, ctx_X, ctx_y, tier):
        candidates = np.atleast_2d(candidates)
        ctx_X = np.asarray(ctx_X)
        ctx_y = np.asarray(ctx_y, dtype=float)
        self._count(tier, len(candidates))

        base = self._predict(candidates, ctx_X, ctx_y)
        # noise is scaled by the spread of the context objective values so it
        # is meaningful regardless of the function's magnitude
        spread = np.std(ctx_y) + 1e-9
        sigma = (self.small_noise if tier == "small" else self.large_noise) * spread
        noise = self._rng.normal(0.0, sigma, size=len(candidates))
        return base + noise

    def propose(self, ctx_X, ctx_y, lb, ub, n, rng):
        """Emulate an LLM asked for 'jump-out' exploratory solutions.

        Strategy: half uniformly random (global exploration), half as large
        Gaussian perturbations of the current best (local escape).
        """
        lb = np.asarray(lb); ub = np.asarray(ub)
        d = lb.shape[0]
        out = []
        ctx_X = np.asarray(ctx_X)
        ctx_y = np.asarray(ctx_y, dtype=float)
        best = ctx_X[int(np.argmin(ctx_y))] if len(ctx_X) else (lb + ub) / 2
        for i in range(n):
            if i % 2 == 0 or len(ctx_X) == 0:
                out.append(rng.uniform(lb, ub))
            else:
                step = 0.3 * (ub - lb) * rng.standard_normal(d)
                out.append(np.clip(best + step, lb, ub))
        return np.asarray(out)


# --------------------------------------------------------------------------
# Real OpenAI-compatible endpoint
# --------------------------------------------------------------------------
class ApiLLMClient(LLMClient):
    """Talks to any OpenAI-compatible chat/completions endpoint.

    The small and large tiers are just two different model names on the same
    endpoint (e.g. "gpt-3.5-turbo" and "gpt-4o", or two open models). The
    solver logic is identical to the mock; only the scoring/proposing is
    delegated to a real model.
    """

    def __init__(
        self,
        base_url=None,
        api_key=None,
        small_model="gpt-3.5-turbo",
        large_model="gpt-4o",
        timeout=60,
    ):
        super().__init__()
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://aihubmix.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.small_model = small_model
        self.large_model = large_model
        self.timeout = timeout
        if not self.api_key:
            raise ValueError(
                "No API key. Set LLM_API_KEY env var or pass api_key=..., "
                "or use MockLLMClient for offline runs."
            )

    def _chat(self, model, system, user):
        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _fmt_context(ctx_X, ctx_y, max_points=15):
        """Format the best context points as compact text for the prompt."""
        ctx_X = np.asarray(ctx_X)
        ctx_y = np.asarray(ctx_y, dtype=float)
        order = np.argsort(ctx_y)[:max_points]
        lines = []
        for j in order:
            xs = ", ".join(f"{v:.3f}" for v in ctx_X[j])
            lines.append(f"  f([{xs}]) = {ctx_y[j]:.4f}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json_array(text):
        """Extract the first JSON array of numbers from a model reply."""
        m = re.search(r"\[\s*[-+0-9.eE,\s\[\]]*\]", text)
        if not m:
            raise ValueError(f"no JSON array found in reply: {text[:200]}")
        return json.loads(m.group(0))

    def score(self, candidates, ctx_X, ctx_y, tier):
        candidates = np.atleast_2d(candidates)
        self._count(tier, len(candidates))
        model = self.small_model if tier == "small" else self.large_model
        ctx = self._fmt_context(ctx_X, ctx_y)
        cand_lines = "\n".join(
            f"  #{i}: [{', '.join(f'{v:.3f}' for v in c)}]"
            for i, c in enumerate(candidates)
        )
        system = (
            "You are a surrogate model for an expensive minimization problem. "
            "Given known (x, f(x)) samples, estimate f(x) for new points. "
            "Reply ONLY with a JSON array of floats, one per candidate, in order."
        )
        user = (
            f"Known samples (lower f is better):\n{ctx}\n\n"
            f"Estimate f for these {len(candidates)} candidates:\n{cand_lines}\n\n"
            f"Return a JSON array of {len(candidates)} floats."
        )
        try:
            reply = self._chat(model, system, user)
            vals = np.asarray(self._parse_json_array(reply), dtype=float).ravel()
            if len(vals) != len(candidates):
                raise ValueError("length mismatch")
            return vals
        except Exception:
            # robust fallback: if the model misbehaves, defer to nearest-neighbor
            return _nn_fallback(candidates, ctx_X, ctx_y)

    def propose(self, ctx_X, ctx_y, lb, ub, n, rng):
        lb = np.asarray(lb); ub = np.asarray(ub)
        ctx = self._fmt_context(ctx_X, ctx_y)
        system = (
            "You explore the search space of an expensive minimization problem. "
            "Propose diverse, promising new points that are different from the "
            "known ones. Reply ONLY with a JSON array of arrays (each inner "
            "array is one point)."
        )
        bounds = ", ".join(f"[{lo:.1f},{hi:.1f}]" for lo, hi in zip(lb, ub))
        user = (
            f"Known samples:\n{ctx}\n\n"
            f"Variable bounds per dimension: {bounds}\n"
            f"Propose {n} new points as a JSON array of arrays."
        )
        try:
            reply = self._chat(self.large_model, system, user)
            arr = np.asarray(json.loads(re.search(r"\[.*\]", reply, re.S).group(0)), dtype=float)
            arr = np.atleast_2d(arr)
            return np.clip(arr[:n], lb, ub)
        except Exception:
            return np.array([rng.uniform(lb, ub) for _ in range(n)])


def _nn_fallback(candidates, ctx_X, ctx_y):
    ctx_X = np.asarray(ctx_X); ctx_y = np.asarray(ctx_y, dtype=float)
    preds = []
    for c in np.atleast_2d(candidates):
        d = np.linalg.norm(ctx_X - c, axis=1)
        preds.append(ctx_y[int(np.argmin(d))])
    return np.asarray(preds)

# LLM + EA Solver for Expensive Optimization

A small, self-contained solver that combines a **Large Language Model (LLM)**
with an **Evolutionary Algorithm (EA)** to solve *expensive* black-box
optimization problems — where you are only allowed a tiny number of real
function evaluations (here: **300 FEs**).

It implements two ideas that are largely unexplored in the literature:

1. **Two-stage hierarchical LLM surrogate (coarse screen → fine judge).**
   The EA proposes a large pool of candidates each generation; a *cheap small
   model* coarsely screens them all, then an *accurate large model* re-ranks
   only the survivors. Only the top few finalists are really evaluated. This
   buys most of the large model's accuracy at a fraction of its cost.

2. **Budget-aware scheduler.** Instead of calling the LLM on a fixed schedule,
   a scheduler watches population **diversity**, **stagnation**, and
   **remaining budget**, and decides *how many* finalists to evaluate and
   *whether* to inject LLM "jump-out" solutions. LLM help becomes an emergency
   measure, not routine.

> **Single-objective** track (BBOB-style functions), 300-FE budget, as allowed
> by the assignment (single- OR multi-objective).

## Quick start

```bash
pip install -r requirements.txt

# reproduce all experiments (offline mock LLM, no API key needed)
python -m experiments.run_experiment --seeds 5 --dim 10 --fes 300

# run the unit tests
python tests/test_core.py
```

Results are written to `results/`:
`summary.csv`, `llm_cost.csv`, `curves_<fn>.csv`, `convergence.png`.

## Using a real LLM (aihubmix / OpenAI-compatible)

The solver talks to an abstract `LLMClient`, so nothing in the algorithm
changes. To use a real endpoint:

```bash
export LLM_API_KEY="sk-..."                       # your key
export LLM_BASE_URL="https://aihubmix.com/v1"     # or any OpenAI-compatible URL
export LLM_SMALL_MODEL="gpt-3.5-turbo"            # cheap tier
export LLM_LARGE_MODEL="gpt-4o-mini"              # accurate tier

python -m experiments.run_experiment --real --seeds 3
```

Without `--real`, an **offline `MockLLMClient`** is used. It emulates the two
tiers with a small internal regressor plus tier-dependent noise (small = rough,
large = sharp), so the whole pipeline — scheduler, funnel, budget control — runs
and is testable with no network. The *algorithm* is identical in both modes.

## Layout

```
src/
  problems.py    # BBOB-style test functions (Sphere/Rosenbrock/Rastrigin), shifted
  evaluator.py   # budget-controlled evaluator (enforces the 300-FE cap)
  llm_client.py  # LLMClient abstraction: MockLLMClient + ApiLLMClient (2 tiers)
  surrogate.py   # Idea 1: two-stage coarse->fine funnel
  scheduler.py   # Idea 2: budget-aware scheduler
  ea.py          # self-implemented EA operators (LHS, DE, BLX, mutation, pool)
  solver.py      # main loop tying the two ideas together
  baselines.py   # random search + standard DE
experiments/run_experiment.py   # runs all methods, saves CSVs + plot
tests/test_core.py              # unit tests for budget / funnel / scheduler
docs/technical_report.md        # architecture, logic, results & analysis
```

See `docs/technical_report.md` for the full write-up.

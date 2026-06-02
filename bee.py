"""
abc_c.py
========
Artificial Bee Colony (ABC) algorithm for optimising real-valued neural
network weights.

CONCEPTUAL OVERVIEW
-------------------
The Artificial Bee Colony algorithm (Karaboga, 2005) mimics the foraging
behaviour of honey bees.  The colony consists of three groups:

  • Employed Bees  — exploit known food sources (solutions) by generating
                     a new solution near their current food source and
                     keeping the better one (greedy selection).

  • Onlooker Bees  — watch the waggle dance of employed bees and choose
                     which food source to exploit with probability
                     proportional to its fitness (roulette selection).
                     They also generate and greedily compare a neighbour.

  • Scout Bees     — when a food source has not improved for `limit`
                     consecutive trials, the employed bee abandons it and
                     becomes a scout, generating a completely new random
                     food source.  This is ABC's built-in restart mechanism
                     against local optima.

Each food source is a candidate weight vector θ ∈ ℝ^n_params.

KEY EQUATIONS
-------------
Neighbour generation (employed / onlooker):

    v_{i,j} = x_{i,j} + φ_{i,j} * (x_{i,j} - x_{k,j})

where:
  i     — current bee's food source index
  k     — randomly chosen partner (k ≠ i)
  j     — randomly chosen dimension to perturb
  φ     — uniform random number in [-1, +1]

The perturbation is thus scaled by the *distance* to a randomly chosen
partner, giving DE-like self-adaptation without an explicit F parameter.

Fitness mapping for maximisation (ABC was designed for minimisation):

    f_sel(x_i) = 1 / (1 + |f(x_i)|)   if f(x_i) < 0
    f_sel(x_i) = 1 + f(x_i)            if f(x_i) ≥ 0

We are maximising F1-score ∈ [0,1], so f_sel simplifies to 1 + F1 ∈ [1,2],
which gives a valid roulette-wheel probability distribution.

HOW ABC DIFFERS FROM GA, PSO, DE
----------------------------------
  GA            PSO            DE              ABC
  ─────────────────────────────────────────────────────────────────────
  Crossover     Velocity       Difference      Neighbourhood perturbation
  selection     bests          vector          + roulette selection
  Elitism       Memory         Greedy 1-to-1   Greedy 1-to-1 + scouts
  Mutation σ    w decay        F, CR           φ, limit
  No explicit   No explicit    No explicit     Scout restarts (limit)
  restart       restart        restart         (explicit stagnation cure)

ABC is known for good local-search ability through employed bees, and for
escaping local optima through scout restarts.  The scout mechanism is
especially relevant for the Parkinson's task, where degenerate solutions
(recall=1, precision=low) are local optima that other algorithms get stuck in.

PARAMETERS
----------
  pop_size  : colony size (= number of food sources = number of employed bees)
              Onlooker count = pop_size  (canonical ABC uses equal split)
  limit     : maximum number of trials before a bee abandons a source
              Recommended: limit = pop_size × n_params / 2
              Large limit → more exploitation; small limit → more exploration

Usage
-----
    from parkinsons_problem_c import fitness_function, compute_n_params
    from abc_c import artificial_bee_colony

    best_theta, best_fitness, history = artificial_bee_colony(
        fitness_fn   = fitness_function,
        n_params     = compute_n_params(),
        fitness_args = (X, y),
        pop_size     = 50,
        n_iterations = 100,
        limit        = 100,
    )
"""

import numpy as np
from typing import Callable, List, Optional, Tuple


def artificial_bee_colony(
    fitness_fn:   Callable,
    n_params:     int,
    fitness_args: tuple         = (),
    pop_size:     int           = 50,
    n_iterations: int           = 100,
    limit:        Optional[int] = None,   # abandonment threshold per source
    low:          float         = -1.0,
    high:         float         = 1.0,
    tol:          float         = 1e-8,
    patience:     int           = 30,
    seed:         Optional[int] = None,
    verbose:      bool          = True,
    # Architecture kwargs (passed through for logging)
    input_size:   int           = 22,
    hidden1_size: int           = 10,
    hidden2_size: int           = 10,
    output_size:  int           = 1,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Artificial Bee Colony (ABC) main loop.

    Parameters
    ----------
    fitness_fn    : Callable — f(solution, *fitness_args) → float (higher = better)
    n_params      : int      — dimensionality of the search space (n_params = 351)
    fitness_args  : tuple    — extra args forwarded to fitness_fn, e.g. (X, y)
    pop_size      : int      — number of food sources (= employed bees)    (default 50)
                              Onlooker bees are also pop_size.
    n_iterations  : int      — maximum number of colony cycles             (default 100)
    limit         : int|None — maximum trials before abandonment per source.
                              If None, set to pop_size * n_params / 2 (canonical).
    low           : float    — lower bound for initial food sources        (default -1.0)
    high          : float    — upper bound for initial food sources        (default +1.0)
    tol           : float    — minimum improvement to reset patience counter
    patience      : int      — iterations without global improvement before stopping
    seed          : int|None — RNG seed for reproducibility
    verbose       : bool     — print progress every 10 iterations

    Returns
    -------
    best_position : np.ndarray, shape (n_params,) — best weight vector θ found
    best_fitness  : float                          — fitness of best_position (F1)
    history       : List[float]                    — global best fitness per cycle
    """
    if seed is not None:
        np.random.seed(seed)

    # Canonical default: limit = pop_size × n_params / 2
    if limit is None:
        limit = max(10, int(pop_size * n_params / 2))

    n_onlookers = pop_size   # canonical ABC: equal employed/onlooker counts

    # ── 1. INITIALISATION ────────────────────────────────────────────────────
    # Food sources: uniform in [low, high]
    sources  = np.random.uniform(low, high, size=(pop_size, n_params))
    fits     = np.array([fitness_fn(sources[i], *fitness_args)
                         for i in range(pop_size)])
    trials   = np.zeros(pop_size, dtype=int)   # trial counter per source

    best_idx      = int(np.argmax(fits))
    best_position = sources[best_idx].copy()
    best_fitness  = float(fits[best_idx])

    history         = []
    no_improve_count = 0

    if verbose:
        print(f"\n{'='*58}")
        print(f"  Artificial Bee Colony  (ABC)")
        print(f"  n_params={n_params}  pop={pop_size}  iters={n_iterations}")
        print(f"  limit={limit}  onlookers={n_onlookers}")
        print(f"  arch: {input_size}→{hidden1_size}→{hidden2_size}→{output_size}")
        print(f"{'='*58}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _selection_fitness(f_vals):
        """
        Convert raw fitness values to selection probabilities for roulette
        wheel selection by onlooker bees.

        For maximisation with F1 ∈ [0,1]:
            sel = 1 + f  → range [1, 2]
        Probabilities are then normalised to sum to 1.
        """
        sel = np.where(f_vals >= 0, 1.0 + f_vals, 1.0 / (1.0 + np.abs(f_vals)))
        return sel / sel.sum()

    def _generate_neighbour(source_idx):
        """
        Generate one neighbour of source[source_idx] by perturbing a single
        randomly chosen dimension using a random partner source.

        v_{i,j} = x_{i,j} + φ * (x_{i,j} - x_{k,j})
        where φ ∈ Uniform(-1, 1), k ≠ i, j ~ Uniform({0, …, n_params-1})
        """
        i = source_idx
        # pick a random partner (≠ i)
        k = i
        while k == i:
            k = np.random.randint(0, pop_size)

        # pick a random dimension to perturb
        j = np.random.randint(0, n_params)

        # generate the neighbour
        phi     = np.random.uniform(-1.0, 1.0)
        neighbour = sources[i].copy()
        neighbour[j] = sources[i, j] + phi * (sources[i, j] - sources[k, j])
        return neighbour

    # ── 2. MAIN LOOP ──────────────────────────────────────────────────────────
    for iteration in range(n_iterations):

        # ── 2a. EMPLOYED BEE PHASE ────────────────────────────────────────────
        # Each employed bee generates a neighbour of its food source
        # and keeps the better solution (greedy selection).
        for i in range(pop_size):
            neighbour     = _generate_neighbour(i)
            neighbour_fit = fitness_fn(neighbour, *fitness_args)

            if neighbour_fit >= fits[i]:
                sources[i] = neighbour
                fits[i]    = neighbour_fit
                trials[i]  = 0       # improvement: reset trial counter
            else:
                trials[i] += 1       # no improvement: increment counter

        # ── 2b. ONLOOKER BEE PHASE ───────────────────────────────────────────
        # Onlooker bees choose food sources with probability proportional
        # to their fitness (roulette-wheel / fitness-proportionate selection).
        # They then exploit those sources the same way as employed bees.
        probs = _selection_fitness(fits)

        for _ in range(n_onlookers):
            # Roulette selection: choose food source i according to probs
            i = int(np.random.choice(pop_size, p=probs))

            neighbour     = _generate_neighbour(i)
            neighbour_fit = fitness_fn(neighbour, *fitness_args)

            if neighbour_fit >= fits[i]:
                sources[i] = neighbour
                fits[i]    = neighbour_fit
                trials[i]  = 0
            else:
                trials[i] += 1

        # ── 2c. SCOUT BEE PHASE ───────────────────────────────────────────────
        # Any food source whose trial counter exceeds `limit` is abandoned.
        # The corresponding employed bee becomes a scout and initialises a
        # completely new random food source.  This is ABC's restart mechanism.
        for i in range(pop_size):
            if trials[i] >= limit:
                sources[i] = np.random.uniform(low, high, size=n_params)
                fits[i]    = fitness_fn(sources[i], *fitness_args)
                trials[i]  = 0

        # ── 2d. Update global best ────────────────────────────────────────────
        gen_best_idx = int(np.argmax(fits))
        gen_best_fit = float(fits[gen_best_idx])

        if gen_best_fit > best_fitness + tol:
            best_fitness  = gen_best_fit
            best_position = sources[gen_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        history.append(best_fitness)

        if verbose and (iteration % 10 == 0 or iteration == n_iterations - 1):
            n_scouts = int((trials >= limit).sum())
            mean_fit = float(fits.mean())
            print(f"  Cycle {iteration+1:4d}/{n_iterations}  |  "
                  f"Best: {best_fitness:.4f}  |  "
                  f"Mean: {mean_fit:.4f}  |  "
                  f"Scouts-pending: {n_scouts:2d}  |  "
                  f"No-improve: {no_improve_count:3d}")

        # ── 2e. Early stopping ────────────────────────────────────────────────
        if no_improve_count >= patience:
            if verbose:
                print(f"\n  Early stopping at cycle {iteration+1} "
                      f"(no improvement for {patience} cycles).")
            history += [best_fitness] * (n_iterations - len(history))
            break

    if verbose:
        print(f"\n  ✓ ABC finished.  Best fitness = {best_fitness:.4f}"
              f"  ({best_fitness*100:.1f}% F1-score)\n")

    return best_position, best_fitness, history


# ===========================================================================
# Quick self-test
# ===========================================================================
if __name__ == '__main__':
    import pandas as pd
    from my_parkinsons_problem import fitness_function, compute_n_params, evaluate_solution

    df       = pd.read_csv('parkinsons_preprocessed.csv')
    X        = df.drop(columns=['status']).values
    y        = df['status'].values
    n_params = compute_n_params()

    print("Running ABC (default hyperparameters) ...")
    best_theta, best_f1, hist = artificial_bee_colony(
        fitness_fn    = fitness_function,
        n_params      = n_params,
        fitness_args  = (X, y),
        pop_size      = 30,
        n_iterations  = 50,
        limit         = 100,
        seed          = 42,
        verbose       = True,
    )

    metrics = evaluate_solution(best_theta, X, y)
    print(f"\nFull evaluation of best ABC solution:")
    print(f"  F1        : {metrics['f1']:.4f}  ← fitness signal")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Predicted positive : {metrics['n_pred_pos']}")
    print(f"  Predicted negative : {metrics['n_pred_neg']}")
    print(f"\nHistory (last 5 cycles): {[round(h,4) for h in hist[-5:]]}")
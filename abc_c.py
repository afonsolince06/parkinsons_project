"""
abc_c.py
========
Artificial Bee Colony (ABC) for optimising real-valued neural network weights.

CONCEPTUAL OVERVIEW
-------------------
ABC was introduced by Karaboga (2005), inspired by the foraging behaviour of
honey bees.  The colony is divided into three groups:

  Employed Bees   — each associated with one food source (solution).
                    They exploit their source by searching in its neighbourhood.
  Onlooker Bees   — watch the waggle dance of employed bees and probabilistically
                    choose a source proportional to its fitness (quality).
  Scout Bees      — generated when a source is abandoned (exhausted after `limit`
                    failed improvement attempts).  They explore randomly.

ALGORITHM STEPS
----------------
1. Initialise `n_employed` food sources randomly.
2. EMPLOYED BEE PHASE:
   For each food source i, generate a neighbour by perturbing one random
   dimension using a random partner k ≠ i:
       v_j = x_i_j + φ * (x_i_j - x_k_j)    φ ~ Uniform[-1, 1]
   If f(v) ≥ f(x_i), replace x_i with v and reset its trial counter.
   Otherwise increment trial[i].
3. ONLOOKER BEE PHASE:
   Each of the `n_onlooker` bees selects a source with probability
       p_i = f_i / Σ f_k
   and applies the same neighbourhood search as employed bees.
4. SCOUT BEE PHASE:
   Any source whose trial counter ≥ `limit` is abandoned and replaced by a
   new random solution (scout exploration).
5. Record global best, repeat for `n_iterations` cycles.

HOW ABC DIFFERS FROM GA, PSO, DE
---------------------------------
  GA/PSO/DE          ABC
  ────────────────────────────────────────────────────────
  Global view         Division of labour (employed/onlooker/scout)
  Crossover/mutation  Neighbour search (1-dim perturbation)
  Elitism             Fitness-proportionate onlooker selection
  Velocity/mutation   Scout phase replaces exhausted sources

ABC is often competitive with PSO and DE on multimodal problems while being
easy to implement and requiring few hyperparameters.

PARAMETERS
----------
  n_bees       : total colony size = n_employed + n_onlooker.
                 By convention n_employed = n_onlooker = n_bees // 2.
  limit        : maximum number of failed improvement attempts before a source
                 is abandoned and replaced by a scout.  A common heuristic:
                 limit = n_employed * n_params.
  n_iterations : number of complete ABC cycles.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional

def artificial_bee_colony(
    fitness_fn:   Callable,
    n_params:     int,
    fitness_args: tuple         = (),
    n_bees:       int           = 40,
    limit:        int           = None,
    n_iterations: int           = 100,
    low:          float         = -1.0,
    high:         float         = 1.0,
    seed:         Optional[int] = None,
    verbose:      bool          = True,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Artificial Bee Colony (ABC) main loop.

    Parameters
    ----------
    fitness_fn   : Callable — f(solution, *fitness_args) → float (higher = better)
    n_params     : int      — dimensionality of the search space
    fitness_args : tuple    — extra args forwarded to fitness_fn, e.g. (X, y)
    n_bees       : int      — total colony size; n_employed = n_onlooker = n_bees // 2
    limit        : int|None — abandonment threshold; defaults to n_employed * n_params
    n_iterations : int      — maximum number of ABC cycles
    low          : float    — lower bound of initial positions
    high         : float    — upper bound of initial positions
    seed         : int|None — RNG seed for reproducibility
    verbose      : bool     — print progress every 10 iterations

    Returns
    -------
    best_solution : np.ndarray, shape (n_params,) — best weight vector found
    best_fitness  : float                          — F1-score of best_solution
    history       : List[float]                    — best F1 per iteration
    """
    if seed is not None:
        np.random.seed(seed)

    n_employed  = max(2, n_bees // 2)
    n_onlooker  = n_bees - n_employed
    if limit is None:
        limit = n_employed * n_params  # Karaboga's heuristic

    # ------------------------------------------------------------------
    # 1. INITIALISATION
    # ------------------------------------------------------------------
    sources   = np.random.uniform(low, high, size=(n_employed, n_params))
    fitnesses = np.array([
        fitness_fn(sources[i], *fitness_args) for i in range(n_employed)
    ])
    trials = np.zeros(n_employed, dtype=int)   # failure counters

    best_idx      = int(np.argmax(fitnesses))
    best_solution = sources[best_idx].copy()
    best_fitness  = float(fitnesses[best_idx])
    history       = []

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Artificial Bee Colony (ABC)")
        print(f"  n_employed={n_employed}  n_onlooker={n_onlooker}")
        print(f"  limit={limit}  iters={n_iterations}")
        print(f"{'='*55}")

    # ------------------------------------------------------------------
    # Helper: one neighbourhood search step for source i
    # ------------------------------------------------------------------
    def _search(i):
        """Perturb one random dimension of source i using a random partner."""
        k    = i
        while k == i:
            k = np.random.randint(0, n_employed)
        j    = np.random.randint(0, n_params)        # dimension to perturb
        phi  = np.random.uniform(-1.0, 1.0)          # perturbation scale
        v    = sources[i].copy()
        v[j] = sources[i, j] + phi * (sources[i, j] - sources[k, j])
        v[j] = np.clip(v[j], low, high)
        return v

    # ------------------------------------------------------------------
    # 2. MAIN LOOP
    # ------------------------------------------------------------------
    for it in range(n_iterations):

        # ── 2a. EMPLOYED BEE PHASE ──────────────────────────────────────
        for i in range(n_employed):
            v    = _search(i)
            f_v  = fitness_fn(v, *fitness_args)
            if f_v >= fitnesses[i]:
                sources[i]   = v
                fitnesses[i] = f_v
                trials[i]    = 0
                if f_v > best_fitness:
                    best_fitness  = f_v
                    best_solution = v.copy()
            else:
                trials[i] += 1

        # ── 2b. ONLOOKER BEE PHASE ──────────────────────────────────────
        # Selection probability proportional to fitness (shift to positive)
        shifted = fitnesses - fitnesses.min() + 1e-8
        probs   = shifted / shifted.sum()

        for _ in range(n_onlooker):
            i   = int(np.random.choice(n_employed, p=probs))
            v   = _search(i)
            f_v = fitness_fn(v, *fitness_args)
            if f_v >= fitnesses[i]:
                sources[i]   = v
                fitnesses[i] = f_v
                trials[i]    = 0
                if f_v > best_fitness:
                    best_fitness  = f_v
                    best_solution = v.copy()
            else:
                trials[i] += 1

        # ── 2c. SCOUT BEE PHASE ─────────────────────────────────────────
        for i in range(n_employed):
            if trials[i] >= limit:
                sources[i]   = np.random.uniform(low, high, size=n_params)
                fitnesses[i] = fitness_fn(sources[i], *fitness_args)
                trials[i]    = 0
                if fitnesses[i] > best_fitness:
                    best_fitness  = fitnesses[i]
                    best_solution = sources[i].copy()

        history.append(best_fitness)

        if verbose and (it % 10 == 0 or it == n_iterations - 1):
            mean_fit = fitnesses.mean()
            print(f"  Iter {it+1:4d}/{n_iterations}  |  "
                  f"Best F1: {best_fitness:.4f}  |  Mean F1: {mean_fit:.4f}")

    if verbose:
        print(f"\n  ✓ ABC finished.  Best F1 = {best_fitness:.4f}  "
              f"({best_fitness*100:.1f}%)\n")

    return best_solution, best_fitness, history


# ===========================================================================
# Quick self-test
# ===========================================================================
if __name__ == '__main__':
    import pandas as pd
    from my_parkinsons_problem import fitness_function, compute_n_params

    df = pd.read_csv('parkinsons_preprocessed.csv')
    X  = df.drop(columns=['status']).values
    y  = df['status'].values

    n_params = compute_n_params()
    print("Running ABC (default settings) ...")
    best_theta, best_f1, hist = artificial_bee_colony(
        fitness_fn    = fitness_function,
        n_params      = n_params,
        fitness_args  = (X, y),
        n_bees        = 40,
        n_iterations  = 50,
        seed          = 42,
        verbose       = True,
    )
    print(f"Best F1: {best_f1*100:.2f}%")
    print(f"History (last 5 iters): {[round(h,4) for h in hist[-5:]]}")

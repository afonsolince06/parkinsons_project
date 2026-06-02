"""
pso.py
======
Particle Swarm Optimization (PSO) for optimising real-valued neural
network weights.

CONCEPTUAL OVERVIEW
-------------------
PSO is inspired by the collective movement of bird flocks and fish schools.
Each particle is a candidate weight vector θ that "flies" through the
241-dimensional search space.  Every particle remembers:

  • Its current position   x_i   — the weight vector being evaluated
  • Its current velocity   v_i   — the direction and speed of movement
  • Its personal best      p_i   — best position it has personally found
  • The global best        g     — best position found by the entire swarm

At each iteration, each particle updates its velocity and position:

    v_i ← w·v_i
          + c1·r1·(p_i − x_i)     ← cognitive component (pull toward own best)
          + c2·r2·(g  − x_i)      ← social    component (pull toward swarm best)

    x_i ← x_i + v_i

where:
  w          — inertia weight: controls exploration vs exploitation
  c1, c2     — acceleration coefficients (cognitive, social)
  r1, r2     — random scalars in [0,1], different per particle per iteration

HOW PSO DIFFERS FROM GA
-----------------------
  GA                              PSO
  ──────────────────────────────────────────────────────────────────
  Discrete generations             Continuous, synchronous updates
  Recombination of parents         Velocity-guided individual movement
  Fitness drives selection         Fitness updates personal/global bests
  Population diversity via mutation Diversity via stochastic velocity
  Memory: none (population only)   Memory: each particle tracks own best
  Exploration: crossover/mutation  Exploration: inertia; exploitation: bests

PSO tends to converge faster on smooth, continuous landscapes (like neural
network weight space), while GA is more robust on rugged, multimodal
landscapes due to crossover diversity.  Comparing them on the same problem
is the core experiment of this project.

INERTIA WEIGHT STRATEGIES
--------------------------
Two strategies are provided:

  1. Constant inertia    — w is fixed throughout the run.
     Simple and predictable; good as a baseline.

  2. Linear decay (LDIW) — w decreases linearly from w_max to w_min.
     High w early  → large velocity → broad exploration.
     Low  w later  → small velocity → fine-grained exploitation.
     This is the most widely cited PSO enhancement in the literature.

VELOCITY CLAMPING
-----------------
Without clamping, velocities can grow unboundedly, causing particles to
"explode" out of the search space.  We clamp each velocity component to
[-v_max, +v_max] after every update, where v_max = (high - low) * 0.2
by default.  This keeps particles within a reasonable range without
hard position boundaries.

CONVERGENCE CRITERION
---------------------
Aside from the generation limit, an optional early-stopping criterion
halts the search if the global best fitness has not improved by more than
`tol` for `patience` consecutive iterations.  This saves compute when
the swarm has clearly converged.

Usage
-----
    from parkinsons_problem import fitness_function, compute_n_params
    from pso import particle_swarm_optimisation

    best_theta, best_fitness, history = particle_swarm_optimisation(
        fitness_fn   = fitness_function,
        n_params     = compute_n_params(),
        fitness_args = (X, y),
        n_particles  = 50,
        n_iterations = 100,
    )
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


def particle_swarm_optimisation(
    fitness_fn:      Callable,
    n_params:        int,
    fitness_args:    tuple         = (),
    n_particles:     int           = 50,
    n_iterations:    int           = 100,
    w:               float         = 0.9,
    w_min:           float         = 0.4,
    c1:              float         = 2.0,
    c2:              float         = 2.0,
    v_max_ratio:     float         = 0.2,
    low:             float         = -1.0,
    high:            float         = 1.0,
    inertia_strategy: str          = 'linear_decay',
    patience:        int           = 20,
    tol:             float         = 1e-6,
    seed:            Optional[int] = None,
    verbose:         bool          = True,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Particle Swarm Optimisation (PSO) main loop.

    Parameters
    ----------
    fitness_fn        : Callable — f(solution, *fitness_args) → float
    n_params          : int      — dimensionality of the search space (241)
    fitness_args      : tuple    — extra args forwarded to fitness_fn, e.g. (X, y)
    n_particles       : int      — swarm size                         (default 50)
    n_iterations      : int      — maximum number of iterations       (default 100)
    w                 : float    — inertia weight (or starting w for decay)
                                   Controls how much of the previous velocity
                                   is carried forward.
                                     w > 1.0  → accelerating (unstable)
                                     w = 0.9  → slow decay, broad exploration
                                     w = 0.4  → aggressive exploitation
    w_min             : float    — minimum inertia for linear decay   (default 0.4)
    c1                : float    — cognitive acceleration coefficient  (default 2.0)
                                   Strength of pull toward personal best.
    c2                : float    — social acceleration coefficient     (default 2.0)
                                   Strength of pull toward global best.
                                   c1 = c2 = 2.0 is the canonical Kennedy &
                                   Eberhart (1995) recommendation.
    v_max_ratio       : float    — v_max = (high-low) * v_max_ratio   (default 0.2)
                                   Clamps velocity magnitude per dimension.
    low               : float    — lower bound of initial positions   (default -1.0)
    high              : float    — upper bound of initial positions   (default +1.0)
    inertia_strategy  : str      — 'constant' | 'linear_decay'
    patience          : int      — early stopping: iterations without improvement
    tol               : float    — minimum improvement to reset patience counter
    seed              : int|None — RNG seed for reproducibility
    verbose           : bool     — print progress every 10 iterations

    Returns
    -------
    best_position : np.ndarray, shape (n_params,) — best weight vector found
    best_fitness  : float                          — fitness of best_position
    history       : List[float]                    — global best fitness per iter
    """
    if seed is not None:
        np.random.seed(seed)

    v_max = (high - low) * v_max_ratio   # velocity clamp magnitude

    # ------------------------------------------------------------------
    # 1. INITIALISATION
    #    Positions: uniform in [low, high]  (same range as GA)
    #    Velocities: uniform in [-v_max, +v_max]  (start with moderate motion)
    # ------------------------------------------------------------------
    positions  = np.random.uniform(low, high, size=(n_particles, n_params))
    velocities = np.random.uniform(-v_max, v_max, size=(n_particles, n_params))

    # Evaluate every particle's starting fitness
    fitnesses  = np.array([fitness_fn(positions[i], *fitness_args)
                           for i in range(n_particles)])

    # Personal bests — each particle starts as its own best
    personal_best_pos = positions.copy()
    personal_best_fit = fitnesses.copy()

    # Global best — the single best position seen by the whole swarm
    global_best_idx = np.argmax(fitnesses)
    global_best_pos = positions[global_best_idx].copy()
    global_best_fit = fitnesses[global_best_idx]

    history          = []
    no_improve_count = 0    # early-stopping counter

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Particle Swarm Optimisation (PSO)")
        print(f"  particles={n_particles}  iters={n_iterations}  "
              f"inertia={inertia_strategy}")
        print(f"  w={w}→{w_min}  c1={c1}  c2={c2}  v_max={v_max:.3f}")
        print(f"{'='*55}")

    # ------------------------------------------------------------------
    # 2. MAIN LOOP
    # ------------------------------------------------------------------
    for iteration in range(n_iterations):

        # --- 2a. Update inertia weight -----------------------------------
        if inertia_strategy == 'linear_decay':
            # Linearly interpolate from w (start) down to w_min (end).
            # At iteration 0 we use w; at the last iteration we use w_min.
            current_w = w - (w - w_min) * (iteration / max(n_iterations - 1, 1))
        else:  # 'constant'
            current_w = w

        # --- 2b. Velocity and position update for each particle ---------
        r1 = np.random.rand(n_particles, n_params)   # cognitive random matrix
        r2 = np.random.rand(n_particles, n_params)   # social    random matrix

        # Cognitive component: attraction toward each particle's personal best
        cognitive = c1 * r1 * (personal_best_pos - positions)

        # Social component: attraction toward the global best
        social    = c2 * r2 * (global_best_pos   - positions)

        # Velocity update equation (vectorised over all particles at once)
        velocities = current_w * velocities + cognitive + social

        # Velocity clamping — prevent runaway acceleration
        velocities = np.clip(velocities, -v_max, v_max)

        # Position update
        positions = positions + velocities

        # --- 2c. Evaluate new positions ----------------------------------
        fitnesses = np.array([fitness_fn(positions[i], *fitness_args)
                               for i in range(n_particles)])

        # --- 2d. Update personal bests -----------------------------------
        improved_mask = fitnesses > personal_best_fit
        personal_best_pos[improved_mask] = positions[improved_mask].copy()
        personal_best_fit[improved_mask] = fitnesses[improved_mask]

        # --- 2e. Update global best --------------------------------------
        current_best_idx = np.argmax(personal_best_fit)
        current_best_fit = personal_best_fit[current_best_idx]

        if current_best_fit > global_best_fit + tol:
            global_best_fit = current_best_fit
            global_best_pos = personal_best_pos[current_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        history.append(global_best_fit)

        if verbose and (iteration % 10 == 0 or iteration == n_iterations - 1):
            mean_fit = fitnesses.mean()
            print(f"  Iter {iteration+1:4d}/{n_iterations}  |  "
                  f"Best: {global_best_fit:.4f}  |  "
                  f"Mean: {mean_fit:.4f}  |  "
                  f"w: {current_w:.3f}")

        # --- 2f. Early stopping ------------------------------------------
        if no_improve_count >= patience:
            if verbose:
                print(f"\n  Early stopping at iteration {iteration+1} "
                      f"(no improvement for {patience} iterations).")
            # Pad history to expected length so plots align with GA
            history += [global_best_fit] * (n_iterations - len(history))
            break

    if verbose:
        print(f"\n  ✓ PSO finished.  Best fitness = {global_best_fit:.4f}  "
              f"({global_best_fit*100:.1f}% F1)\n")

    return global_best_pos, global_best_fit, history


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

    print("Running PSO (linear decay inertia, default hyperparameters) ...")
    best_theta, best_f1, hist = particle_swarm_optimisation(
        fitness_fn    = fitness_function,
        n_params      = n_params,
        fitness_args  = (X, y),
        n_particles   = 30,
        n_iterations  = 50,
        seed          = 42,
        verbose       = True,
    )

    metrics = evaluate_solution(best_theta, X, y)
    print(f"\nFull evaluation of best PSO solution:")
    print(f"  F1        : {metrics['f1']:.4f}  ← fitness signal")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Predicted positive : {metrics['n_pred_pos']}")
    print(f"  Predicted negative : {metrics['n_pred_neg']}")
    print(f"\nHistory (last 5 iters): {[round(h,4) for h in hist[-5:]]}")

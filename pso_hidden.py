"""
pso_hidden.py
=============
Particle Swarm Optimisation (PSO) for optimising real-valued neural
network weights — **two hidden layers** variant.

Changes from pso_c.py
----------------------
* The function signature gains `hidden1_size` and `hidden2_size` in place
  of the single `hidden_size` argument, to mirror the architecture used by
  `parkinsons_hidden.py`.
* The verbose header prints the updated architecture string
  (input → h1 → h2 → output).
* The PSO algorithm itself (velocity update, clamping, personal/global
  best tracking, early stopping) is **completely unchanged** — PSO
  operates on a flat real-valued vector and is agnostic to n_params.
* `n_params` is now 351 instead of 241; callers obtain it via
  `parkinsons_hidden.compute_n_params()`.

All other behaviour (inertia strategies, patience early-stop, return
signature) is identical to pso_c.py.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


def particle_swarm_optimisation(
    fitness_fn:       Callable,
    n_params:         int,
    fitness_args:     tuple         = (),
    n_particles:      int           = 50,
    n_iterations:     int           = 100,
    w:                float         = 0.9,
    w_min:            float         = 0.4,
    c1:               float         = 2.0,
    c2:               float         = 2.0,
    v_max_ratio:      float         = 0.2,
    low:              float         = -1.0,
    high:             float         = 1.0,
    inertia_strategy: str           = 'linear_decay',
    patience:         int           = 20,
    tol:              float         = 1e-6,
    # --- CHANGED: architecture now has two hidden layers ---
    input_size:       int           = 22,
    hidden1_size:     int           = 10,   # first  hidden layer
    hidden2_size:     int           = 10,   # second hidden layer  ← NEW
    output_size:      int           = 1,
    seed:             Optional[int] = None,
    verbose:          bool          = True,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    PSO main loop — two-hidden-layer variant.

    Parameters
    ----------
    (All parameters identical to pso_c.py except the architecture args.)

    hidden1_size : int  — neurons in the first  hidden layer (default 10)
    hidden2_size : int  — neurons in the second hidden layer (default 10) ← NEW

    Returns
    -------
    best_position : np.ndarray, shape (n_params,)
    best_fitness  : float
    history       : List[float]  — global best fitness per iteration
    """
    if seed is not None:
        np.random.seed(seed)

    v_max = (high - low) * v_max_ratio

    # ------------------------------------------------------------------
    # 1. INITIALISATION  (unchanged — architecture is transparent here)
    # ------------------------------------------------------------------
    positions  = np.random.uniform(low, high, size=(n_particles, n_params))
    velocities = np.random.uniform(-v_max, v_max, size=(n_particles, n_params))

    fitnesses  = np.array([fitness_fn(positions[i], *fitness_args)
                           for i in range(n_particles)])

    personal_best_pos = positions.copy()
    personal_best_fit = fitnesses.copy()

    global_best_idx   = np.argmax(fitnesses)
    global_best_pos   = positions[global_best_idx].copy()
    global_best_fit   = fitnesses[global_best_idx]

    history           = []
    no_improve_count  = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Particle Swarm Optimisation  [2 hidden layers]")
        # CHANGED: display both hidden layer sizes
        print(f"  arch: {input_size}→{hidden1_size}→{hidden2_size}→{output_size}"
              f"  |  n_params={n_params}")
        print(f"  particles={n_particles}  iters={n_iterations}  "
              f"inertia={inertia_strategy}")
        print(f"  w={w}→{w_min}  c1={c1}  c2={c2}  v_max={v_max:.3f}")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 2. MAIN LOOP  (structurally unchanged from pso_c.py)
    # ------------------------------------------------------------------
    for iteration in range(n_iterations):

        # --- 2a. Inertia update ---
        if inertia_strategy == 'linear_decay':
            current_w = w - (w - w_min) * (iteration / max(n_iterations - 1, 1))
        else:
            current_w = w

        # --- 2b. Velocity + position update (vectorised) ---
        r1 = np.random.rand(n_particles, n_params)
        r2 = np.random.rand(n_particles, n_params)

        cognitive  = c1 * r1 * (personal_best_pos - positions)
        social     = c2 * r2 * (global_best_pos   - positions)

        velocities = current_w * velocities + cognitive + social
        velocities = np.clip(velocities, -v_max, v_max)
        positions  = positions + velocities

        # --- 2c. Evaluate ---
        fitnesses = np.array([fitness_fn(positions[i], *fitness_args)
                               for i in range(n_particles)])

        # --- 2d. Personal bests ---
        improved_mask = fitnesses > personal_best_fit
        personal_best_pos[improved_mask] = positions[improved_mask].copy()
        personal_best_fit[improved_mask] = fitnesses[improved_mask]

        # --- 2e. Global best ---
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
            print(f"  Iter {iteration+1:4d}/{n_iterations}  |  "
                  f"Best: {global_best_fit:.4f}  |  "
                  f"Mean: {fitnesses.mean():.4f}  |  "
                  f"w: {current_w:.3f}")

        # --- 2f. Early stopping ---
        if no_improve_count >= patience:
            if verbose:
                print(f"\n  Early stopping at iteration {iteration+1} "
                      f"(no improvement for {patience} iterations).")
            history += [global_best_fit] * (n_iterations - len(history))
            break

    if verbose:
        print(f"\n  ✓ PSO finished.  Best fitness = {global_best_fit:.4f}"
              f"  ({global_best_fit*100:.1f}% recall)\n")

    return global_best_pos, global_best_fit, history


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import pandas as pd
    from parkinsons_hidden import fitness_function, compute_n_params, evaluate_solution

    df       = pd.read_csv('parkinsons_preprocessed.csv')
    X        = df.drop(columns=['status']).values
    y        = df['status'].values
    n_params = compute_n_params()

    print(f"n_params = {n_params}")
    best_theta, best_recall, hist = particle_swarm_optimisation(
        fitness_fn    = fitness_function,
        n_params      = n_params,
        fitness_args  = (X, y),
        n_particles   = 20,
        n_iterations  = 30,
        seed          = 42,
        verbose       = True,
    )

    metrics = evaluate_solution(best_theta, X, y)
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")

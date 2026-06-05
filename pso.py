import numpy as np
import random
import pandas as pd


def pso(
    fitness_func,
    n_particles,
    n_params,
    n_iterations,
    fitness_args=(),
    w=None,
    c1=None,
    c2=None,
    low=-1.0,
    high=1.0,
    seed=None
):
    """
    Particle Swarm Optimisation (PSO) — maximisation variant.

    Each particle maintains a position in the search space and a velocity
    that is updated every iteration according to three influences:

      Inertia        w  * v(t)                  — keeps the particle moving
                                                   in its current direction.
      Cognitive term c1 * r1 * (p_best − x(t))  — attraction toward the
                                                   particle's own best position.
      Social term    c2 * r2 * (g_best − x(t))  — attraction toward the
                                                   swarm's global best position.

    Velocity update:
        v(t+1) = w * v(t)
               + c1 * r1 * (p_best − x(t))
               + c2 * r2 * (g_best − x(t))
        x(t+1) = x(t) + v(t+1)

    Positions are clipped to [low, high] after each update; velocities are
    initialised uniformly in [-(high−low), +(high−low)].

    Parameters
    ----------
    fitness_func : callable
        Objective function f(position, *fitness_args) → float.
        Higher values are treated as better (maximisation).
    n_particles : int
        Number of particles in the swarm.
    n_params : int
        Dimensionality of the search space (= number of MLP weights).
    n_iterations : int
        Number of update cycles to run.
    fitness_args : tuple, optional
        Extra positional arguments forwarded to `fitness_func`.
    w : float
        Inertia weight — balances global exploration (high w) against local
        exploitation (low w).  Typical static value: 0.4 – 0.9.
    c1 : float
        Cognitive acceleration coefficient.  Scales attraction toward each
        particle's personal best.  Typical value: 1.5 – 2.0.
    c2 : float
        Social acceleration coefficient.  Scales attraction toward the global
        best.  Typical value: 1.5 – 2.0.
    low : float, optional (default=-1.0)
        Lower bound of the search space, applied to every dimension.
    high : float, optional (default=1.0)
        Upper bound of the search space, applied to every dimension.
    seed : int or None, optional (default=None)
        NumPy random seed for reproducibility.

    Returns
    -------
    g_best : np.ndarray, shape (n_params,)
        Position (weight vector) with the highest fitness found.
    g_best_fit : float
        Fitness value of `g_best`.
    history : list of float
        Global best fitness recorded at the end of each iteration
        (length = n_iterations), useful for convergence plots.
    """

    if seed is not None:
        np.random.seed(seed)

    positions = []
    velocities = []
    p_bests = []
    p_bests_fit = []

    for particle in range(n_particles):
        position    = np.random.uniform(low, high, n_params)
        p_best      = position.copy()
        p_best_fit  = fitness_func(p_best)
        velocity    = np.random.uniform(-abs(high - low), abs(high - low), n_params)

        positions.append(position)
        velocities.append(velocity)
        p_bests.append(p_best)
        p_bests_fit.append(p_best_fit)

    positions    = np.array(positions)
    velocities   = np.array(velocities)
    p_bests      = np.array(p_bests)
    p_bests_fit  = np.array(p_bests_fit)

    best_idx = 0

    for i in range(1, len(p_bests_fit)):
        if p_bests_fit[i] > p_bests_fit[best_idx]:
            best_idx = i

    g_best     = p_bests[best_idx].copy()
    g_best_fit = p_bests_fit[best_idx]

    print("Initial global best fitness:", g_best_fit)

    history = []

    for iteration in range(n_iterations):
        for particle in range(n_particles):
            r1 = np.random.uniform(0, 1, n_params)
            r2 = np.random.uniform(0, 1, n_params)

            velocities[particle] = (
                w  * velocities[particle]
                + c1 * r1 * (p_bests[particle] - positions[particle])
                + c2 * r2 * (g_best            - positions[particle])
            )

            positions[particle] = positions[particle] + velocities[particle]

            for j in range(n_params):
                if positions[particle][j] < low:
                    positions[particle][j] = low
                elif positions[particle][j] > high:
                    positions[particle][j] = high

            current_fit = fitness_func(positions[particle], *fitness_args)

            if current_fit > p_bests_fit[particle]:
                p_bests[particle]     = positions[particle].copy()
                p_bests_fit[particle] = current_fit

            if current_fit > g_best_fit:
                g_best     = positions[particle].copy()
                g_best_fit = current_fit

        history.append(g_best_fit)

        print("Iteration", iteration + 1, "Best fitness:", round(g_best_fit, 4))

    return g_best, g_best_fit, history


results_pso = pd.read_csv("pso_gridsearch_full.csv")
top5_pso    = results_pso.sort_values(by='mean_fitness', ascending=False).head(5)
top5_dict   = top5_pso.to_dict(orient='records')
top5_dict

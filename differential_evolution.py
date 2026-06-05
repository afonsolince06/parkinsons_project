import numpy as np
import random
import pandas as pd
import time
from parkinsons_problem import fitness_function, compute_n_params


def differential_evolution(
    fitness_func,
    n_params,
    pop_size=None,
    generations=None,
    bounds=(-1.0, 1.0),
    F=None,
    CR=None,
    maximization=True,
    seed=None,
):
    """
    Differential Evolution — DE/rand/1/bin (maximisation variant).

    Implements the canonical DE loop with three operators applied in
    sequence for every individual in the population each generation:

      Mutation  (rand/1)   mutant  = x_a + F * (x_b − x_c)
                           where a, b, c are distinct indices ≠ i.
      Crossover (binomial) trial[j] = mutant[j]  if U(0,1) < CR or j == j_rand
                                      target[j]  otherwise.
                           j_rand guarantees at least one gene from the mutant.
      Selection (greedy)   trial replaces target only when
                           fitness(trial) ≥ fitness(target).

    The population is initialised uniformly at random within `bounds` and
    all mutant vectors are clipped back into bounds after mutation.

    Parameters
    ----------
    fitness_func : callable
        Objective function f(individual) → float.
        The function must accept a 1-D numpy array of length `n_params`.
    n_params : int
        Number of dimensions / MLP weight parameters to optimise.
    pop_size : int
        Population size.  Must be ≥ 4 (so that three distinct donors
        a, b, c can always be sampled).  Typical rule of thumb: 10×n_params.
    generations : int
        Number of generations (outer loop iterations).
    bounds : tuple (low, high), optional (default=(-1.0, 1.0))
        Search-space boundaries applied uniformly to all dimensions.
        A list of per-dimension tuples [(l1,h1), …, (ln,hn)] is also accepted.
    F : float
        Mutation scaling factor, controls the step size of the differential
        perturbation.  Typical range: [0.4, 1.0].
    CR : float
        Crossover rate, probability that a given gene comes from the mutant
        rather than the target.  Typical range: [0.5, 1.0].
    maximization : bool, optional (default=True)
        Direction of optimisation.  True → maximise (e.g. F1, accuracy);
        False → minimise (e.g. MSE).  Only the maximisation branch is active
        in this implementation (the minimisation guard is omitted).
    seed : int or None, optional (default=None)
        Random seed for both numpy and Python's random module.

    Returns
    -------
    best_individual : np.ndarray, shape (n_params,)
        Weight vector that achieved the highest fitness across all generations.
    best_fitness : float
        Fitness value of `best_individual`.
    history : list of float
        Best fitness at the end of each generation (length = generations + 1,
        including the initial evaluation), suitable for convergence plots.
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if isinstance(bounds, tuple) and len(bounds) == 2 and not isinstance(bounds[0], (list, tuple, np.ndarray)):
        low  = np.full(n_params, bounds[0], dtype=float)
        high = np.full(n_params, bounds[1], dtype=float)
    else:
        bounds = list(bounds)
        low  = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)

    population = np.random.uniform(low, high, size=(pop_size, n_params))

    fitness = np.array([fitness_func(ind) for ind in population])

    best_idx = np.argmax(fitness)

    best_individual = population[best_idx].copy()
    best_fitness    = fitness[best_idx]
    history         = [best_fitness]

    for gen in range(generations):
        for i in range(pop_size):
            candidates = list(range(pop_size))
            candidates.remove(i)
            a_idx, b_idx, c_idx = random.sample(candidates, 3)

            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            mutant = a + F * (b - c)

            mutant = np.clip(mutant, low, high)

            j_rand    = random.randint(0, n_params - 1)
            cross_mask = (np.random.uniform(0, 1, n_params) < CR)
            cross_mask[j_rand] = True

            trial = np.where(cross_mask, mutant, population[i])
            trial_fitness = fitness_func(trial)

            if trial_fitness >= fitness[i]:
                population[i] = trial
                fitness[i]    = trial_fitness

            if trial_fitness > best_fitness:
                best_fitness    = trial_fitness
                best_individual = trial.copy()

        history.append(best_fitness)

        print(f"  [DE] Generation {gen + 1:>4}/{generations} | Best fitness: {best_fitness:.6f}")

    return best_individual, best_fitness, history

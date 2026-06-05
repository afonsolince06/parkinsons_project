import numpy as np
import random
import pandas as pd
import time
from my_parkinsons_problem import fitness_function, compute_n_params
def differential_evolution(
    fitness_func,
    n_params,
    pop_size= None ,
    generations= None ,
    bounds=(-1.0, 1.0),
    F= None,
    CR=None,
    maximization=True,
    seed=None,
):

   
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


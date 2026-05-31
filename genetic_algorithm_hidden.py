"""
genetic_algorithm_hidden.py
============================
Genetic Algorithm (GA) for optimising real-valued neural network weights
— **two hidden layers** variant.

Changes from genetic_algorithm_c.py
-------------------------------------
* Default `input_size`, `hidden_size`, `output_size` parameters in
  `genetic_algorithm()` are replaced by `input_size`, `hidden1_size`,
  `hidden2_size`, `output_size` to match the new architecture.
* `xavier_initialisation()` now scales three layers instead of two:
    Layer 1  (input   → hidden-1)  limit1 = √(6 / (input + h1))
    Layer 2  (hidden-1 → hidden-2) limit2 = √(6 / (h1    + h2))   ← NEW
    Layer 3  (hidden-2 → output)   limit3 = √(6 / (h2    + out))
* All selection / crossover / mutation operators are **unchanged** —
  they operate on the flat weight vector regardless of its length.
* `n_params` is now 351 instead of 241; callers obtain it from
  `parkinsons_hidden.compute_n_params()`.

Everything else (population structure, elitism, verbose output,
return signature) is identical to the original module so the rest of the
project can use it as a drop-in replacement.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


# =====================================================
# 1. POPULATION INITIALISATION


def random_initialisation(pop_size: int, n_params: int) -> np.ndarray:
    """Uniform random in [-1, 1].  Unchanged from original."""
    return np.random.uniform(-1.0, 1.0, size=(pop_size, n_params))


def xavier_initialisation(
    pop_size:     int,
    n_params:     int,
    input_size:   int,
    hidden1_size: int,
    hidden2_size: int,   # NEW — second hidden layer
    output_size:  int,
) -> np.ndarray:
    """
    Xavier initialisation for a **three-layer** network.

    CHANGED: Three limit values are computed — one per weight matrix.

        limit_l = sqrt(6 / (fan_in_l + fan_out_l))

    Biases are initialised to zero (standard practice).
    """
    population = np.zeros((pop_size, n_params))
    idx = 0

    # --- Layer 1: W1 (input → hidden-1) + b1 ---
    limit1 = np.sqrt(6.0 / (input_size + hidden1_size))
    n_W1   = input_size * hidden1_size
    n_b1   = hidden1_size
    population[:, idx: idx + n_W1]       = np.random.uniform(-limit1, limit1, (pop_size, n_W1))
    idx += n_W1
    population[:, idx: idx + n_b1]       = 0.0   # biases → 0
    idx += n_b1

    # --- Layer 2: W2 (hidden-1 → hidden-2) + b2  (NEW) ---
    limit2 = np.sqrt(6.0 / (hidden1_size + hidden2_size))
    n_W2   = hidden1_size * hidden2_size
    n_b2   = hidden2_size
    population[:, idx: idx + n_W2]       = np.random.uniform(-limit2, limit2, (pop_size, n_W2))
    idx += n_W2
    population[:, idx: idx + n_b2]       = 0.0
    idx += n_b2

    # --- Layer 3: W3 (hidden-2 → output) + b3 ---
    limit3 = np.sqrt(6.0 / (hidden2_size + output_size))
    n_W3   = hidden2_size * output_size
    n_b3   = output_size
    population[:, idx: idx + n_W3]       = np.random.uniform(-limit3, limit3, (pop_size, n_W3))
    idx += n_W3
    population[:, idx: idx + n_b3]       = 0.0
    idx += n_b3

    assert idx == n_params, f"Xavier init mismatch: {idx} vs {n_params}"
    return population


# =====================================================
# 2. SELECTION  (unchanged — operate on arbitrary-length vectors)


def tournament_selection(population, fitnesses, tournament_size=3):
    """Pick the best individual from a random subset of size k."""
    pop_size = len(population)
    indices  = np.random.choice(pop_size, size=tournament_size, replace=False)
    winner   = indices[np.argmax(fitnesses[indices])]
    return population[winner].copy()


def roulette_wheel_selection(population, fitnesses):
    """Fitness-proportionate (roulette wheel) selection."""
    adjusted = fitnesses - fitnesses.min() + 1e-8
    probs    = adjusted / adjusted.sum()
    idx      = np.random.choice(len(population), p=probs)
    return population[idx].copy()


# =====================================================
# 3. CROSSOVER  (unchanged)


def arithmetic_crossover(parent1, parent2, alpha=None):
    """Convex combination of two parents.  α ~ U[0,1] if not given."""
    if alpha is None:
        alpha = np.random.uniform(0.0, 1.0)
    child1 = alpha * parent1 + (1.0 - alpha) * parent2
    child2 = (1.0 - alpha) * parent1 + alpha * parent2
    return child1, child2


def blend_crossover(parent1, parent2, alpha=0.5):
    """BLX-α: may extrapolate beyond the parental gene range."""
    d      = np.abs(parent1 - parent2)
    lo     = np.minimum(parent1, parent2) - alpha * d
    hi     = np.maximum(parent1, parent2) + alpha * d
    child1 = np.random.uniform(lo, hi)
    child2 = np.random.uniform(lo, hi)
    return child1, child2


# =====================================================
# 4. MUTATION  (unchanged)


def gaussian_mutation(individual, mutation_rate=0.1, sigma=0.1):
    """Additive Gaussian noise on each gene with probability mutation_rate."""
    mutant = individual.copy()
    mask   = np.random.rand(len(mutant)) < mutation_rate
    mutant[mask] += np.random.normal(0, sigma, size=mask.sum())
    return mutant


def uniform_reset_mutation(individual, mutation_rate=0.1,
                           low=-1.0, high=1.0):
    """Reset random genes to a fresh uniform draw."""
    mutant = individual.copy()
    mask   = np.random.rand(len(mutant)) < mutation_rate
    mutant[mask] = np.random.uniform(low, high, size=mask.sum())
    return mutant


# =====================================================
# 5. MAIN GA LOOP


def genetic_algorithm(
    fitness_fn:       Callable,
    n_params:         int,
    fitness_args:     tuple         = (),
    pop_size:         int           = 50,
    n_generations:    int           = 100,
    crossover_rate:   float         = 0.8,
    mutation_rate:    float         = 0.1,
    sigma:            float         = 0.1,
    tournament_size:  int           = 3,
    elitism:          int           = 2,
    selection_method: str           = 'tournament',
    crossover_method: str           = 'arithmetic',
    mutation_method:  str           = 'gaussian',
    init_method:      str           = 'random',
    # --- CHANGED: three size arguments instead of two ---
    input_size:       int           = 22,
    hidden1_size:     int           = 10,   # first  hidden layer
    hidden2_size:     int           = 10,   # second hidden layer  ← NEW
    output_size:      int           = 1,
    seed:             Optional[int] = None,
    verbose:          bool          = True,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Genetic Algorithm main loop — two-hidden-layer variant.

    CHANGED from genetic_algorithm_c.py:
      * Xavier init now takes `hidden1_size` and `hidden2_size`.
      * Verbose header prints the updated architecture string.
      * Everything else is structurally identical.

    Returns
    -------
    best_solution : np.ndarray, shape (n_params,)
    best_fitness  : float
    history       : List[float]  — best fitness per generation
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Initialise population ---
    if init_method == 'xavier':
        population = xavier_initialisation(
            pop_size, n_params,
            input_size, hidden1_size, hidden2_size, output_size   # ← updated
        )
    else:
        population = random_initialisation(pop_size, n_params)

    best_solution = None
    best_fitness  = -np.inf
    history       = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Genetic Algorithm  [2 hidden layers]")
        print(f"  arch: {input_size}→{hidden1_size}→{hidden2_size}→{output_size}"
              f"  |  n_params={n_params}")
        print(f"  pop={pop_size}  gens={n_generations}  "
              f"cx={crossover_method}  mut={mutation_method}")
        print(f"  sel={selection_method}  init={init_method}  "
              f"elitism={elitism}")
        print(f"{'='*60}")

    for gen in range(n_generations):

        # 1. Evaluate
        fitnesses = np.array([
            fitness_fn(ind, *fitness_args) for ind in population
        ])

        # 2. Track global best
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness  = fitnesses[gen_best_idx]
            best_solution = population[gen_best_idx].copy()

        history.append(best_fitness)

        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            print(f"  Gen {gen+1:4d}/{n_generations}  |  "
                  f"Best: {best_fitness:.4f}  |  Mean: {fitnesses.mean():.4f}")

        # 3. Elitism
        elite_indices = np.argsort(fitnesses)[::-1][:elitism]
        next_gen      = [population[i].copy() for i in elite_indices]

        # 4. Selection + Crossover + Mutation
        while len(next_gen) < pop_size:
            if selection_method == 'roulette':
                p1 = roulette_wheel_selection(population, fitnesses)
                p2 = roulette_wheel_selection(population, fitnesses)
            else:
                p1 = tournament_selection(population, fitnesses, tournament_size)
                p2 = tournament_selection(population, fitnesses, tournament_size)

            if np.random.rand() < crossover_rate:
                c1, c2 = (blend_crossover(p1, p2) if crossover_method == 'blend'
                          else arithmetic_crossover(p1, p2))
            else:
                c1, c2 = p1.copy(), p2.copy()

            if mutation_method == 'uniform':
                c1 = uniform_reset_mutation(c1, mutation_rate)
                c2 = uniform_reset_mutation(c2, mutation_rate)
            else:
                c1 = gaussian_mutation(c1, mutation_rate, sigma)
                c2 = gaussian_mutation(c2, mutation_rate, sigma)

            next_gen.append(c1)
            if len(next_gen) < pop_size:
                next_gen.append(c2)

        population = np.array(next_gen)

    if verbose:
        print(f"\n  ✓ GA finished.  Best fitness = {best_fitness:.4f}"
              f"  ({best_fitness*100:.1f}% recall)\n")

    return best_solution, best_fitness, history


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import pandas as pd
    from parkinsons_hidden import fitness_function, compute_n_params

    df = pd.read_csv('parkinsons_preprocessed.csv')
    X  = df.drop(columns=['status']).values
    y  = df['status'].values

    n_params = compute_n_params()
    print(f"n_params = {n_params}")

    best_theta, best_recall, hist = genetic_algorithm(
        fitness_fn    = fitness_function,
        n_params      = n_params,
        fitness_args  = (X, y),
        pop_size      = 20,
        n_generations = 30,
        seed          = 42,
        verbose       = True,
    )
    print(f"Best recall: {best_recall*100:.2f}%")

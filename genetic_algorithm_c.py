"""
genetic_algorithm.py
====================
Genetic Algorithm (GA) for optimising real-valued neural network weights.

This module implements a complete, modular GA with:

  Initialisation  (2 methods)
  ├── random_initialisation      — uniform draw in [low, high]
  └── xavier_initialisation      — layer-aware scaled initialisation

  Selection       (2 operators)
  ├── tournament_selection        — pick best of k random individuals
  └── roulette_wheel_selection    — fitness-proportionate sampling

  Crossover       (2 operators)
  ├── arithmetic_crossover        — convex combination of two parents
  └── blend_crossover (BLX-α)     — expand search beyond parent range

  Mutation        (2 operators)
  ├── gaussian_mutation           — add N(0,σ) noise to each gene
  └── uniform_reset_mutation      — randomly reset a fraction of genes

  Main loop
  └── genetic_algorithm(...)      — ties everything together; returns
                                    best solution, best fitness, and the
                                    full convergence history

"""

import numpy as np
from typing import Callable, List, Tuple, Optional


# =====================================================
# 1. POPULATION INITIALISATION


def random_initialisation(pop_size, n_params): #np.ndarray
    """
    Method 1: Uniform random initialisation

    Each gene of each individual is drawn independently from Uniform[low, high].
    This is the simplest and most common GA starting point: it covers the
    search space broadly with no assumptions about where good solutions lie.
    """
    return np.random.uniform(-1.0, 1.0, size=(pop_size, n_params))


def xavier_initialisation(pop_size,
                           n_params,
                           input_size,
                           hidden_size,
                           output_size): #np.ndarray
    """
    Method 2: Xavier

    Instead of using the same uniform range for all weights, this method
    scales the range differently for each layer:

        limit_l = sqrt(6 / (fan_in + fan_out))

    where fan_in and fan_out are the number of neurons feeding *into* and
    *out of* a layer respectively.  This is the initialisation scheme used
    in deep learning practice.

    Why this matters for optimization:
      A small, well-scaled initial population is closer to the region of
      good solutions, so the GA needs fewer generations to converge.  The
      flat random init covers the space broadly, but can take more time to finf the good regions; Xavier gives a smarter
      starting distribution, it is effective and does good exploration.
    """
    population = np.zeros((pop_size, n_params))

    # Layer 1: W1 (input→hidden) + b1
    limit1 = np.sqrt(6.0 / (input_size + hidden_size))
    n_W1   = input_size * hidden_size
    n_b1   = hidden_size
    population[:, :n_W1]              = np.random.uniform(-limit1, limit1, (pop_size, n_W1))
    population[:, n_W1:n_W1 + n_b1]  = np.zeros((pop_size, n_b1))   # biases start at 0

    # Layer 2: W2 (hidden→output) + b2
    limit2 = np.sqrt(6.0 / (hidden_size + output_size))
    n_W2   = hidden_size * output_size
    n_b2   = output_size
    offset = n_W1 + n_b1
    population[:, offset:offset + n_W2]              = np.random.uniform(-limit2, limit2, (pop_size, n_W2))
    population[:, offset + n_W2:offset + n_W2 + n_b2] = np.zeros((pop_size, n_b2))

    return population


# ====================================
# 2. SELECTION


def tournament_selection(population,
                          fitnesses,
                          tournament_size = 3): #np.ndarray
    """
    Selection Method 1: Tournament Selection

    Algorithm:
      1. Randomly pick `tournament_size` individuals from the population.
      2. Return the one with the highest fitness.

    This is repeated once per call, so to select a full mating pool of
    pop_size parents you call this function pop_size times.

    Properties:
      • Selection pressure is controlled by tournament_size:
          k=2  →  weak pressure, high diversity maintained
          k=5  →  stronger pressure, faster convergence (risk: premature)
      • Does not require fitness values to be positive (unlike roulette wheel).

    Returns the winning individual
    """
    pop_size = len(population)
    indices  = np.random.choice(pop_size, size=tournament_size, replace=False)
    winner   = indices[np.argmax(fitnesses[indices])]
    return population[winner].copy()


def roulette_wheel_selection(population,
                              fitnesses):
    """
    Selection Method 2: Roulette Wheel (Fitness-Proportionate) Selection.

    Each individual is assigned a selection probability proportional to its
    fitness.  A random spin of the wheel picks one individual.

    Because fitnesses can be in [0, 1] (accuracy) and may have a narrow
    spread, we shift them so the minimum fitness maps to a small positive
    value rather than zero.  This prevents the worst individual from having
    zero selection probability (which would be biologically unrealistic and
    reduce diversity).

    Shift formula:
        adjusted = fitness - min(fitness) + error

    Returns the selected individual
    """
    adjusted = fitnesses - fitnesses.min() + 1e-8   # shift to be strictly positive
    probs    = adjusted / adjusted.sum()             # normalise to a probability vector
    idx      = np.random.choice(len(population), p=probs)
    return population[idx].copy()


# =================================
# 3. CROSSOVER


def arithmetic_crossover(parent1,
                          parent2,
                          alpha = None): #Tuple[np.ndarray (child1), np.ndarray (child2)]:
    """
    Crossover Method 1: Arithmetic Crossover.

    Produces two children as weighted averages of the parents:
        child1 = α·parent1 + (1−α)·parent2
        child2 = (1−α)·parent1 + α·parent2

    When α is random (None), each call draws a new α ~ Uniform[0,1],
    so children smoothly interpolate between parents with varying blending.

    Why this works for real-valued weights:
        Arithmetic crossover creates intermediate weight configurations.
        This is appropriate here because good weights are real valued and
        the fitness landscape is continuous, a midpoint between two
        decent solutions is often also decent, which means the children
        should be good if their parents are good solutions.

    Returns a tuple (child1, child2)
    """
    if alpha is None:
        alpha = np.random.uniform(0.0, 1.0)
    child1 = alpha * parent1 + (1.0 - alpha) * parent2
    child2 = (1.0 - alpha) * parent1 + alpha * parent2
    return child1, child2


def blend_crossover(parent1,
                    parent2,
                    alpha= 0.5):
    """
    Crossover Method 2: BLX-α (Blend Crossover).

    For each gene position i, the child's gene is drawn uniformly from
    an *extended* interval:

        [min(p1_i, p2_i) − α·d_i,   max(p1_i, p2_i) + α·d_i]

    where d_i = |p1_i − p2_i| is the gene-wise distance between parents.

    The α parameter controls how far outside the parental range children
    may fall:
        α = 0.0  →  children stay within [min, max] of parents (no exploration)
        α = 0.5  →  (standard BLX-0.5) extends range by 50% on each side
        α > 0.5  →  more exploration, but risk of drifting into bad regions

    Why BLX-α is useful:
      Arithmetic crossover only interpolates (stays between parents).
      BLX-α also *extrapolates*, which helps the GA escape local optima by
      producing children with gene values never seen in either parent.

    Returns a tuple (child1, child2)
    """
    d      = np.abs(parent1 - parent2)          # gene-wise distance
    lo     = np.minimum(parent1, parent2) - alpha * d   # extended lower bound
    hi     = np.maximum(parent1, parent2) + alpha * d   # extended upper bound
    child1 = np.random.uniform(lo, hi)
    child2 = np.random.uniform(lo, hi)
    return child1, child2


# ===============================
# 4. MUTATION


def gaussian_mutation(individual,
                       mutation_rate = 0.1,
                       sigma  = 0.1): #np.ndarray
    """
    Mutation Method 1: Gaussian Mutation.

    Each gene is mutated independently with probability `mutation_rate`.
    Mutated genes receive additive Gaussian noise:  gene += N(0, sigma²).

    This is the standard mutation for real valued GAs.  It makes small
    perturbations to the weights, analogous to a local random walk.
    """
    mutant = individual.copy()
    mask   = np.random.rand(len(mutant)) < mutation_rate
    mutant[mask] += np.random.normal(0, sigma, size=mask.sum())
    return mutant


def uniform_reset_mutation(individual,
                            mutation_rate = 0.1,
                            low = -1.0,
                            high = 1.0):
    """
    Mutation Method 2: Uniform Reset Mutation.

    Each gene is reset independently with probability `mutation_rate`.
    Reset genes are drawn fresh from Uniform[low, high], ignoring the
    current value entirely.

    Contrast with Gaussian mutation:
      • Gaussian mutation makes small steps, good for exploitation
      • Uniform reset makes large jumps, good for exploration and escaping
        local optimal when the GA has stagnated

    Using both in different phases (or with different probabilities) can
    balance exploration vs exploitation.
    """
    mutant = individual.copy()
    mask   = np.random.rand(len(mutant)) < mutation_rate
    mutant[mask] = np.random.uniform(low, high, size=mask.sum())
    return mutant


# =====================================================
# 5. MAIN GENETIC ALGORITHM LOOP

def genetic_algorithm(
    fitness_fn:         Callable,
    n_params:           int,
    fitness_args:       tuple         = (),
    pop_size:           int           = 50,
    n_generations:      int           = 100,
    crossover_rate:     float         = 0.8,
    mutation_rate:      float         = 0.1,
    sigma:              float         = 0.1,
    tournament_size:    int           = 3,
    elitism:            int           = 2,
    selection_method:   str           = 'tournament',
    crossover_method:   str           = 'arithmetic',
    mutation_method:    str           = 'gaussian',
    init_method:        str           = 'random',
    input_size:         int           = 22,
    hidden_size:        int           = 10,
    output_size:        int           = 1,
    seed:               Optional[int] = None,
    verbose:            bool          = True,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Main Genetic Algorithm loop.

    The GA maintains a fixed-size population of candidate weight vectors.
    Each generation it:
      1. Evaluates every individual via fitness_fn.
      2. Copies the top `elitism` individuals unchanged (elitism).
      3. Fills the rest of the next generation by:
         a. Selecting two parents via the chosen selection method.
         b. Applying crossover with probability `crossover_rate`.
         c. Applying mutation to both children.
      4. Records the best fitness seen so far.

    Parameters
    ----------
    fitness_fn       : Callable — f(solution, *fitness_args) → float
    n_params         : int      — length of weight vector θ
    fitness_args     : tuple    — extra arguments forwarded to fitness_fn
                                  e.g. (X, y) for the Parkinson's problem
    pop_size         : int      — population size               (default 50)
    n_generations    : int      — number of generations         (default 100)
    crossover_rate   : float    — probability of crossover      (default 0.8)
    mutation_rate    : float    — per-gene mutation probability  (default 0.1)
    sigma            : float    — std-dev for Gaussian mutation  (default 0.1)
    tournament_size  : int      — contestants in tournament sel. (default 3)
    elitism          : int      — # of top individuals kept each gen (default 2)
    selection_method : str      — 'tournament' | 'roulette'
    crossover_method : str      — 'arithmetic' | 'blend'
    mutation_method  : str      — 'gaussian'   | 'uniform'
    init_method      : str      — 'random'     | 'xavier'
    input_size       : int      — needed only when init_method='xavier'
    hidden_size      : int      — needed only when init_method='xavier'
    output_size      : int      — needed only when init_method='xavier'
    seed             : int|None — RNG seed for reproducibility
    verbose          : bool     — print progress every 10 generations

    Returns
    -------
    best_solution : np.ndarray, shape (n_params,) — best weight vector found
    best_fitness  : float                          — fitness of best_solution
    history       : List[float]                    — best fitness per generation
    """
    if seed is not None:
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # 0. Initialise population
    # ------------------------------------------------------------------
    if init_method == 'xavier':
        population = xavier_initialisation(pop_size, n_params,
                                           input_size, hidden_size, output_size)
    else:  # 'random'
        population = random_initialisation(pop_size, n_params)

    best_solution = None
    best_fitness  = -np.inf
    history       = []

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Genetic Algorithm")
        print(f"  pop={pop_size}  gens={n_generations}  "
              f"cx={crossover_method}  mut={mutation_method}")
        print(f"  sel={selection_method}  init={init_method}  "
              f"elitism={elitism}")
        print(f"{'='*55}")

    # ------------------------------------------------------------------
    # Main generational loop
    # ------------------------------------------------------------------
    for gen in range(n_generations):

        # 1. Evaluate every individual in the population
        fitnesses = np.array([
            fitness_fn(ind, *fitness_args) for ind in population
        ])

        # 2. Track the global best across all generations
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness  = fitnesses[gen_best_idx]
            best_solution = population[gen_best_idx].copy()

        history.append(best_fitness)

        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            mean_fit = fitnesses.mean()
            print(f"  Gen {gen+1:4d}/{n_generations}  |  "
                  f"Best: {best_fitness:.4f}  |  Mean: {mean_fit:.4f}")

        # 3. Build next generation
        # --- 3a. Elitism: carry forward the top `elitism` individuals ---
        elite_indices = np.argsort(fitnesses)[::-1][:elitism]
        next_gen      = [population[i].copy() for i in elite_indices]

        # --- 3b. Fill the rest via selection + crossover + mutation ---
        while len(next_gen) < pop_size:

            # Selection
            if selection_method == 'roulette':
                p1 = roulette_wheel_selection(population, fitnesses)
                p2 = roulette_wheel_selection(population, fitnesses)
            else:  # 'tournament' (default)
                p1 = tournament_selection(population, fitnesses, tournament_size)
                p2 = tournament_selection(population, fitnesses, tournament_size)

            # Crossover
            if np.random.rand() < crossover_rate:
                if crossover_method == 'blend':
                    c1, c2 = blend_crossover(p1, p2)
                else:  # 'arithmetic' (default)
                    c1, c2 = arithmetic_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            if mutation_method == 'uniform':
                c1 = uniform_reset_mutation(c1, mutation_rate)
                c2 = uniform_reset_mutation(c2, mutation_rate)
            else:  # 'gaussian' (default)
                c1 = gaussian_mutation(c1, mutation_rate, sigma)
                c2 = gaussian_mutation(c2, mutation_rate, sigma)

            next_gen.append(c1)
            if len(next_gen) < pop_size:
                next_gen.append(c2)

        population = np.array(next_gen)

    if verbose:
        print(f"\n  ✓ GA finished.  Best fitness = {best_fitness:.4f}  "
              f"({best_fitness*100:.1f}% recall)\n")

    return best_solution, best_fitness, history


# ===========================================================================
# Quick self-test
# ===========================================================================
if __name__ == '__main__':
    import pandas as pd
    from parkinsons_problem import fitness_function, compute_n_params

    df = pd.read_csv('parkinsons_preprocessed.csv')
    X  = df.drop(columns=['status']).values
    y  = df['status'].values

    n_params = compute_n_params()

    print("Running GA with default settings (arithmetic cx, gaussian mut, tournament sel) ...")
    best_theta, best_recall, hist = genetic_algorithm(
        fitness_fn    = fitness_function,
        n_params      = n_params,
        fitness_args  = (X, y),
        pop_size      = 30,
        n_generations = 50,
        seed          = 42,
        verbose       = True,
    )
    print(f"Best recall: {best_recall*100:.2f}%")
    print(f"History (last 5 gens): {[round(h,4) for h in hist[-5:]]}")

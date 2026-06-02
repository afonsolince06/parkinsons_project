"""
genetic_algorithm_c.py
======================
Genetic Algorithm for optimising real-valued neural network weights
- two hidden layers variant (22 → 10 → 10 → 1).

Basic Operators Implemented
---------------------------
  Selection:
    1. Tournament selection  (selection_method='tournament')
    2. Roulette-wheel selection (selection_method='roulette')

  Crossover:
    1. Arithmetic crossover  (crossover_method='arithmetic')
    2. Blend crossover (BLX-α)  (crossover_method='blend')

  Mutation:
    1. Gaussian mutation  (mutation_method='gaussian')
    2. Uniform mutation   (mutation_method='uniform')

  Initialisation:
    1. Xavier (Glorot) uniform initialisation (init_method='xavier')
    2. Random uniform in [-1, 1]              (init_method='random')

Usage
-----
    from genetic_algorithm_c import genetic_algorithm
    best_theta, best_fitness, history = genetic_algorithm(
        ...
    )
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


# ---------------------------------------------------------------------------
# SELECTION OPERATORS
# ---------------------------------------------------------------------------

def _tournament_selection(population, fitnesses, tournament_size):
    """
    Tournament selection: pick `tournament_size` individuals at random,
    return the fittest.

    Characteristics: high selection pressure, fast convergence.
    Risk: premature convergence when a few super-fit individuals dominate.
    """
    n = len(population)
    candidates = np.random.choice(n, size=tournament_size, replace=False)
    best = candidates[np.argmax(fitnesses[candidates])]
    return population[best].copy()


def _roulette_selection(population, fitnesses):
    """
    Fitness-proportionate (roulette) selection.
    Degenerate when all fitnesses are identical; shift to avoid negatives.
    """
    shifted = fitnesses - fitnesses.min() + 1e-8
    probs   = shifted / shifted.sum()
    idx     = np.random.choice(len(population), p=probs)
    return population[idx].copy()


# ---------------------------------------------------------------------------
# CROSSOVER OPERATORS
# ---------------------------------------------------------------------------

def _arithmetic_crossover(p1, p2, alpha=0.5):
    """
    Arithmetic (convex) crossover: child = α·p1 + (1-α)·p2.

    Produces intermediate weight values; good for continuous optimisation
    but does not respect the block structure of the weight vector.
    """
    a = np.random.uniform(0, 1)
    c1 = a * p1 + (1 - a) * p2
    c2 = a * p2 + (1 - a) * p1
    return c1, c2


def _blend_crossover(p1, p2, alpha=0.5):
    """
    Blend (BLX-α) crossover: sample uniformly from an extended interval
    [min - α·d, max + α·d] for each gene.  More exploratory than arithmetic.
    """
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    d  = hi - lo
    c1 = np.random.uniform(lo - alpha * d, hi + alpha * d)
    c2 = np.random.uniform(lo - alpha * d, hi + alpha * d)
    return c1, c2


# ---------------------------------------------------------------------------
# MUTATION OPERATORS
# ---------------------------------------------------------------------------

def _gaussian_mutation(individual, mutation_rate, sigma):
    """
    Gaussian mutation: perturb each gene independently with probability
    `mutation_rate` by adding N(0, σ) noise.
    Fixed σ throughout the run.
    """
    mask    = np.random.rand(len(individual)) < mutation_rate
    noise   = np.random.randn(len(individual)) * sigma
    mutant  = individual.copy()
    mutant[mask] += noise[mask]
    return mutant


def _uniform_mutation(individual, mutation_rate, low=-1.0, high=1.0):
    """
    Uniform mutation: replace selected genes with a uniform random value
    in [low, high].  More disruptive than Gaussian; useful for escaping
    deep local optima.
    """
    mask   = np.random.rand(len(individual)) < mutation_rate
    mutant = individual.copy()
    mutant[mask] = np.random.uniform(low, high, size=mask.sum())
    return mutant

# INITIALISATION STRATEGIES


def _xavier_init(n_params, input_size, hidden1_size, hidden2_size,
                 output_size):
    """
    Xavier (Glorot) uniform initialisation.

    Each weight matrix Wl is drawn from Uniform[-√(6/(fan_in+fan_out)),
    +√(6/(fan_in+fan_out))].  Biases are initialised to zero.

    Rationale: Xavier init keeps signal variance roughly constant across
    layers, reducing the risk of vanishing/exploding signals at start.
    For weight-optimisation this gives a better starting population than
    pure random uniform.
    """
    solution = np.zeros(n_params)
    idx = 0

    # W1: input → hidden1
    fan = input_size + hidden1_size
    lim = np.sqrt(6.0 / fan)
    nW1 = input_size * hidden1_size
    solution[idx:idx+nW1] = np.random.uniform(-lim, lim, nW1); idx += nW1
    solution[idx:idx+hidden1_size] = 0.0;                       idx += hidden1_size

    # W2: hidden1 → hidden2
    fan = hidden1_size + hidden2_size
    lim = np.sqrt(6.0 / fan)
    nW2 = hidden1_size * hidden2_size
    solution[idx:idx+nW2] = np.random.uniform(-lim, lim, nW2); idx += nW2
    solution[idx:idx+hidden2_size] = 0.0;                       idx += hidden2_size

    # W3: hidden2 → output
    fan = hidden2_size + output_size
    lim = np.sqrt(6.0 / fan)
    nW3 = hidden2_size * output_size
    solution[idx:idx+nW3] = np.random.uniform(-lim, lim, nW3); idx += nW3
    solution[idx:idx+output_size] = 0.0

    return solution


def _init_population(pop_size, n_params, method, input_size=22,
                      hidden1_size=10, hidden2_size=10, output_size=1):
    """Generate initial population using the specified strategy."""
    if method == 'xavier':
        return np.array([
            _xavier_init(n_params, input_size, hidden1_size,
                         hidden2_size, output_size)
            for _ in range(pop_size)
        ])
    else:   # 'random'
        return np.random.uniform(-1.0, 1.0, size=(pop_size, n_params))

# MAIN GENETIC ALGORITHM

def genetic_algorithm(
    fitness_fn,
    n_params,
    fitness_args = (),
    pop_size= 50,
    n_generations= 100,
    crossover_rate= 0.8,
    mutation_rate = 0.05,
    sigma= 0.1,
    tournament_size = 3,
    elitism  = 2,
    # selection_method options : 'tournament' , 'roulette'
    # crossover_method options : 'arithmetic' , 'blend'
    # mutation_method  options : 'gaussian' , 'uniform'
    # init_method      options : 'random' , 'xavier'
    selection_method = 'tournament',
    crossover_method = 'arithmetic',
    mutation_method = 'gaussian',
    init_method  = 'random',
    # used by Xavier init
    input_size = 22,
    hidden1_size = 10,
    hidden2_size = 10,
    output_size = 1,
    # ── Misc ──────────────────────────────────────────────────────────────
    seed:             Optional[int] = None,
    verbose:          bool          = True,
):
    """
    Genetic Algorithm main loop — two-hidden-layer variant with basic operators.

    Returns
    -------
    best_individual : np.ndarray, shape (n_params,)
    best_fitness    : float
    history         : List[float] — best fitness per generation
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Operator name normalisation ---
    sel_m = selection_method.lower().strip()
    cx_m  = crossover_method.lower().strip()
    mut_m = mutation_method.lower().strip()

    # --- Initialisation ---
    population = _init_population(
        pop_size, n_params, init_method,
        input_size, hidden1_size, hidden2_size, output_size
    )
    fitnesses  = np.array([fitness_fn(ind, *fitness_args)
                           for ind in population])

    best_idx      = int(np.argmax(fitnesses))
    best_ind      = population[best_idx].copy()
    best_fit      = fitnesses[best_idx]
    history       = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Genetic Algorithm  [2 hidden layers]")
        print(f"  arch: {input_size}→{hidden1_size}→{hidden2_size}→{output_size}"
              f"  |  n_params={n_params}")
        print(f"  pop={pop_size}  gens={n_generations}  elitism={elitism}")
        print(f"  selection={sel_m}  crossover={cx_m}  mutation={mut_m}"
              f"  init={init_method}")
        print(f"  mut_rate={mutation_rate}  σ={sigma}  cx_rate={crossover_rate}")
        print(f"{'='*60}")

    # ---  Main loop ---
    for gen in range(n_generations):

        new_population = []

        # --- Elitism: carry over top-k individuals unchanged ---
        if elitism > 0:
            elite_idx = np.argsort(fitnesses)[::-1][:elitism]
            for ei in elite_idx:
                new_population.append(population[ei].copy())

        # --- Fill remaining slots ---
        while len(new_population) < pop_size:

            # ── SELECTION ────────────────────────────────────────────────
            if sel_m == 'tournament':
                p1 = _tournament_selection(population, fitnesses, tournament_size)
                p2 = _tournament_selection(population, fitnesses, tournament_size)
            elif sel_m == 'roulette':
                p1 = _roulette_selection(population, fitnesses)
                p2 = _roulette_selection(population, fitnesses)
            else:
                raise ValueError(
                    f"Unknown selection_method '{sel_m}'. "
                    "Choose: 'tournament', 'roulette'."
                )

            # ── CROSSOVER ────────────────────────────────────────────────
            if np.random.rand() < crossover_rate:
                if cx_m == 'arithmetic':
                    c1, c2 = _arithmetic_crossover(p1, p2)
                elif cx_m == 'blend':
                    c1, c2 = _blend_crossover(p1, p2)
                else:
                    raise ValueError(
                        f"Unknown crossover_method '{cx_m}'. "
                        "Choose: 'arithmetic', 'blend'."
                    )
            else:
                c1, c2 = p1.copy(), p2.copy()

            # ── MUTATION ─────────────────────────────────────────────────
            if mut_m == 'gaussian':
                c1 = _gaussian_mutation(c1, mutation_rate, sigma)
                c2 = _gaussian_mutation(c2, mutation_rate, sigma)
            elif mut_m == 'uniform':
                c1 = _uniform_mutation(c1, mutation_rate)
                c2 = _uniform_mutation(c2, mutation_rate)
            else:
                raise ValueError(
                    f"Unknown mutation_method '{mut_m}'. "
                    "Choose: 'gaussian', 'uniform'."
                )

            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        # --- Evaluate new population ---
        population = np.array(new_population[:pop_size])
        fitnesses  = np.array([fitness_fn(ind, *fitness_args)
                               for ind in population])

        # --- Track best ---
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_ind = population[gen_best_idx].copy()

        history.append(best_fit)

        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            print(f"  Gen {gen+1:4d}/{n_generations}  |  "
                  f"Best: {best_fit:.4f}  |  "
                  f"Mean: {fitnesses.mean():.4f}")

    if verbose:
        print(f"\n  ✓ GA finished.  Best fitness = {best_fit:.4f}"
              f"  ({best_fit*100:.1f}% F1)\n")

    return best_ind, best_fit, history


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import pandas as pd
    from parkinsons_hidden import fitness_function, compute_n_params, evaluate_solution

    df       = pd.read_csv('parkinsons_preprocessed.csv')
    X        = df.drop(columns=['status']).values.astype(float)
    y        = df['status'].values.astype(int)
    n_params = compute_n_params()

    print(f"n_params = {n_params}")

    # Standard operators
    print("\n--- Standard GA (tournament + arithmetic + gaussian) ---")
    theta, fit, hist = genetic_algorithm(
        fitness_fn   = fitness_function,
        n_params     = n_params,
        fitness_args = (X, y),
        pop_size     = 20,
        n_generations= 20,
        selection_method='tournament',
        crossover_method='arithmetic',
        mutation_method ='gaussian',
        seed         = 42,
        verbose      = True,
    )
    m = evaluate_solution(theta, X, y)
    print(f"Recall={m['recall']:.4f}  Accuracy={m['accuracy']:.4f}  F1={m['f1']:.4f}")

    # Alternative operators
    print("\n--- Alternative GA (roulette + blend + uniform) ---")
    theta2, fit2, hist2 = genetic_algorithm(
        fitness_fn   = fitness_function,
        n_params     = n_params,
        fitness_args = (X, y),
        pop_size     = 20,
        n_generations= 20,
        selection_method='roulette',
        crossover_method='blend',
        mutation_method ='uniform',
        seed         = 42,
        verbose      = True,
    )
    m2 = evaluate_solution(theta2, X, y)
    print(f"Recall={m2['recall']:.4f}  Accuracy={m2['accuracy']:.4f}  F1={m2['f1']:.4f}")
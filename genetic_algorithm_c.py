"""
genetic_algorithm_hidden.py
============================
Genetic Algorithm for optimising real-valued neural network weights
— two hidden layers variant (22 → 10 → 10 → 1).

Extended Operators (NEW)
------------------------
This module implements both the original GA operators from the single-layer
version and three additional operators introduced for the report's
comparison experiments:

  1. Rank-based selection  (selection_method='rank')
     ─────────────────────────────────────────────
     Why:  Tournament selection can prematurely converge when a few
           super-fit individuals dominate early on.  Rank-based selection
           assigns selection probabilities proportional to *rank* rather
           than raw fitness, providing more uniform selection pressure and
           preserving diversity.  Expected impact: slower early convergence
           but better avoidance of local optima over longer runs.

  2. Two-point crossover  (crossover_method='two_point')
     ────────────────────────────────────────────────────
     Why:  Single-point and arithmetic crossover do not fully exploit the
           block structure of the weight vector (W1|b1|W2|b2|W3|b3).
           Two-point crossover selects two cut points and swaps the middle
           segment, preserving larger contiguous weight sub-vectors from
           each parent.  Expected impact: better inheritance of layer-level
           weight blocks, potentially faster per-layer specialisation.

  3. Adaptive mutation  (mutation_method='adaptive')
     ─────────────────────────────────────────────
     Why:  A fixed mutation rate risks either disrupting good solutions
           (if too high) or stagnating (if too low).  Adaptive mutation
           increases σ when the population has stagnated (no improvement
           for several generations) and decreases it after improvements,
           maintaining exploration/exploitation balance throughout the run.
           Expected impact: more robust convergence, especially for recall
           optimisation on the imbalanced Parkinson's dataset.

Statistical Context
-------------------
These three operators are tested alongside the standard operators in
utils_hidden.py using multiple independent runs.  Paired t-tests and
Wilcoxon signed-rank tests compare:
  • GA-standard vs GA-extended (operator comparison)
  • GA-best vs PSO-best (algorithm comparison)
Results and p-values are saved to CSV for inclusion in the report.

Usage
-----
    from genetic_algorithm_hidden import genetic_algorithm
    best_theta, best_fitness, history = genetic_algorithm(
        fitness_fn     = fitness_function,
        n_params       = 351,
        fitness_args   = (X, y),
        selection_method  = 'rank',        # extended operator
        crossover_method  = 'two_point',   # extended operator
        mutation_method   = 'adaptive',    # extended operator
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


def _rank_selection(population, fitnesses):
    """
    Rank-based selection  (NEW operator).

    Steps
    -----
    1. Rank individuals by fitness (worst = rank 1, best = rank N).
    2. Assign selection probability proportional to rank (linear ranking).
    3. Sample one individual according to those probabilities.

    Why rank-based?
    ---------------
    When raw fitness differences are large (e.g., recall=0.99 vs recall=0.97),
    tournament selection overwhelmingly picks the top individual every time.
    Rank-based selection compresses the fitness landscape: the best gets
    probability ∝ N, the worst ∝ 1.  This preserves population diversity
    and reduces premature convergence, at the cost of slightly slower
    early progress.

    Expected impact on Parkinson's task
    ------------------------------------
    Because recall is bounded [0,1] and the population often clusters near
    high recall, rank selection can redistribute exploration effort to
    individuals that differ structurally even if their recall is close.
    """
    n = len(population)
    ranks = np.argsort(np.argsort(fitnesses)) + 1   # rank 1 (worst) … N (best)
    probs = ranks / ranks.sum()
    idx   = np.random.choice(n, p=probs)
    return population[idx].copy()


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


def _single_point_crossover(p1, p2):
    """
    Single-point crossover: swap tails beyond a random cut point.
    Preserves large contiguous blocks from each parent.
    """
    n   = len(p1)
    pt  = np.random.randint(1, n)
    c1  = np.concatenate([p1[:pt], p2[pt:]])
    c2  = np.concatenate([p2[:pt], p1[pt:]])
    return c1, c2


def _two_point_crossover(p1, p2):
    """
    Two-point crossover  (NEW operator).

    Steps
    -----
    1. Sample two cut points pt1 < pt2 uniformly from [1, n-1].
    2. Child 1 inherits: [p1[:pt1], p2[pt1:pt2], p1[pt2:]]
    3. Child 2 inherits: [p2[:pt1], p1[pt1:pt2], p2[pt2:]]

    Why two-point?
    --------------
    The flat weight vector has a natural block structure:
        [ W1 | b1 | W2 | b2 | W3 | b3 ]
    Single-point crossover always inherits the *tail* of one parent.
    Two-point crossover can swap the *middle* block (e.g., all of W2+b2),
    preserving complete layer-weight matrices from each parent.
    This aligns better with the hypothesis that individual weight matrices
    (layers) have emergent functional roles.

    Expected impact on Parkinson's task
    ------------------------------------
    Swapping middle blocks (hidden-layer weights) may recombine
    parents that have each specialised in different sub-problems
    (e.g., one parent strong on true-positives, another on specificity),
    producing children that inherit the best of both layer's representations.
    """
    n   = len(p1)
    pts = sorted(np.random.choice(range(1, n), size=2, replace=False))
    pt1, pt2 = pts[0], pts[1]

    # Child 1: outer segments from p1, inner segment from p2
    c1 = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]])
    # Child 2: outer segments from p2, inner segment from p1
    c2 = np.concatenate([p2[:pt1], p1[pt1:pt2], p2[pt2:]])
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


def _adaptive_mutation(individual, mutation_rate, sigma,
                        stagnation_count, stagnation_threshold=10,
                        sigma_scale_up=2.0, sigma_scale_down=0.5,
                        sigma_min=0.01, sigma_max=1.0):
    """
    Adaptive mutation  (NEW operator).

    Mechanism
    ---------
    The mutation strength σ is dynamically adjusted based on population
    stagnation:

      • If no improvement for `stagnation_threshold` generations:
            σ_eff = min(σ * sigma_scale_up, sigma_max)
        → Increase exploration to escape local optima.

      • Otherwise:
            σ_eff = max(σ * sigma_scale_down, sigma_min)
        → Reduce perturbation to refine the current best solution.

    Gaussian noise N(0, σ_eff) is applied to each gene with probability
    `mutation_rate`.

    Why adaptive mutation?
    ----------------------
    A fixed σ creates a fundamental tension:
      - Too small → premature stagnation near a suboptimal peak
      - Too large → constant disruption of good solutions
    For the Parkinson's recall task, the optimisation surface has many
    near-equivalent weight configurations that achieve recall ≈ 1.0 by
    predicting all positive; adaptive mutation helps the GA actively
    explore when stuck and refine when making progress.

    Expected impact
    ---------------
    More stable convergence across multiple independent runs with fewer
    degenerate solutions.  The report's variance analysis should show
    lower inter-run standard deviation compared to fixed-σ Gaussian mutation.

    Parameters
    ----------
    stagnation_count     : int   — generations since last improvement
    stagnation_threshold : int   — generations of no improvement before scaling up
    sigma_scale_up       : float — σ multiplier when stagnating (default 2.0)
    sigma_scale_down     : float — σ multiplier when improving  (default 0.5)
    sigma_min / sigma_max: float — clamps on σ_eff
    """
    if stagnation_count >= stagnation_threshold:
        # Stagnating → expand search radius
        sigma_eff = min(sigma * sigma_scale_up, sigma_max)
    else:
        # Improving → refine
        sigma_eff = max(sigma * sigma_scale_down, sigma_min)

    mask   = np.random.rand(len(individual)) < mutation_rate
    noise  = np.random.randn(len(individual)) * sigma_eff
    mutant = individual.copy()
    mutant[mask] += noise[mask]
    return mutant


# ---------------------------------------------------------------------------
# INITIALISATION STRATEGIES
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# MAIN GENETIC ALGORITHM
# ---------------------------------------------------------------------------

def genetic_algorithm(
    fitness_fn:       Callable,
    n_params:         int,
    fitness_args:     tuple         = (),
    pop_size:         int           = 50,
    n_generations:    int           = 100,
    crossover_rate:   float         = 0.8,
    mutation_rate:    float         = 0.05,
    sigma:            float         = 0.1,
    tournament_size:  int           = 3,
    elitism:          int           = 2,
    # ── Operator selection ────────────────────────────────────────────────
    # selection_method options : 'tournament' | 'rank' | 'roulette'
    # crossover_method options : 'arithmetic' | 'blend' | 'single_point'
    #                            | 'two_point'   ← NEW
    # mutation_method  options : 'gaussian' | 'uniform' | 'adaptive' ← NEW
    # init_method      options : 'random' | 'xavier'
    selection_method: str           = 'tournament',
    crossover_method: str           = 'arithmetic',
    mutation_method:  str           = 'gaussian',
    init_method:      str           = 'random',
    # ── Architecture (used by Xavier init) ────────────────────────────────
    input_size:       int           = 22,
    hidden1_size:     int           = 10,
    hidden2_size:     int           = 10,
    output_size:      int           = 1,
    # ── Adaptive mutation config ──────────────────────────────────────────
    stagnation_threshold: int       = 10,   # generations before σ scales up
    sigma_scale_up:   float         = 2.0,
    sigma_scale_down: float         = 0.5,
    sigma_min:        float         = 0.01,
    sigma_max:        float         = 1.0,
    # ── Misc ──────────────────────────────────────────────────────────────
    seed:             Optional[int] = None,
    verbose:          bool          = True,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Genetic Algorithm main loop — two-hidden-layer variant.

    All three extended operators (rank selection, two-point crossover,
    adaptive mutation) are drop-in replacements selected via the
    corresponding method strings above.

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
    stagnation    = 0   # consecutive generations without improvement (adaptive mutation)

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
                # Standard tournament selection
                p1 = _tournament_selection(population, fitnesses, tournament_size)
                p2 = _tournament_selection(population, fitnesses, tournament_size)
            elif sel_m == 'rank':
                # Rank-based selection (NEW): fitness-rank probabilities,
                # reduces dominance of super-fit individuals
                p1 = _rank_selection(population, fitnesses)
                p2 = _rank_selection(population, fitnesses)
            elif sel_m == 'roulette':
                # Fitness-proportionate (roulette wheel) selection
                p1 = _roulette_selection(population, fitnesses)
                p2 = _roulette_selection(population, fitnesses)
            else:
                raise ValueError(
                    f"Unknown selection_method '{sel_m}'. "
                    "Choose: 'tournament', 'rank', 'roulette'."
                )

            # ── CROSSOVER ────────────────────────────────────────────────
            if np.random.rand() < crossover_rate:
                if cx_m == 'arithmetic':
                    # Standard arithmetic crossover: weighted average
                    c1, c2 = _arithmetic_crossover(p1, p2)
                elif cx_m == 'blend':
                    # BLX-α crossover: extended-interval uniform sampling
                    c1, c2 = _blend_crossover(p1, p2)
                elif cx_m == 'single_point':
                    # Single-point crossover: swap tail at one cut point
                    c1, c2 = _single_point_crossover(p1, p2)
                elif cx_m == 'two_point':
                    # Two-point crossover (NEW): swap middle block between
                    # two cut points; preserves layer-level weight blocks
                    c1, c2 = _two_point_crossover(p1, p2)
                else:
                    raise ValueError(
                        f"Unknown crossover_method '{cx_m}'. "
                        "Choose: 'arithmetic', 'blend', 'single_point', 'two_point'."
                    )
            else:
                # No crossover: children are clones of parents
                c1, c2 = p1.copy(), p2.copy()

            # ── MUTATION ─────────────────────────────────────────────────
            if mut_m == 'gaussian':
                # Standard Gaussian mutation: fixed σ throughout
                c1 = _gaussian_mutation(c1, mutation_rate, sigma)
                c2 = _gaussian_mutation(c2, mutation_rate, sigma)
            elif mut_m == 'uniform':
                # Uniform mutation: resample selected genes at random
                c1 = _uniform_mutation(c1, mutation_rate)
                c2 = _uniform_mutation(c2, mutation_rate)
            elif mut_m == 'adaptive':
                # Adaptive mutation (NEW): σ scales up when stagnating,
                # down when improving — balances exploration/exploitation
                c1 = _adaptive_mutation(
                    c1, mutation_rate, sigma, stagnation,
                    stagnation_threshold, sigma_scale_up, sigma_scale_down,
                    sigma_min, sigma_max
                )
                c2 = _adaptive_mutation(
                    c2, mutation_rate, sigma, stagnation,
                    stagnation_threshold, sigma_scale_up, sigma_scale_down,
                    sigma_min, sigma_max
                )
            else:
                raise ValueError(
                    f"Unknown mutation_method '{mut_m}'. "
                    "Choose: 'gaussian', 'uniform', 'adaptive'."
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
            stagnation = 0
        else:
            stagnation += 1

        history.append(best_fit)

        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            print(f"  Gen {gen+1:4d}/{n_generations}  |  "
                  f"Best: {best_fit:.4f}  |  "
                  f"Mean: {fitnesses.mean():.4f}  |  "
                  f"Stagnation: {stagnation:3d}")

    if verbose:
        print(f"\n  ✓ GA finished.  Best fitness = {best_fit:.4f}"
              f"  ({best_fit*100:.1f}% recall)\n")

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

    # Extended operators
    print("\n--- Extended GA (rank + two_point + adaptive) ---")
    theta2, fit2, hist2 = genetic_algorithm(
        fitness_fn   = fitness_function,
        n_params     = n_params,
        fitness_args = (X, y),
        pop_size     = 20,
        n_generations= 20,
        selection_method='rank',
        crossover_method='two_point',
        mutation_method ='adaptive',
        seed         = 42,
        verbose      = True,
    )
    m2 = evaluate_solution(theta2, X, y)
    print(f"Recall={m2['recall']:.4f}  Accuracy={m2['accuracy']:.4f}  F1={m2['f1']:.4f}")
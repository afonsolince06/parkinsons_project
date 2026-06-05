import numpy as np


def artificial_bee_colony(
    fitness_fn,
    n_params,
    fitness_args=(),
    colony_size=None,
    n_iterations=None,
    limit=None,
    lower_bound=-1.0,
    upper_bound=1.0,
    seed=42
):
    """
    Artificial Bee Colony (ABC) optimiser — maximisation variant.

    Implements the standard three-phase ABC loop (Karaboga, 2005):
      1. Employed-bee phase  — each bee exploits its own food source by
         perturbing a single random dimension and applies a greedy swap.
      2. Onlooker-bee phase  — bees are recruited proportionally to
         fitness (roulette-wheel selection) and repeat the same greedy
         perturbation, biasing the search toward currently better sources.
      3. Scout-bee phase     — any source whose trial counter reaches
         `limit` without improvement is abandoned and replaced by a new
         source drawn uniformly at random, restoring diversity.

    Parameters
    ----------
    fitness_fn : callable
        Objective function f(solution, *fitness_args) → float.
        Higher values are considered better (maximisation).
    n_params : int
        Dimensionality of each solution vector (= number of MLP weights).
    fitness_args : tuple, optional
        Extra positional arguments forwarded to `fitness_fn`.
    colony_size : int
        Total number of bees.  Employed bees = colony_size // 2;
        the same count is used for onlooker bees.
    n_iterations : int
        Number of full ABC cycles to execute.
    limit : int
        Stagnation threshold.  A food source is abandoned and scouted
        anew once it has failed to improve for `limit` consecutive trials.
        Typical values: n_params × colony_size / 2.
    lower_bound : float, optional (default=-1.0)
        Lower bound applied uniformly to every dimension.
    upper_bound : float, optional (default=1.0)
        Upper bound applied uniformly to every dimension.
    seed : int, optional (default=42)
        NumPy random seed for reproducibility.

    Returns
    -------
    best_solution : np.ndarray, shape (n_params,)
        Weight vector with the highest fitness found across all iterations.
    best_fitness : float
        Fitness value of `best_solution`.
    history : list of float
        Best fitness recorded at the end of each iteration (length =
        n_iterations), useful for convergence plots.
    """

    np.random.seed(seed)

    n_food_sources = colony_size // 2

    population = np.random.uniform(
        lower_bound,
        upper_bound,
        size=(n_food_sources, n_params)
    )

    fitness_values = np.array([
        fitness_fn(solution, *fitness_args)
        for solution in population
    ])

    trials = np.zeros(n_food_sources)

    best_index = np.argmax(fitness_values)
    best_solution = population[best_index].copy()
    best_fitness = fitness_values[best_index]

    history = []

    def generate_neighbour(index):
        """
        Produce a candidate neighbour of food source `index`.

        A single dimension j is chosen at random and perturbed by a
        random fraction phi ∈ (-1, 1) of the difference between the
        current source and a randomly chosen peer k ≠ index:

            neighbour[j] = source[j] + phi * (source[j] - peer[j])

        All dimensions are clipped to [lower_bound, upper_bound].

        Parameters
        ----------
        index : int
            Row index of the food source to be perturbed.

        Returns
        -------
        neighbour : np.ndarray, shape (n_params,)
            The perturbed candidate solution.
        """

        neighbour = population[index].copy()

        k = np.random.choice([i for i in range(n_food_sources) if i != index])

        j = np.random.randint(0, n_params)

        phi = np.random.uniform(-1, 1)

        neighbour[j] = population[index, j] + phi * (
            population[index, j] - population[k, j]
        )

        neighbour = np.clip(neighbour, lower_bound, upper_bound)

        return neighbour

    for iteration in range(n_iterations):

        for i in range(n_food_sources):
            neighbour = generate_neighbour(i)
            neighbour_fitness = fitness_fn(neighbour, *fitness_args)

            if neighbour_fitness > fitness_values[i]:
                population[i] = neighbour
                fitness_values[i] = neighbour_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        total_fitness = np.sum(fitness_values)

        if total_fitness == 0:
            probabilities = np.ones(n_food_sources) / n_food_sources
        else:
            probabilities = fitness_values / total_fitness

        for _ in range(n_food_sources):
            i = np.random.choice(n_food_sources, p=probabilities)

            neighbour = generate_neighbour(i)
            neighbour_fitness = fitness_fn(neighbour, *fitness_args)

            if neighbour_fitness > fitness_values[i]:
                population[i] = neighbour
                fitness_values[i] = neighbour_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        for i in range(n_food_sources):
            if trials[i] >= limit:
                population[i] = np.random.uniform(
                    lower_bound,
                    upper_bound,
                    size=n_params
                )
                fitness_values[i] = fitness_fn(population[i], *fitness_args)
                trials[i] = 0

        current_best_index = np.argmax(fitness_values)
        current_best_fitness = fitness_values[current_best_index]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_index].copy()

        history.append(best_fitness)

    return best_solution, best_fitness, history

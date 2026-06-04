import numpy as np


def artificial_bee_colony(
    fitness_fn,
    n_params,
    fitness_args=(),
    colony_size=50,
    n_iterations=50,
    limit=10,
    lower_bound=-1.0,
    upper_bound=1.0,
    seed=42,
    verbose=False
):

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

        if verbose:
            print(
                f"Iteration {iteration + 1}/{n_iterations} "
                f"| Best fitness: {best_fitness:.4f}"
            )

    return best_solution, best_fitness, history

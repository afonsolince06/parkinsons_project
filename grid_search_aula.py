import pandas as pd
import numpy as np

results_gridsearch = pd.DataFrame(columns=[
    'initialization', 'selection', 'crossover', 'mutation', 'mean_fitness', 'all_fitnesses'
])

# Example lists of operators
initializations = [random_init, xavier_initialization]
selections      = [roulette_selection, tournament_selection, rank_selection]
crossovers      = [arithmetic_crossover, blend_crossover]
mutations       = [gaussian_mutation, uniform_mutation, non_uniform_mutation]

for ini in initializations:
    for sel in selections:
        for cro in crossovers:
            for mut in mutations:
                fitness_vector = []
                
                for run in range(3):  # 30 independent runs
                    best_solution, best_fitness = genetic_algorithm(
                        initialization_func = ini,
                        fitness_func        = fitness_func,
                        selection_func      = sel,
                        crossover_func      = cro,
                        mutator             = mut,
                        pop_size            = 30,
                        generations         = 50,
                        mutation_rate       = 0.1
                    )
                    fitness_vector.append(best_fitness)
                
                mean_fit = np.mean(fitness_vector)
                results_gridsearch = results_gridsearch.append({
                    'initialization': ini.__name__,
                    'selection': sel.__name__,
                    'crossover': cro.__name__,
                    'mutation': mut.__name__,
                    'mean_fitness': mean_fit,
                    'all_fitnesses': fitness_vector
                }, ignore_index=True)

# Save results
results_gridsearch.to_csv('ga_gridsearch_30runs.csv', index=False)
import numpy as np
import pandas as pd
import warnings
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

from my_parkinsons_problem import (
    compute_n_params,
    generate_solution,
    fitness_function,
    evaluate_solution,
    unpack_weights,
    input_size,
    hidden_sizes,
    output_size,
)
from my_GA import (
    genetic_algorithm,
    xavier_initialization,
    random_initialization,
    roulette_selection,
    tournament_selection,
    rank_selection,
    arithmetic_crossover,
    blend_crossover,
    gaussian_mutation,
    uniform_mutation,
    non_uniform_mutation,
)
from my_pso import pso
df = pd.read_csv("parkinsons_preprocessed.csv")
X = df.drop(columns=["status"]).values.astype(float)
y = df["status"].values.astype(int)

def fitness_fn(solution):
    return fitness_function(solution, X, y,
                            input_size=input_size,
                            hidden_sizes=hidden_sizes,
                            output_size=output_size)
# Grid search configuration
initializations = [random_initialization, xavier_initialization]
selections      = [roulette_selection, tournament_selection, rank_selection]
crossovers      = [arithmetic_crossover, blend_crossover]
mutations       = [gaussian_mutation, uniform_mutation, non_uniform_mutation]

n_runs          = 2  # repetitions per combination
pop_size        = [30, 50, 100]
n_generations   = [30, 50, 100]
mutation_rate   = [0.05, 0.1, 0.2]
crossover_rate  = [0.7, 0.8, 0.9]
elitism         = [1,2]

# Prepare results DataFrame
results_gridsearch = pd.DataFrame(columns=[
    'initialization','selection','crossover','mutation',
    'pop_size','n_generations','mutation_rate','crossover_rate','elitism',
    'mean_fitness','std_fitness','all_fitnesses'
])

# Total number of parameters
n_params = compute_n_params(input_size, hidden_sizes, output_size)

for pop in pop_size:
    for ngen in n_generations:
        for mut_r in mutation_rate:
            for cro_r in crossover_rate:
                for eli in elitism:
                    for init in initializations:
                        for sel in selections:
                            for cros in crossovers:
                                for mut in mutations:
                                    all_fit= []
                                    for run in range(n_runs):
                                        best_sol, best_fit, history = genetic_algorithm(
                                            fitness_func=fitness_fn,
                                            init=init,
                                            selector=sel,
                                            mutator=mut,
                                            crossover=cros,
                                            pop_size = pop,
                                            generations = ngen,
                                            mutation_rate = mut_r,
                                            crossover_rate = cro_r,
                                            elitism=  eli ,
                                            layer_sizes=[22, 10, 10, 1],
                                            maximization=True
                                        )
                                        all_fit.append(best_fit)

                                        mean_fit = np.mean(all_fit)
                                        std_fit = np.std(all_fit)

                                        results_gridsearch = pd.concat([
                                            results_gridsearch,
                                            pd.DataFrame([{
                                                'initialization': init.__name__,
                                                'selection': sel.__name__,
                                                'crossover': cros.__name__,
                                                'mutation': mut.__name__,
                                                'pop_size': pop,
                                                'n_generations': ngen,
                                                'mutation_rate': mut_r,
                                                'crossover_rate': cro_r,
                                                'elitism': eli,
                                                'mean_fitness': mean_fit,
                                                'std_fitness': std_fit,
                                                'all_fitnesses': all_fit
                                            }])
                                        ], ignore_index=True)



results_gridsearch = pd.read_csv("ga_gridsearch_full.csv")
top5 = results_gridsearch.sort_values(by='mean_fitness', ascending=False).head(5)

plt.figure(figsize=(14, 7))

for i, (_, row) in enumerate(top5.iterrows(), 1):

    all_histories = ast.literal_eval(row['all_fitnesses'])
    n_runs = len(all_histories)

    for run_idx, history in enumerate(all_histories, 1):
        plt.plot(range(1, len(history) + 1),
                 history,
                 marker='o',
                 linestyle='-' if run_idx == 1 else '--',
                 alpha=0.7,
                 label=f"Top {i}, Run {run_idx}: {row['initialization']}-{row['selection']}-{row['crossover']}-{row['mutation']}" if run_idx == 1 else None)

plt.xlabel("Geração")
plt.ylabel("Fitness (Recall ou F1)")
plt.title("Convergência do GA – Top 5 combinações do Grid Search (todas as runs)")
plt.grid(True)
plt.legend(loc="lower right", fontsize=8)
plt.show()
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parkinsons_problem import (
    compute_n_params,
    fitness_function,
    input_size,
    hidden_sizes,
    output_size,
)

from Genetic_algorithm import (
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


RESULTS_FILE = "ga_gridsearch_results.csv"





def main():
    np.random.seed(42)

    initializations = [random_initialization, xavier_initialization]
    selections = [roulette_selection, tournament_selection, rank_selection]
    crossovers = [arithmetic_crossover, blend_crossover]
    mutations = [gaussian_mutation, uniform_mutation, non_uniform_mutation]

    n_runs = 2
    pop_size = [100]
    n_generations = [200]
    mutation_rate = [0.05, 0.1]
    crossover_rate = [0.9]
    elitism = [2]

    n_params = compute_n_params(input_size, hidden_sizes, output_size)

    total_combinations = (
        len(initializations)
        * len(selections)
        * len(crossovers)
        * len(mutations)
        * len(pop_size)
        * len(n_generations)
        * len(mutation_rate)
        * len(crossover_rate)
        * len(elitism)
    )

    results = []
    combo_count = 0
    start_time = time.time()

    for pop in pop_size:
        for gen in n_generations:
            for mut_r in mutation_rate:
                for cross_r in crossover_rate:
                    for eli in elitism:
                        for init in initializations:
                            for sel in selections:
                                for cross in crossovers:
                                    for mut in mutations:
                                        combo_count += 1
                                        all_fitness = []

                                        for run in range(n_runs):
                                            run_start = time.time()

                                            best_solution, best_fitness, history = genetic_algorithm(
                                                fitness_func=fitness_function,
                                                init=init,
                                                selector=sel,
                                                mutator=mut,
                                                crossover=cross,
                                                pop_size=pop,
                                                generations=gen,
                                                mutation_rate=mut_r,
                                                crossover_rate=cross_r,
                                                elitism=eli,
                                                layer_sizes=[22, 10, 10, 1],
                                                maximization=True,
                                            )

                                            elapsed_run = time.time() - run_start

                                            print(
                                                f"Run {run + 1} completed in {elapsed_run:.2f} seconds"
                                            )

                                            all_fitness.append(best_fitness)

                                        mean_fitness = np.mean(all_fitness)
                                        std_fitness = np.std(all_fitness)

                                        elapsed_total = time.time() - start_time
                                        avg_per_combo = elapsed_total / combo_count
                                        remaining = (
                                            total_combinations - combo_count
                                        ) * avg_per_combo

                                        print(
                                            f"[{combo_count}/{total_combinations}] "
                                            f"mean={mean_fitness:.4f} | "
                                            f"elapsed={elapsed_total / 60:.1f} min | "
                                            f"remaining={remaining / 60:.1f} min"
                                        )

                                        results.append(
                                            {
                                                "initialization": init.__name__,
                                                "selection": sel.__name__,
                                                "crossover": cross.__name__,
                                                "mutation": mut.__name__,
                                                "pop_size": pop,
                                                "n_generations": gen,
                                                "mutation_rate": mut_r,
                                                "crossover_rate": cross_r,
                                                "elitism": eli,
                                                "mean_fitness": mean_fitness,
                                                "std_fitness": std_fitness,
                                                "all_fitness": all_fitness,
                                            }
                                        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="mean_fitness", ascending=False)

    results_df.to_csv(RESULTS_FILE, index=False)

    print(f"\nGrid search concluded in {(time.time() - start_time) / 60:.1f} minutes.")
    print("\nBest configurations:")
    print(results_df.head(5))

    print("\nSaved file:")
    print(f"- {RESULTS_FILE}")

    


if __name__ == "__main__":
    main()
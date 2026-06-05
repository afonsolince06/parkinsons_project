import time
import pandas as pd
import random

from abc_c import artificial_bee_colony
from parkinsons_problem import (
    evaluate_solution,
    compute_n_params
)

def main():
    data_path = "parkinsons_preprocessed.csv"

    df = pd.read_csv(data_path)
    X = df.drop(columns=["status"]).values
    y = df["status"].values

    input_size = X.shape[1]
    hidden_sizes = (10, 10)
    output_size = 1

    n_params = compute_n_params(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size
    )

    def fitness_fn(solution):
        metrics = evaluate_solution(
            solution,
            X,
            y,
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size
        )
        return metrics["f1"]

    colony_sizes = [20, 50, 100]
    n_iterations_list = [30, 50, 100]
    limits = [5, 10]
    n_runs = 3

    results = []

    for colony_size in colony_sizes:
        for n_iterations in n_iterations_list:
            for limit in limits:

                all_fit = []

                for run in range(n_runs):

                    seed_run = 42 + run

                    start_time = time.time()
                    best_solution, best_fitness, history = artificial_bee_colony(
                        fitness_fn=fitness_fn,
                        n_params=n_params,
                        colony_size=colony_size,
                        n_iterations=n_iterations,
                        limit=limit,
                        lower_bound=-1.0,
                        upper_bound=1.0,
                        seed=seed_run
                    )
                    runtime = time.time() - start_time

                    metrics = evaluate_solution(
                        best_solution,
                        X,
                        y,
                        input_size=input_size,
                        hidden_sizes=hidden_sizes,
                        output_size=output_size
                    )

                    all_fit.append(best_fitness)

                    print(
                        f"Run {run + 1} | ABC | colony_size={colony_size}, "
                        f"iterations={n_iterations}, limit={limit} "
                        f"-> F1={metrics['f1']:.4f}, "
                        f"Accuracy={metrics['accuracy']:.4f}, "
                        f"Recall={metrics['recall']:.4f}, "
                        f"Precision={metrics['precision']:.4f}"
                    )

                # Guardar resultados por configuração média
                results.append({
                    "algorithm": "ABC",
                    "fitness_metric": "f1",
                    "colony_size": colony_size,
                    "n_iterations": n_iterations,
                    "limit": limit,
                    "mean_fitness": sum(all_fit)/len(all_fit),
                    "std_fitness": pd.Series(all_fit).std(),
                    "all_fitnesses": all_fit
                })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="mean_fitness", ascending=False)
    results_df.to_csv("my_abc_grid_search_results.csv", index=False)

    print("\nABC mini grid search completed.")
    print("\nBest configuration:")
    print(results_df.iloc[0])
    print("\nSaved file: my_abc_grid_search_results.csv")

if __name__ == "__main__":
    main()
import time
import pandas as pd

from my_abc_c import artificial_bee_colony

from my_parkinsons_problem import (
    evaluate_solution,
    compute_n_params
)


def main():
    seed = 42
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

    colony_sizes = [20, 50,100]
    n_iterations_list = [30, 50,100]
    limits = [5, 10]

    results = []

    for colony_size in colony_sizes:
        for n_iterations in n_iterations_list:
            for limit in limits:

                start_time = time.time()

                best_solution, best_fitness, history = artificial_bee_colony(
                    fitness_fn=fitness_fn,
                    n_params=n_params,
                    colony_size=colony_size,
                    n_iterations=n_iterations,
                    limit=limit,
                    lower_bound=-1.0,
                    upper_bound=1.0,
                    seed=seed
                )

                runtime = time.time() - start_time

                metrics = evaluate_solution(
                    best_solution,
                    X,
                    y,
                    input_size=input_size,
                    hidden_sizes= hidden_sizes,
                    output_size=output_size
                )

                results.append({
                    "algorithm": "ABC",
                    "fitness_metric": "f1",
                    "colony_size": colony_size,
                    "n_iterations": n_iterations,
                    "limit": limit,
                    "best_fitness": best_fitness,
                    "accuracy": metrics["accuracy"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "f1": metrics["f1"],
                    "runtime": runtime,
                    "seed": seed
                })

                print(
                    f"ABC | colony_size={colony_size}, "
                    f"iterations={n_iterations}, "
                    f"limit={limit} "
                    f"-> F1={metrics['f1']:.4f}, "
                    f"Accuracy={metrics['accuracy']:.4f}, "
                    f"Recall={metrics['recall']:.4f}, "
                    f"Precision={metrics['precision']:.4f}"
                )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False)

    results_df.to_csv("abc_grid_search_results.csv", index=False)

    print("\nABC mini grid search completed.")
    print("\nBest configuration:")
    print(results_df.iloc[0])

    print("\nSaved file:")
    print("- abc_grid_search_results.csv")


if __name__ == "__main__":
    main()
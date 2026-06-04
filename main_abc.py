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
    hidden_sizes= (10,10)
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

    start_time = time.time()

    best_solution, best_fitness, history = artificial_bee_colony(
        fitness_fn=fitness_fn,
        n_params=n_params,
        colony_size=50,
        n_iterations=50,
        limit=10,
        lower_bound=-1.0,
        upper_bound=1.0,
        seed=seed,
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

    print("\nABC Results")
    print("-" * 55)
    print(f"Best fitness: {best_fitness}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Recall: {metrics['recall']}")
    print(f"Precision: {metrics['precision']}")
    print(f"F1: {metrics['f1']}")
    print(f"Predicted positives: {metrics['n_pred_pos']}")
    print(f"Predicted negatives: {metrics['n_pred_neg']}")
    print(f"Runtime: {runtime:.2f} seconds")

    results = pd.DataFrame([{
        "algorithm": "ABC",
        "fitness_metric": "f1",
        "colony_size": 50,
        "n_iterations": 50,
        "limit": 10,
        "best_fitness": best_fitness,
        "accuracy": metrics["accuracy"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "n_pred_pos": metrics["n_pred_pos"],
        "n_pred_neg": metrics["n_pred_neg"],
        "runtime": runtime,
        "seed": seed
    }])

    history_df = pd.DataFrame({
        "iteration": range(1, len(history) + 1),
        "best_fitness": history
    })

    results.to_csv("abc_results.csv", index=False)
    history_df.to_csv("abc_history.csv", index=False)

    print("\nSaved files:")
    print("- abc_results.csv")
    print("- abc_history.csv")


if __name__ == "__main__":
    main()
import time
import numpy as np
import pandas as pd

from abc_c import artificial_bee_colony

from parkinsons_problem_c import (
    fitness_function,
    evaluate_solution,
    compute_n_params
)


def main():

    SEED = 42
    DATA_PATH = "parkinsons_preprocessed.csv"

    COLONY_SIZE = 50
    N_ITERATIONS = 50
    LIMIT = 10

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["status"]).values.astype(float)
    y = df["status"].values.astype(int)

    n_samples, n_features = X.shape

    print("=" * 60)
    print("Artificial Bee Colony — Parkinson's Disease MLP")
    print("=" * 60)

    print(f"Dataset: {DATA_PATH}")
    print(f"Samples: {n_samples}")
    print(f"Features: {n_features}")
    print(f"Parkinson's cases: {(y == 1).sum()}")
    print(f"Healthy controls: {(y == 0).sum()}")

    n_params = compute_n_params()

    print(f"\nNumber of parameters to optimise: {n_params}")
    print(f"Colony size: {COLONY_SIZE}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Limit: {LIMIT}")
    print(f"Seed: {SEED}")

    def abc_fitness(solution, X, y):
        return fitness_function(solution, X, y)

    print("\nRunning ABC optimisation...\n")

    start_time = time.perf_counter()

    best_solution, best_fitness, history = artificial_bee_colony(
        fitness_fn=abc_fitness,
        n_params=n_params,
        fitness_args=(X, y),
        colony_size=COLONY_SIZE,
        n_iterations=N_ITERATIONS,
        limit=LIMIT,
        lower_bound=-1.0,
        upper_bound=1.0,
        seed=SEED,
        verbose=True
    )

    runtime = time.perf_counter() - start_time

    metrics = evaluate_solution(best_solution, X, y)

    print("\nABC Results")
    print("-" * 60)
    print(f"Best fitness: {best_fitness}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Recall: {metrics['recall']}")
    print(f"Precision: {metrics['precision']}")
    print(f"F1: {metrics['f1']}")
    print(f"Predicted positives: {metrics['n_pred_pos']}")
    print(f"Predicted negatives: {metrics['n_pred_neg']}")
    print(f"Runtime: {runtime:.2f} seconds")

    results = {
        "algorithm": "ABC",
        "fitness_metric": "same_as_problem_definition",
        "best_fitness": best_fitness,
        "accuracy": metrics["accuracy"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "n_pred_pos": metrics["n_pred_pos"],
        "n_pred_neg": metrics["n_pred_neg"],
        "colony_size": COLONY_SIZE,
        "iterations": N_ITERATIONS,
        "limit": LIMIT,
        "runtime_seconds": round(runtime, 2),
        "seed": SEED
    }

    pd.DataFrame([results]).to_csv("abc_results.csv", index=False)

    history_df = pd.DataFrame({
        "iteration": np.arange(1, len(history) + 1),
        "best_fitness": history
    })

    history_df.to_csv("abc_history.csv", index=False)

    print("\nSaved files:")
    print("- abc_results.csv")
    print("- abc_history.csv")


if __name__ == "__main__":
    main()
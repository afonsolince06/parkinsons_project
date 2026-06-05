import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_parkinsons_problem import (
    compute_n_params,
    fitness_function,
    evaluate_solution,
    input_size,
    hidden_sizes,
    output_size,
)

from my_GA import (
    genetic_algorithm,
    tournament_selection,
    blend_crossover,
)

from my_pso import pso

from my_differential_evolution import differential_evolution

from my_abc_c import artificial_bee_colony


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------

df = pd.read_csv("parkinsons_preprocessed.csv")

X = df.drop(columns=["status"]).values.astype(float)
y = df["status"].values.astype(int)

n_params = compute_n_params(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    output_size=output_size
)


def fitness_fn(solution):
    return fitness_function(
        solution,
        X,
        y,
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
    )


def print_metrics(name, metrics, runtime):
    print(f"\n=== {name} ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Predicted positives: {metrics['predicted_positive']}")
    print(f"Predicted negatives: {metrics['predicted_negative']}")
    print(f"Runtime: {runtime:.2f} seconds")


def evaluate_algorithm(name, solution, runtime):
    metrics = evaluate_solution(
        solution,
        X,
        y,
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
    )

    print_metrics(name, metrics, runtime)

    return {
        "algorithm": name,
        "accuracy": metrics["accuracy"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "predicted_positive": metrics["predicted_positive"],
        "predicted_negative": metrics["predicted_negative"],
        "runtime": runtime,
    }


def main():
    results = []

    # ---------------------------------------------------------------------
    # Genetic Algorithm
    # ---------------------------------------------------------------------
    # Best GA configuration from grid search:
    # random initialization, tournament selection, blend crossover,
    # non-uniform mutation, pop_size=100, generations=200,
    # mutation_rate=0.05, crossover_rate=0.9, elitism=2

    start_time = time.time()

    ga_solution, ga_fitness, ga_history = genetic_algorithm(
        fitness_func=fitness_fn,
        init="random",
        selector=tournament_selection,
        mutator="non_uniform",
        crossover=blend_crossover,
        pop_size=100,
        generations=200,
        mutation_rate=0.05,
        crossover_rate=0.9,
        elitism=2,
        layer_sizes=[22, 10, 10, 1],
        maximization=True,
    )

    ga_runtime = time.time() - start_time
    results.append(evaluate_algorithm("GA", ga_solution, ga_runtime))

    # ---------------------------------------------------------------------
    # Particle Swarm Optimization
    # ---------------------------------------------------------------------
    # Best PSO configuration from grid search:
    # n_particles=100, n_iterations=100, w=0.9, c1=1.0, c2=1.0

    start_time = time.time()

    pso_solution, pso_fitness, pso_history = pso(
        fitness_func=fitness_fn,
        n_particles=50,
        n_params=n_params,
        n_iterations=100,
        w=0.5,
        c1=2.0,
        c2=1.5,
        low=-1.0,
        high=1.0,
    )

    pso_runtime = time.time() - start_time
    results.append(evaluate_algorithm("PSO", pso_solution, pso_runtime))

    # ---------------------------------------------------------------------
    # Differential Evolution
    # ---------------------------------------------------------------------
    # Best DE configuration from grid search:
    # pop_size=50, generations=100, F=0.8, CR=0.9

    start_time = time.time()

    de_solution, de_fitness, de_history = differential_evolution(
        fitness_func=fitness_fn,
        n_params=n_params,
        pop_size=50,
        generations=100,
        bounds=(-1.0, 1.0),
        F=0.8,
        CR=0.9,
        maximization=True,
        seed=42,
    )

    de_runtime = time.time() - start_time
    results.append(evaluate_algorithm("DE", de_solution, de_runtime))

    # ---------------------------------------------------------------------
    # Artificial Bee Colony
    # ---------------------------------------------------------------------
    # Best ABC configuration from grid search:
    # colony_size=100, n_iterations=100, limit=10

    start_time = time.time()

    abc_solution, abc_fitness, abc_history = artificial_bee_colony(
        fitness_fn=fitness_fn,
        n_params=n_params,
        colony_size=100,
        n_iterations=100,
        limit=10,
        lower_bound=-1.0,
        upper_bound=1.0,
        seed=42,
    )

    abc_runtime = time.time() - start_time
    results.append(evaluate_algorithm("ABC", abc_solution, abc_runtime))

    # ---------------------------------------------------------------------
    # Save final results
    # ---------------------------------------------------------------------

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False)

    results_df.to_csv("final_algorithm_comparison.csv", index=False)

    print("\n=== Final comparison ===")
    print(results_df)

    print("\nSaved file:")
    print("- final_algorithm_comparison.csv")

    # ---------------------------------------------------------------------
    # Convergence plot
    # ---------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(ga_history, label="GA", linewidth=1.5)
    plt.plot(pso_history, label="PSO", linewidth=1.5)
    plt.plot(de_history, label="DE", linewidth=1.5)
    plt.plot(abc_history, label="ABC", linewidth=1.5)

    plt.xlabel("Generation / Iteration")
    plt.ylabel("Best F1-score")
    plt.title("Convergence comparison — GA, PSO, DE and ABC")
    plt.legend()
    plt.tight_layout()

    plt.savefig("final_convergence_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSaved plot:")
    print("- final_convergence_comparison.png")


if __name__ == "__main__":
    main()
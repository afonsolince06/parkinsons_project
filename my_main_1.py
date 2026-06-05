import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

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
    gaussian_mutation,
)
from my_pso import pso

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("parkinsons_preprocessed.csv")
X  = df.drop(columns=["status"]).values.astype(float)
y  = df["status"].values.astype(int)

n_params = compute_n_params(input_size, hidden_sizes, output_size)

def fitness_fn(solution):
    return fitness_function(solution, X, y,
                            input_size=input_size,
                            hidden_sizes=hidden_sizes,
                            output_size=output_size)

# ── GA ────────────────────────────────────────────────────────────────────────
# Best config found in grid search:
# xavier init, tournament selection, blend crossover, gaussian mutation
# pop=100, generations=400, mut_rate=0.03, crossover_rate=0.9, elitism=2

ga_solution, ga_fitness, ga_history = genetic_algorithm(
    fitness_func   = fitness_fn,
    init           = 'random_initialization',
    selector       = tournament_selection,
    mutator        = 'gaussian',
    crossover      = blend_crossover,
    pop_size       = 100,
    generations    = 400,
    mutation_rate  = 0.03,
    crossover_rate = 0.9,
    sigma          = 0.1,
    elitism        = 2,
    layer_sizes    = [22, 10, 10, 1],
    maximization   = True,
)

# ── PSO ───────────────────────────────────────────────────────────────────────
# Best config found in grid search:
# n_particles=100, n_iterations=100, w=0.9, c1=1.0, c2=1.0  (F1 ≈ 0.9735)

pso_solution, pso_fitness, pso_history = pso(
    fitness_func  = fitness_fn,
    n_particles   = 100,
    n_params      = n_params,
    n_iterations  = 100,
    w             = 0.9,
    c1            = 1.0,
    c2            = 1.0,
    low           = -1.0,
    high          = 1.0,
)

# ── results ───────────────────────────────────────────────────────────────────
ga_metrics  = evaluate_solution(ga_solution,  X, y, input_size, hidden_sizes, output_size)
pso_metrics = evaluate_solution(pso_solution, X, y, input_size, hidden_sizes, output_size)

print("\n=== GA ===")
for k, v in ga_metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

print("\n=== PSO ===")
for k, v in pso_metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# ── convergence plot ──────────────────────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.plot(ga_history,  label="GA",  linewidth=1.5)
plt.plot(pso_history, label="PSO", linewidth=1.5)
plt.xlabel("Generation / Iteration")
plt.ylabel("Best F1-score")
plt.title("Convergence – GA vs PSO")
plt.legend()
plt.tight_layout()
plt.savefig("convergence.png", dpi=150)
plt.show()
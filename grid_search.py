"""
grid_search.py
==============
Simple GA hyperparameter grid search for the Parkinson's MLP problem.

Network architecture:  22 -> 10 -> 10 -> 1   (n_params = 351)
Fitness function    :  F1-score (pos_label = 1)

Grid dimensions
---------------
  initialization : random | xavier
  selection      : tournament | roulette
  crossover      : arithmetic | blend
  mutation       : gaussian | uniform

For each of the 2 x 2 x 2 x 2 = 16 combinations we run N_RUNS = 30
independent GA runs and record the best fitness of each run.

Output
------
  grid_search.csv   -- one row per combination with mean fitness and
                        the full 30-run vector
  stdout            -- progress + best-combination evaluation
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import random

# ---- project imports --------------------------------------------------------
from my_parkinsons_problem import (
    fitness_function,
    evaluate_solution,
    compute_n_params,
    DEFAULT_INPUT_SIZE,
    DEFAULT_HIDDEN1_SIZE,
    DEFAULT_HIDDEN2_SIZE,
    DEFAULT_OUTPUT_SIZE,
)
from genetic_algorithm_c import genetic_algorithm

# =============================================================================
# 0.  SETUP
# =============================================================================

# Load dataset
df = pd.read_csv("parkinsons_preprocessed.csv")
X  = df.drop(columns=["status"]).values.astype(float)
y  = df["status"].values.astype(int)

# Network configuration
N_PARAMS   = compute_n_params()            # 351  (22->10->10->1)
H1, H2     = DEFAULT_HIDDEN1_SIZE, DEFAULT_HIDDEN2_SIZE

# Fixed GA settings (same for every combination)
POP_SIZE      = 30
N_ITER        = 50
MUTATION_RATE = 0.1
N_RUNS        = 30    # independent runs per combination

# Operator lists to search over
initialization = ["random", "xavier"]
selection      = ["tournament", "roulette"]
crossover      = ["arithmetic", "blend"]
mutation       = ["gaussian", "uniform"]

print("=" * 65)
print("  GA Grid Search  --  Parkinson's MLP  (22->10->10->1)")
print(f"  pop={POP_SIZE}  iters={N_ITER}  mut_rate={MUTATION_RATE}  runs={N_RUNS}")
print(f"  n_params={N_PARAMS}")
print(f"  Total combinations: {len(initialization)*len(selection)*len(crossover)*len(mutation)}")
print("=" * 65)

# =============================================================================
# 1.  GRID SEARCH
# =============================================================================

results = []   # list of dicts, one per combination

combo_number = 0
total_combos = len(initialization) * len(selection) * len(crossover) * len(mutation)

for ini in initialization:
    for sel in selection:
        for cro in crossover:
            for mut in mutation:

                combo_number += 1
                tag = f"init={ini:<8} sel={sel:<12} cx={cro:<12} mut={mut}"
                print(f"\n[{combo_number:>2}/{total_combos}]  {tag}")
                print(f"  Running {N_RUNS} independent runs ...", end="", flush=True)

                run_fitnesses = []
                t0 = time.perf_counter()

                for run in range(N_RUNS):
                    # Each run uses a different seed so results are independent
                    run_seed = run

                    best_theta, best_fit, _ = genetic_algorithm(
                        fitness_fn       = fitness_function,
                        n_params         = N_PARAMS,
                        fitness_args     = (X, y),
                        pop_size         = POP_SIZE,
                        n_generations    = N_ITER,
                        mutation_rate    = MUTATION_RATE,
                        crossover_rate   = 0.8,          # fixed
                        sigma            = 0.1,          # fixed Gaussian std
                        tournament_size  = 3,            # fixed
                        elitism          = 2,            # fixed
                        init_method      = ini,
                        selection_method = sel,
                        crossover_method = cro,
                        mutation_method  = mut,
                        input_size       = DEFAULT_INPUT_SIZE,
                        hidden1_size     = H1,
                        hidden2_size     = H2,
                        output_size      = DEFAULT_OUTPUT_SIZE,
                        seed             = run_seed,
                        verbose          = False,
                    )

                    run_fitnesses.append(best_fit)

                elapsed      = time.perf_counter() - t0
                mean_fitness = float(np.mean(run_fitnesses))

                print(f"  done  ({elapsed:.0f}s)")
                print(f"  mean F1 = {mean_fitness:.4f}   "
                      f"min={min(run_fitnesses):.4f}   "
                      f"max={max(run_fitnesses):.4f}")

                results.append({
                    "initialization": ini,
                    "selection"     : sel,
                    "crossover"     : cro,
                    "mutation"      : mut,
                    "mean_fitness"  : round(mean_fitness, 6),
                    "all_fitnesses" : run_fitnesses,   # list of 30 values
                })

# =============================================================================
# 2.  SAVE TO CSV
# =============================================================================

df_results = pd.DataFrame(results)
df_results.to_csv("grid_search.csv", index=False)
print("\n" + "=" * 65)
print("  Saved results to  grid_search.csv")

# =============================================================================
# 3.  BEST COMBINATION -- full evaluation
# =============================================================================

best_row = df_results.loc[df_results["mean_fitness"].idxmax()]

print("\n" + "=" * 65)
print("  BEST COMBINATION  (by mean F1 across 30 runs)")
print("=" * 65)
print(f"  initialization : {best_row['initialization']}")
print(f"  selection      : {best_row['selection']}")
print(f"  crossover      : {best_row['crossover']}")
print(f"  mutation       : {best_row['mutation']}")
print(f"  mean F1        : {best_row['mean_fitness']:.4f}")

print("\n  Running best combination once (seed=42) for full metrics ...")

best_theta, best_fit, _ = genetic_algorithm(
    fitness_fn       = fitness_function,
    n_params         = N_PARAMS,
    fitness_args     = (X, y),
    pop_size         = POP_SIZE,
    n_generations    = N_ITER,
    mutation_rate    = MUTATION_RATE,
    crossover_rate   = 0.8,
    sigma            = 0.1,
    tournament_size  = 3,
    elitism          = 2,
    init_method      = best_row["initialization"],
    selection_method = best_row["selection"],
    crossover_method = best_row["crossover"],
    mutation_method  = best_row["mutation"],
    input_size       = DEFAULT_INPUT_SIZE,
    hidden1_size     = H1,
    hidden2_size     = H2,
    output_size      = DEFAULT_OUTPUT_SIZE,
    seed             = 42,
    verbose          = False,
)

metrics = evaluate_solution(best_theta, X, y)

print("\n  Full evaluation on complete dataset:")
print(f"    F1        : {metrics['f1']:.4f}")
print(f"    Recall    : {metrics['recall']:.4f}")
print(f"    Precision : {metrics['precision']:.4f}")
print(f"    Accuracy  : {metrics['accuracy']:.4f}")
print(f"    Predicted positive : {metrics['n_pred_pos']}  "
      f"(of {int((y==1).sum())} actual)")
print(f"    Predicted negative : {metrics['n_pred_neg']}  "
      f"(of {int((y==0).sum())} actual)")
if metrics.get("degenerate", False):
    print("    WARNING: degenerate solution (all predictions positive)")

# =============================================================================
# 4.  TOP-5 SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("  TOP-5 combinations (by mean F1)")
print("=" * 65)
print(f"  {'init':<8} {'sel':<12} {'cx':<12} {'mut':<10}  mean_F1")
print("  " + "-" * 55)
for _, row in df_results.nlargest(5, "mean_fitness").iterrows():
    print(f"  {row['initialization']:<8} {row['selection']:<12} "
          f"{row['crossover']:<12} {row['mutation']:<10}  {row['mean_fitness']:.4f}")

print("\nDone.\n")

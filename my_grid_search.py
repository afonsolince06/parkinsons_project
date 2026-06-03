import pandas as pd
import numpy as np

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
    non_uniform_mutation)

from my_parkinsons_problem import (
    compute_n_params,
    generate_solution,
    fitness_function,
    evaluate_solution,
    unpack_weights,
    input_size,
    hidden_sizes,
    output_size)

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Load dataset
# ──────────────────────────────────────────────────────────────────────────────
DATASET_PATH = "parkinsons_preprocessed.csv"  # adjust if needed
LABEL_COLUMN = "status"  # 1 = Parkinson's, 0 = healthy

print("=" * 60)
print("Loading dataset …")
try:
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=[LABEL_COLUMN]).values.astype(float)
    y = df[LABEL_COLUMN].values.astype(int)
    print(f"  → {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  → class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Dataset not found at '{DATASET_PATH}'.\n"
        "Place 'parkinsons_preprocessed.csv' in the same directory."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: wrapped fitness (single-argument, as GA/PSO expect)
# ──────────────────────────────────────────────────────────────────────────────
def fitness_fn(solution):
    """Single-argument wrapper used by both GA and PSO."""
    return fitness_function(solution, X, y, input_size, hidden_sizes, output_size)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SANITY TESTS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1 – SANITY TESTS")
print("=" * 60)

# 1-a  parameter count
expected_n = compute_n_params(input_size, hidden_sizes, output_size)
print(f"\n[1a] Expected parameter count : {expected_n}")
assert expected_n == 351, f"Expected 351 parameters, got {expected_n}"
print("     ✓ Correct (351 params for 22-10-10-1 architecture)")

# 1-b  generate_solution length
sol = generate_solution(expected_n)
assert len(sol) == expected_n, f"generate_solution returned {len(sol)} params, expected {expected_n}"
print(f"[1b] generate_solution length : {len(sol)} ✓")

# 1-c  unpack_weights shapes
coefs, intercepts = unpack_weights(sol, input_size, hidden_sizes, output_size)
expected_shapes = [(22, 10), (10, 10), (10, 1)]
for k, (c, s) in enumerate(zip(coefs, expected_shapes)):
    assert c.shape == s, f"coefs[{k}] shape {c.shape} ≠ {s}"
print(f"[1c] unpack_weights shapes    : {[c.shape for c in coefs]} ✓")

expected_bias_shapes = [(10,), (10,), (1,)]
for k, (b, s) in enumerate(zip(intercepts, expected_bias_shapes)):
    assert b.shape == s, f"intercepts[{k}] shape {b.shape} ≠ {s}"
print(f"     bias shapes               : {[b.shape for b in intercepts]} ✓")

# 1-d  fitness_function returns float in [0, 1]
fit_val = fitness_fn(sol)
assert 0.0 <= fit_val <= 1.0, f"Fitness out of [0,1]: {fit_val}"
print(f"[1d] fitness_function output  : {fit_val:.4f}  (in [0,1]) ✓")

# 1-e  evaluate_solution returns all metrics
metrics = evaluate_solution(sol, X, y, input_size, hidden_sizes, output_size)
required_keys = {"accuracy", "recall", "precision", "f1", "predicted_positive", "predicted_negative"}
assert required_keys.issubset(metrics.keys()), f"Missing keys: {required_keys - metrics.keys()}"
print(f"[1e] evaluate_solution keys   : {set(metrics.keys())} ✓")
print(f"     sample metrics (random sol): "
      f"acc={metrics['accuracy']:.3f}, rec={metrics['recall']:.3f}, "
      f"f1={metrics['f1']:.3f}, +pred={metrics['predicted_positive']}")

# 1-f  Xavier init shape
layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
pop_xavier = xavier_initialization(5, layer_sizes)
assert len(pop_xavier) == 5
assert len(pop_xavier[0]) == expected_n, (
    f"Xavier individual length {len(pop_xavier[0])} ≠ {expected_n}"
)
print(f"[1f] xavier_initialization    : 5 individuals × {len(pop_xavier[0])} params ✓")

# 1-g  Quick GA smoke test (tiny params)
print("\n[1g] GA smoke test (pop=10, gen=3) …")

best_sol, best_fit, history = genetic_algorithm(
    fitness_func=fitness_fn,
    init='random',
    selector=tournament_selection,
    mutator='gaussian',
    crossover=arithmetic_crossover,
    pop_size=30,
    generations=20,
    mutation_rate=0.1,
    crossover_rate=0.8,
    elitism=2,
    layer_sizes=layer_sizes,
)

assert len(history) == 4, f"History length {len(history)} (expected gen+1=4)"
assert len(best_sol) == expected_n
print(f"     best_fit={best_fit:.4f}  history_len={len(history)}✓")

# 1-h  Quick PSO smoke test
print("\n[1h] PSO smoke test (particles=5, iter=3) …")

pso_sol, pso_fit, pso_history = my_pso(
    fitness_func=fitness_fn,
    n_particles=30,
    n_params=expected_n,
    n_iterations=30,
    seed=42,
)

assert len(pso_history) == 3, f"PSO history length {len(pso_history)} (expected 3)"
assert len(pso_sol) == expected_n
print(f"     pso_fit={pso_fit:.4f}  history_len={len(pso_history)} ")

print("\n✅  All sanity tests passed.\n")
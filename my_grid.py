import pandas as pd
import numpy as np

from my_GA import (
    genetic_algorithm,
    xavier_initialisation,
    random_initialisation,
    roulette_wheel_selection,
    tournament_selection,
    rank_selection,
    arithmetic_crossover,
    blend_crossover,
    gaussian_mutation,
    uniform_reset_mutation,
    non_uniform_mutation
)

from my_parkinsons_problem import (
    compute_n_params,
    generate_solution,
    fitness_function,
    evaluate_solution,
    _unpack_weights,
    DEFAULT_INPUT_SIZE as input_size,
    DEFAULT_HIDDEN_SIZE as hidden_sizes,
    DEFAULT_OUTPUT_SIZE as output_size
)

# ───────────────────────────────────────────────────────────────
# 0. Load dataset
# ───────────────────────────────────────────────────────────────
DATASET_PATH = "parkinsons_preprocessed.csv"

df = pd.read_csv(DATASET_PATH)
X = df.drop(columns=['status']).values.astype(float)
y = df['status'].values.astype(int)

print("="*60)
print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ───────────────────────────────────────────────────────────────
# Helper for GA/PSO: single-argument fitness
# ───────────────────────────────────────────────────────────────
def fitness_fn(solution):
    return fitness_function(solution, X, y,
                            input_size=input_size,
                            hidden_size=hidden_sizes,
                            output_size=output_size)

# ───────────────────────────────────────────────────────────────
# 1. SANITY TESTS
# ───────────────────────────────────────────────────────────────
print("\nSECTION 1 – SANITY TESTS")

# 1-a. Parameter count
expected_n = compute_n_params(input_size, hidden_sizes, output_size)
print(f"[1a] Total parameters: {expected_n} ✓")

# 1-b. generate_solution length
sol = generate_solution(expected_n)
assert len(sol) == expected_n
print(f"[1b] generate_solution length: {len(sol)} ✓")

# 1-c. unpack_weights shapes
coefs, intercepts = _unpack_weights(sol, input_size, hidden_sizes, output_size)
expected_shapes = [(input_size, hidden_sizes),
                   (hidden_sizes, hidden_sizes),
                   (hidden_sizes, output_size)]
for k, (c, s) in enumerate(zip(coefs, expected_shapes)):
    assert c.shape == s, f"coefs[{k}] shape {c.shape} != {s}"
print(f"[1c] unpack_weights shapes: {[c.shape for c in coefs]} ✓")

# Bias shapes
expected_bias_shapes = [(hidden_sizes,), (hidden_sizes,), (output_size,)]
for k, (b, s) in enumerate(zip(intercepts, expected_bias_shapes)):
    assert b.shape == s, f"intercepts[{k}] shape {b.shape} != {s}"
print(f"     bias shapes: {[b.shape for b in intercepts]} ✓")

# 1-d. fitness_function output
fit_val = fitness_fn(sol)
assert 0.0 <= fit_val <= 1.0
print(f"[1d] fitness_function output: {fit_val:.4f} ✓")

# 1-e. evaluate_solution output
metrics = evaluate_solution(sol, X, y,
                            input_size, hidden_sizes, output_size)
required_keys = {"accuracy", "recall", "precision", "f1", "n_pred_pos", "n_pred_neg"}
assert required_keys.issubset(metrics.keys())
print(f"[1e] evaluate_solution keys: {set(metrics.keys())} ✓")
print(f"     sample metrics: acc={metrics['accuracy']:.3f}, rec={metrics['recall']:.3f}, f1={metrics['f1']:.3f}")

# 1-f. Xavier initialization shape
layer_sizes = [input_size, hidden_sizes, hidden_sizes, output_size]
pop_xavier = xavier_initialisation(5, layer_sizes)
assert len(pop_xavier) == 5
assert len(pop_xavier[0]) == expected_n
print(f"[1f] xavier_initialisation: 5 individuals × {len(pop_xavier[0])} params ✓")

# 1-g. GA smoke test (small)
best_sol, best_fit, history = genetic_algorithm(
    fitness_fn=fitness_fn,
    n_params=expected_n,
    pop_size=10,
    n_generations=3,
    elitism=2,
    selection_method='tournament',
    crossover_method='arithmetic',
    mutation_method='gaussian',
    mutation_rate=0.1,
    crossover_rate=0.8,
    init_method='random',
    input_size=input_size,
    hidden_size=hidden_sizes,
    output_size=output_size,
    verbose=False
)
assert len(best_sol) == expected_n
print(f"[1g] GA smoke test passed. best_fit={best_fit:.4f}, history={history} ✓")

print("\n✅ All sanity tests passed!")
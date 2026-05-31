"""
main_hidden.py
==============
Experiment runner for the Parkinson's Disease MLP weight optimisation
— **two hidden layers** variant (22 → 10 → 10 → 1).

Mirrors main_c.py in structure and output format so results from both
architectures are directly comparable.

Changes from main_c.py
-----------------------
* Imports from parkinsons_hidden / genetic_algorithm_hidden / pso_hidden.
* Architecture string printed as  22 → 10 → 10 → 1.
* n_params is 351 (instead of 241).
* GA and PSO calls pass hidden1_size / hidden2_size where relevant
  (only the GA's xavier init actually uses them; PSO is architecture-agnostic
  at the algorithm level, but the size is forwarded for the verbose header).
* A random baseline solution is evaluated first so the report can compare
  a random 2-layer network against the optimised one.

Usage
-----
    python main_hidden.py

Outputs
-------
  • Console: per-algorithm logs + comparison table
  • convergence_plot_hidden.png
  • results_summary_hidden.txt
"""

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from parkinsons_hidden import (
    fitness_function,
    evaluate_solution,
    compute_n_params,
    generate_solution,
    DEFAULT_INPUT_SIZE,
    DEFAULT_HIDDEN1_SIZE,
    DEFAULT_HIDDEN2_SIZE,
    DEFAULT_OUTPUT_SIZE,
)
from genetic_algorithm_hidden import genetic_algorithm
from pso_hidden import particle_swarm_optimisation

# ---------------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------------
SEED      = 42
POP_SIZE  = 50
N_ITER    = 100
DATA_PATH = 'parkinsons_preprocessed.csv'

# ---------------------------------------------------------------------------
# 1.  Load dataset
# ---------------------------------------------------------------------------
print("=" * 65)
print("  Parkinson's Disease — MLP Weight Optimisation")
print("  Two Hidden Layers  |  GA  vs  PSO  comparison")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=['status']).values.astype(float)
y  = df['status'].values.astype(int)

n_positive           = int((y == 1).sum())
n_negative           = int((y == 0).sum())
n_samples, n_features = X.shape

# CHANGED: compute_n_params for 2 hidden layers → 351
n_params = compute_n_params()

print(f"\n  Dataset       : {DATA_PATH}")
print(f"  Samples       : {n_samples}  ({n_features} features)")
print(f"  Positive (PD) : {n_positive}  ({n_positive/n_samples*100:.1f}%)")
print(f"  Negative      : {n_negative}  ({n_negative/n_samples*100:.1f}%)")
# CHANGED: architecture line now shows two hidden layers
print(f"\n  Architecture  : {DEFAULT_INPUT_SIZE} → {DEFAULT_HIDDEN1_SIZE}"
      f" → {DEFAULT_HIDDEN2_SIZE} → {DEFAULT_OUTPUT_SIZE}")
print(f"  Parameters    : {n_params}  (θ ∈ ℝ^{n_params})")
print(f"\n  Pop / swarm   : {POP_SIZE}")
print(f"  Iterations    : {N_ITER}")
print(f"  Seed          : {SEED}")

baseline_recall   = n_positive / n_samples
baseline_accuracy = n_positive / n_samples
print(f"\n  Baseline (all-positive) recall   : {baseline_recall:.4f}")
print(f"  Baseline (all-positive) accuracy : {baseline_accuracy:.4f}")

# ---------------------------------------------------------------------------
# 1b.  Random baseline — evaluate a random 2-layer weight vector
#      (gives context: how well does an untrained network do?)
# ---------------------------------------------------------------------------
np.random.seed(SEED)
random_theta   = generate_solution(n_params)
random_metrics = evaluate_solution(random_theta, X, y)
print(f"\n  Random θ baseline:")
print(f"    Recall    : {random_metrics['recall']:.4f}")
print(f"    Accuracy  : {random_metrics['accuracy']:.4f}")
print(f"    F1        : {random_metrics['f1']:.4f}")

# ---------------------------------------------------------------------------
# 2.  Run Genetic Algorithm
# ---------------------------------------------------------------------------
print("\n" + "─" * 65)
print("  Running GA …")
print("─" * 65)

t0 = time.perf_counter()
ga_best_theta, ga_best_fitness, ga_history = genetic_algorithm(
    fitness_fn      = fitness_function,
    n_params        = n_params,
    fitness_args    = (X, y),
    pop_size        = POP_SIZE,
    n_generations   = N_ITER,
    crossover_rate  = 0.8,
    mutation_rate   = 0.05,
    sigma           = 0.1,
    tournament_size = 3,
    elitism         = 2,
    selection_method  = 'tournament',
    crossover_method  = 'arithmetic',
    mutation_method   = 'gaussian',
    init_method       = 'random',
    # CHANGED: pass both hidden layer sizes for Xavier init compatibility
    input_size        = DEFAULT_INPUT_SIZE,
    hidden1_size      = DEFAULT_HIDDEN1_SIZE,
    hidden2_size      = DEFAULT_HIDDEN2_SIZE,
    output_size       = DEFAULT_OUTPUT_SIZE,
    seed              = SEED,
    verbose           = True,
)
ga_time = time.perf_counter() - t0

# CHANGED: evaluate_solution now accepts hidden1_size and hidden2_size
ga_metrics = evaluate_solution(ga_best_theta, X, y)

# ---------------------------------------------------------------------------
# 3.  Run Particle Swarm Optimisation
# ---------------------------------------------------------------------------
print("\n" + "─" * 65)
print("  Running PSO …")
print("─" * 65)

t0 = time.perf_counter()
pso_best_theta, pso_best_fitness, pso_history = particle_swarm_optimisation(
    fitness_fn       = fitness_function,
    n_params         = n_params,
    fitness_args     = (X, y),
    n_particles      = POP_SIZE,
    n_iterations     = N_ITER,
    w                = 0.9,
    w_min            = 0.4,
    c1               = 2.0,
    c2               = 2.0,
    v_max_ratio      = 0.2,
    inertia_strategy = 'linear_decay',
    patience         = 20,
    # CHANGED: forward architecture args for verbose header
    input_size       = DEFAULT_INPUT_SIZE,
    hidden1_size     = DEFAULT_HIDDEN1_SIZE,
    hidden2_size     = DEFAULT_HIDDEN2_SIZE,
    output_size      = DEFAULT_OUTPUT_SIZE,
    seed             = SEED,
    verbose          = True,
)
pso_time = time.perf_counter() - t0

pso_metrics = evaluate_solution(pso_best_theta, X, y)

# ---------------------------------------------------------------------------
# 4.  Comparison table
# ---------------------------------------------------------------------------
def degenerate_flag(metrics):
    if metrics['n_pred_neg'] == 0:
        return '  ⚠ DEGENERATE (all-positive predictions)'
    return ''


header = (f"\n{'=' * 65}\n"
          f"  Results Comparison — Two Hidden Layers\n"
          f"  Architecture: {DEFAULT_INPUT_SIZE}→{DEFAULT_HIDDEN1_SIZE}"
          f"→{DEFAULT_HIDDEN2_SIZE}→{DEFAULT_OUTPUT_SIZE}"
          f"  |  n_params={n_params}\n"
          f"{'=' * 65}")

rows = [
    ("Metric",           "GA",                            "PSO"),
    ("-" * 20,           "-" * 22,                        "-" * 22),
    ("Recall (fitness)",
     f"{ga_metrics['recall']:.4f}",
     f"{pso_metrics['recall']:.4f}"),
    ("Accuracy",
     f"{ga_metrics['accuracy']:.4f}",
     f"{pso_metrics['accuracy']:.4f}"),
    ("Precision",
     f"{ga_metrics['precision']:.4f}",
     f"{pso_metrics['precision']:.4f}"),
    ("F1 score",
     f"{ga_metrics['f1']:.4f}",
     f"{pso_metrics['f1']:.4f}"),
    ("Pred positive",
     f"{ga_metrics['n_pred_pos']} / {n_samples}",
     f"{pso_metrics['n_pred_pos']} / {n_samples}"),
    ("Pred negative",
     f"{ga_metrics['n_pred_neg']} / {n_samples}",
     f"{pso_metrics['n_pred_neg']} / {n_samples}"),
    ("Wall-clock time",
     f"{ga_time:.1f}s",
     f"{pso_time:.1f}s"),
    ("Best fitness",
     f"{ga_best_fitness:.6f}",
     f"{pso_best_fitness:.6f}"),
    ("Converged iter",
     str(ga_history.index(ga_best_fitness) + 1
         if ga_best_fitness in ga_history else N_ITER),
     str(pso_history.index(pso_best_fitness) + 1
         if pso_best_fitness in pso_history else N_ITER)),
]

table_lines = [header]
for r in rows:
    table_lines.append(f"  {r[0]:<22} {r[1]:<22} {r[2]:<22}")

ga_flag  = degenerate_flag(ga_metrics)
pso_flag = degenerate_flag(pso_metrics)
if ga_flag:
    table_lines.append(f"\n  GA  {ga_flag}")
if pso_flag:
    table_lines.append(f"\n  PSO {pso_flag}")
table_lines.append("=" * 65)
table_str = "\n".join(table_lines)
print(table_str)

# Save to text
with open('results_summary_hidden.txt', 'w') as f:
    f.write(table_str + "\n\n")
    f.write(f"Random baseline metrics:\n")
    for k, v in random_metrics.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nGA best θ (first 20 values):\n")
    f.write(str(ga_best_theta[:20].round(6)) + "\n\n")
    f.write("PSO best θ (first 20 values):\n")
    f.write(str(pso_best_theta[:20].round(6)) + "\n")
print("\n  Saved: results_summary_hidden.txt")

# ---------------------------------------------------------------------------
# 5.  Convergence plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#fafafa')

iterations = list(range(1, N_ITER + 1))

# ── Left: convergence curves ─────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#ffffff')
ax.plot(iterations, ga_history,  color='#4A7CC3', lw=2, label='GA',  alpha=0.9)
ax.plot(iterations, pso_history, color='#E07B3A', lw=2, label='PSO', alpha=0.9)

ga_best_iter  = int(np.argmax(ga_history))
pso_best_iter = int(np.argmax(pso_history))
ax.scatter([ga_best_iter + 1],  [ga_history[ga_best_iter]],
           color='#4A7CC3', s=80, zorder=5)
ax.scatter([pso_best_iter + 1], [pso_history[pso_best_iter]],
           color='#E07B3A', s=80, zorder=5)

ax.axhline(y=baseline_recall, color='#999', lw=1, ls='--',
           label=f'All-positive baseline ({baseline_recall:.2f})')
ax.set_xlabel('Iteration / Generation', fontsize=11)
ax.set_ylabel('Best Fitness (Recall)', fontsize=11)
# CHANGED: title notes the two-layer architecture
ax.set_title('Convergence Curves: GA vs PSO\n'
             f'[{DEFAULT_INPUT_SIZE}→{DEFAULT_HIDDEN1_SIZE}'
             f'→{DEFAULT_HIDDEN2_SIZE}→{DEFAULT_OUTPUT_SIZE}]',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(1, N_ITER); ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, ls=':')
for s in ax.spines.values(): s.set_linewidth(0.5)

# ── Right: metric bar chart ───────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#ffffff')

metric_names = ['Recall', 'Accuracy', 'Precision', 'F1 Score']
ga_vals  = [ga_metrics['recall'],  ga_metrics['accuracy'],
            ga_metrics['precision'], ga_metrics['f1']]
pso_vals = [pso_metrics['recall'], pso_metrics['accuracy'],
            pso_metrics['precision'], pso_metrics['f1']]

x     = np.arange(len(metric_names))
width = 0.35
bars_ga  = ax2.bar(x - width/2, ga_vals,  width,
                   color='#4A7CC3', alpha=0.85, label='GA',  zorder=3)
bars_pso = ax2.bar(x + width/2, pso_vals, width,
                   color='#E07B3A', alpha=0.85, label='PSO', zorder=3)

for bar in bars_ga:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
             f'{h:.3f}', ha='center', va='bottom', fontsize=8.5, color='#4A7CC3')
for bar in bars_pso:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
             f'{h:.3f}', ha='center', va='bottom', fontsize=8.5, color='#E07B3A')

ax2.set_xticks(x); ax2.set_xticklabels(metric_names, fontsize=10)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('Final Metrics: GA vs PSO', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.15); ax2.legend(fontsize=10)
ax2.grid(True, axis='y', alpha=0.3, ls=':')
ax2.axhline(y=baseline_accuracy, color='#999', lw=1, ls='--', zorder=2)
ax2.text(3.55, baseline_accuracy + 0.01, 'baseline',
         fontsize=8, color='#999', ha='right')
for s in ax2.spines.values(): s.set_linewidth(0.5)

plt.suptitle(
    f"Parkinson's Disease MLP — 2 Hidden Layers ({DEFAULT_INPUT_SIZE}"
    f"→{DEFAULT_HIDDEN1_SIZE}→{DEFAULT_HIDDEN2_SIZE}→{DEFAULT_OUTPUT_SIZE},"
    f" n_params={n_params})\n"
    f"pop/swarm={POP_SIZE}, iterations={N_ITER}, seed={SEED}",
    fontsize=10, y=1.02, color='#333',
)
plt.tight_layout()
plt.savefig('convergence_plot_hidden.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved: convergence_plot_hidden.png")
print("\nDone.\n")

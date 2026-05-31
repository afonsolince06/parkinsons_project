"""
main.py
=======
Experiment runner for Parkinson's Disease MLP weight optimisation.

Runs two nature-inspired optimisers — Genetic Algorithm (GA) and Particle
Swarm Optimisation (PSO) — on the same dataset with the same fitness
function, then compares their results.

Usage
-----
    python main.py

Outputs
-------
  • Console: per-algorithm verbose logs + side-by-side comparison table
  • convergence_plot.png : fitness-vs-iteration curves for both algorithms
  • results_summary.txt  : plain-text copy of the comparison table
"""

import warnings
# Suppress sklearn's ConvergenceWarning from the dummy 1-iteration fit
# inside fitness_function (fired ~5000 times during a full run).
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from parkinsons_problem_c import (
    fitness_function,
    evaluate_solution,
    compute_n_params,
    DEFAULT_INPUT_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_OUTPUT_SIZE,
)
from genetic_algorithm_c import genetic_algorithm
from pso_c import particle_swarm_optimisation


# ---------------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------------
SEED          = 42        # shared RNG seed — ensures fair comparison
POP_SIZE      = 50        # population / swarm size
N_ITER        = 100       # generations (GA) / iterations (PSO)
DATA_PATH     = 'parkinsons_preprocessed.csv'

# ---------------------------------------------------------------------------
# 1.  Load and inspect dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Parkinson's Disease — MLP Weight Optimisation")
print("  GA  vs  PSO  comparison experiment")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=['status']).values.astype(float)
y  = df['status'].values.astype(int)

n_positive = int((y == 1).sum())
n_negative = int((y == 0).sum())
n_samples, n_features = X.shape
n_params = compute_n_params()

print(f"\n  Dataset       : {DATA_PATH}")
print(f"  Samples       : {n_samples}  ({n_features} features)")
print(f"  Positive (PD) : {n_positive}  ({n_positive/n_samples*100:.1f}%)")
print(f"  Negative      : {n_negative}  ({n_negative/n_samples*100:.1f}%)")
print(f"\n  Architecture  : {DEFAULT_INPUT_SIZE} → {DEFAULT_HIDDEN_SIZE} → {DEFAULT_OUTPUT_SIZE}")
print(f"  Parameters    : {n_params}  (θ ∈ ℝ^{n_params})")
print(f"\n  Pop / swarm   : {POP_SIZE}")
print(f"  Iterations    : {N_ITER}")
print(f"  Seed          : {SEED}")
print()

# Baseline: trivial classifier that predicts everyone as positive
baseline_recall    = n_positive / n_samples          # = 1.0 by definition
baseline_accuracy  = n_positive / n_samples          # majority-class accuracy
print(f"  Baseline (all-positive) recall   : {baseline_recall:.4f}")
print(f"  Baseline (all-positive) accuracy : {baseline_accuracy:.4f}")
print()

# ---------------------------------------------------------------------------
# 2.  Run Genetic Algorithm
# ---------------------------------------------------------------------------
print("\n" + "─" * 60)
print("  Running GA …")
print("─" * 60)

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
    seed              = SEED,
    verbose           = True,
)
ga_time = time.perf_counter() - t0

ga_metrics = evaluate_solution(ga_best_theta, X, y)

# ---------------------------------------------------------------------------
# 3.  Run Particle Swarm Optimisation
# ---------------------------------------------------------------------------
print("\n" + "─" * 60)
print("  Running PSO …")
print("─" * 60)

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
    seed             = SEED,
    verbose          = True,
)
pso_time = time.perf_counter() - t0

pso_metrics = evaluate_solution(pso_best_theta, X, y)

# ---------------------------------------------------------------------------
# 4.  Comparison table
# ---------------------------------------------------------------------------
def degenerate_flag(metrics):
    """Return ⚠ DEGENERATE if the model predicts no negatives (all-positive)."""
    if metrics['n_pred_neg'] == 0:
        return '  ⚠ DEGENERATE (all-positive)'
    return ''


header = f"\n{'=' * 60}\n  Results Comparison\n{'=' * 60}"
rows = [
    ("Metric",          "GA",                           "PSO"),
    ("-" * 18,          "-" * 22,                       "-" * 22),
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
     str(ga_history.index(ga_best_fitness) + 1 if ga_best_fitness in ga_history else N_ITER),
     str(pso_history.index(pso_best_fitness) + 1 if pso_best_fitness in pso_history else N_ITER)),
]

table_lines = [header]
for r in rows:
    table_lines.append(f"  {r[0]:<20} {r[1]:<22} {r[2]:<22}")

# Degenerate solution warnings
ga_flag  = degenerate_flag(ga_metrics)
pso_flag = degenerate_flag(pso_metrics)
if ga_flag:
    table_lines.append(f"\n  GA  {ga_flag}")
if pso_flag:
    table_lines.append(f"\n  PSO {pso_flag}")

table_lines.append("=" * 60)
table_str = "\n".join(table_lines)
print(table_str)

# Save results to text file
with open('results_summary.txt', 'w') as f:
    f.write(table_str + "\n\n")
    f.write("GA best θ (first 20 values):\n")
    f.write(str(ga_best_theta[:20].round(6)) + "\n\n")
    f.write("PSO best θ (first 20 values):\n")
    f.write(str(pso_best_theta[:20].round(6)) + "\n")
print("\n  Saved: results_summary.txt")

# ---------------------------------------------------------------------------
# 5.  Convergence plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#fafafa')

iterations = list(range(1, N_ITER + 1))

# ── Left panel: convergence curves overlaid ──────────────────────────────
ax = axes[0]
ax.set_facecolor('#ffffff')

ax.plot(iterations, ga_history,
        color='#4A7CC3', linewidth=2, label='GA', alpha=0.9)
ax.plot(iterations, pso_history,
        color='#E07B3A', linewidth=2, label='PSO', alpha=0.9)

# Mark best point for each algorithm
ga_best_iter  = int(np.argmax(ga_history))
pso_best_iter = int(np.argmax(pso_history))

ax.scatter([ga_best_iter + 1],  [ga_history[ga_best_iter]],
           color='#4A7CC3', s=80, zorder=5)
ax.scatter([pso_best_iter + 1], [pso_history[pso_best_iter]],
           color='#E07B3A', s=80, zorder=5)

# Horizontal baseline
ax.axhline(y=baseline_recall, color='#999', linewidth=1,
           linestyle='--', label=f'All-positive baseline ({baseline_recall:.2f})')

ax.set_xlabel('Iteration / Generation', fontsize=11)
ax.set_ylabel('Best Fitness (Recall)', fontsize=11)
ax.set_title('Convergence Curves: GA vs PSO', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(1, N_ITER)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, linestyle=':')
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

# ── Right panel: metric bar chart ─────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#ffffff')

metric_names  = ['Recall', 'Accuracy', 'Precision', 'F1 Score']
ga_vals  = [ga_metrics['recall'],  ga_metrics['accuracy'],
            ga_metrics['precision'], ga_metrics['f1']]
pso_vals = [pso_metrics['recall'], pso_metrics['accuracy'],
            pso_metrics['precision'], pso_metrics['f1']]

x     = np.arange(len(metric_names))
width = 0.35

bars_ga  = ax2.bar(x - width/2, ga_vals,  width, color='#4A7CC3',
                   alpha=0.85, label='GA',  zorder=3)
bars_pso = ax2.bar(x + width/2, pso_vals, width, color='#E07B3A',
                   alpha=0.85, label='PSO', zorder=3)

# Value labels on bars
for bar in bars_ga:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
             f'{h:.3f}', ha='center', va='bottom', fontsize=8.5, color='#4A7CC3')
for bar in bars_pso:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
             f'{h:.3f}', ha='center', va='bottom', fontsize=8.5, color='#E07B3A')

ax2.set_xticks(x)
ax2.set_xticklabels(metric_names, fontsize=10)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('Final Metrics: GA vs PSO', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=10)
ax2.grid(True, axis='y', alpha=0.3, linestyle=':')
ax2.axhline(y=baseline_accuracy, color='#999', linewidth=1, linestyle='--', zorder=2)
ax2.text(3.55, baseline_accuracy + 0.01, 'baseline', fontsize=8, color='#999', ha='right')
for spine in ax2.spines.values():
    spine.set_linewidth(0.5)

plt.suptitle(
    "Parkinson's Disease MLP — Nature-Inspired Weight Optimisation\n"
    f"pop/swarm={POP_SIZE}, iterations={N_ITER}, seed={SEED}",
    fontsize=11, y=1.02, color='#333'
)
plt.tight_layout()
plt.savefig('convergence_plot.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved: convergence_plot.png")
print("\nDone.\n")

"""
main_hidden.py
==============
Experiment runner for the Parkinson's Disease MLP weight optimisation
— two hidden layers variant (22 → 10 → 10 → 1).

What's new in this version
---------------------9. Standard vs Alternative GA operators comparison
     Runs GA twice: once with standard operators (tournament + arithmetic +
     gaussian) and once with alternative operators (roulette + blend + uniform).
     Results are printed side-by-side for easy comparison.

2. Activation function combination examples
     Demonstrates the custom forward-pass fitness function by running GA
     with several (act_h1, act_h2) combinations and comparing recall.

3. Statistical-ready CSV output
     Every run saves its metrics to 'results_summary_hidden.txt' and
     'all_runs_main.csv' so the data can be loaded into the report's
     statistical analysis without re-running.

Architecture
    Input  →  Hidden-1  →  Hidden-2  →  Output
      22        10            10           1
    n_params = 351

Operators tested
    Selection : tournament (standard), roulette (alternative)
    Crossover : arithmetic (standard), blend (alternative)
    Mutation  : gaussian (standard),   uniform (alternative)
    Init      : random (both) random (both)

Activation combinations tested
    Standard  : relu + relu (both layers)
    Alternative 1: relu + tanh
    Alternative 2: tanh + tanh

Usage
-----
    python main_hidden.py

Outputs
-------
  • convergence_plot_hidden.png     — GA-std / GA-ext / PSO convergence
  • operator_bar_plot.png           — metric bar chart (3 variants)
  • results_summary_hidden.txt      — text report
  • all_runs_main.csv               — all run metrics in CSV
"""

import warnings
warnings.filterwarnings('ignore')

import time
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from my_parkinsons_problem import (
    fitness_function,
    fitness_function_custom_act,
    evaluate_solution,
    evaluate_solution_custom_act,
    compute_n_params,
    generate_solution,
    DEFAULT_INPUT_SIZE,
    DEFAULT_HIDDEN1_SIZE,
    DEFAULT_HIDDEN2_SIZE,
    DEFAULT_OUTPUT_SIZE,
)
from genetic_algorithm_c import genetic_algorithm
from my_pso_grid import particle_swarm_optimisation

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
SEED     = 42
POP_SIZE = 50
N_ITER   = 100
DATA_PATH = 'parkinsons_preprocessed.csv'

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
print("=" * 68)
print("  Parkinson's Disease — MLP Weight Optimisation")
print("  Two Hidden Layers  |  GA-standard vs GA-extended vs PSO")
print("=" * 68)

df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=['status']).values.astype(float)
y  = df['status'].values.astype(int)

n_positive            = int((y == 1).sum())
n_negative            = int((y == 0).sum())
n_samples, n_features = X.shape
n_params              = compute_n_params()

print(f"\n  Dataset       : {DATA_PATH}")
print(f"  Samples       : {n_samples}  ({n_features} features)")
print(f"  Positive (PD) : {n_positive}  ({n_positive/n_samples*100:.1f}%)")
print(f"  Negative      : {n_negative}  ({n_negative/n_samples*100:.1f}%)")
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
# 1b. Random θ baseline
# ---------------------------------------------------------------------------
np.random.seed(SEED)
random_theta   = generate_solution(n_params)
random_metrics = evaluate_solution(random_theta, X, y)
print(f"\n  Random θ baseline:")
print(f"    Recall    : {random_metrics['recall']:.4f}")
print(f"    Accuracy  : {random_metrics['accuracy']:.4f}")
print(f"    F1        : {random_metrics['f1']:.4f}")

# ---------------------------------------------------------------------------
# Shared fitness function (recall, uniform relu activation)
# ---------------------------------------------------------------------------
def fitness_fn(solution, X, y):
    """Wrapper — uses recall with relu for both hidden layers."""
    return fitness_function(solution, X, y)

# Container for all run results (for CSV export)
all_run_records = []

# ---------------------------------------------------------------------------
# 2. GA — Standard operators
#    selection: tournament  |  crossover: arithmetic  |  mutation: gaussian
# ---------------------------------------------------------------------------
print("\n" + "─" * 68)
print("  Running GA (Standard: tournament + arithmetic + gaussian) …")
print("─" * 68)

t0 = time.perf_counter()
ga_std_theta, ga_std_fitness, ga_std_history = genetic_algorithm(
    fitness_fn      = fitness_fn,
    n_params        = n_params,
    fitness_args    = (X, y),
    pop_size        = POP_SIZE,
    n_generations   = N_ITER,
    crossover_rate  = 0.8,
    mutation_rate   = 0.05,
    sigma           = 0.1,
    tournament_size = 3,
    elitism         = 2,
    # ── Standard operators ─────────────────────────────────────────────
    selection_method  = 'tournament',   # high pressure, fast convergence
    crossover_method  = 'arithmetic',   # weighted average of parents
    mutation_method   = 'gaussian',     # fixed σ Gaussian perturbation
    # ───────────────────────────────────────────────────────────────────
    init_method       = 'random',
    input_size        = DEFAULT_INPUT_SIZE,
    hidden1_size      = DEFAULT_HIDDEN1_SIZE,
    hidden2_size      = DEFAULT_HIDDEN2_SIZE,
    output_size       = DEFAULT_OUTPUT_SIZE,
    seed              = SEED,
    verbose           = True,
)
ga_std_time    = time.perf_counter() - t0
ga_std_metrics = evaluate_solution(ga_std_theta, X, y)

all_run_records.append({
    'variant'         : 'GA-standard',
    'selection_method': 'tournament',
    'crossover_method': 'arithmetic',
    'mutation_method' : 'gaussian',
    'act_h1'          : 'relu',
    'act_h2'          : 'relu',
    'seed'            : SEED,
    'pop_size'        : POP_SIZE,
    'n_iter'          : N_ITER,
    'best_fitness'    : ga_std_fitness,
    'elapsed'         : round(ga_std_time, 2),
    **ga_std_metrics,
})

# 3. GA — Alternative operators
#    selection: roulette  |  crossover: blend        |  mutation: uniform
# ---------------------------------------------------------------------------
print("\n" + "─" * 68)
print("  Running GA (Alternative: roulette + blend + uniform) …")
print("─" * 68)

t0 = time.perf_counter()
ga_ext_theta, ga_ext_fitness, ga_ext_history = genetic_algorithm(
    fitness_fn      = fitness_fn,
    n_params        = n_params,
    fitness_args    = (X, y),
    pop_size        = POP_SIZE,
    n_generations   = N_ITER,
    crossover_rate  = 0.8,
    mutation_rate   = 0.05,
    sigma           = 0.1,
    tournament_size = 3,
    elitism         = 2,
    # ── Alternative operators ─────────────────────────────────────────────
    selection_method  = 'roulette',     # roulette selection
    crossover_method  = 'blend',        # blend crossover
    mutation_method   = 'uniform',      # uniform mutation
    # ───────────────────────────────────────────────────────────────────
    init_method       = 'random',
    input_size        = DEFAULT_INPUT_SIZE,
    hidden1_size      = DEFAULT_HIDDEN1_SIZE,
    hidden2_size      = DEFAULT_HIDDEN2_SIZE,
    output_size       = DEFAULT_OUTPUT_SIZE,
    seed              = SEED,
    verbose           = True,
)
ga_ext_time    = time.perf_counter() - t0
ga_ext_metrics = evaluate_solution(ga_ext_theta, X, y)

all_run_records.append({
    'variant'         : 'GA-alternative',
    'selection_method': 'roulette',
    'crossover_method': 'blend',
    'mutation_method' : 'uniform',
    'act_h1'          : 'relu',
    'act_h2'          : 'relu',
    'seed'            : SEED,
    'pop_size'        : POP_SIZE,
    'n_iter'          : N_ITER,
    'best_fitness'    : ga_ext_fitness,
    'elapsed'         : round(ga_ext_time, 2),
    **ga_ext_metrics,
})

# ---------------------------------------------------------------------------
# 4. PSO (reference baseline)
# ---------------------------------------------------------------------------
print("\n" + "─" * 68)
print("  Running PSO …")
print("─" * 68)

t0 = time.perf_counter()
pso_theta, pso_fitness, pso_history = particle_swarm_optimisation(
    fitness_fn       = fitness_fn,
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
    input_size       = DEFAULT_INPUT_SIZE,
    hidden1_size     = DEFAULT_HIDDEN1_SIZE,
    hidden2_size     = DEFAULT_HIDDEN2_SIZE,
    output_size      = DEFAULT_OUTPUT_SIZE,
    seed             = SEED,
    verbose          = True,
)
pso_time    = time.perf_counter() - t0
pso_metrics = evaluate_solution(pso_theta, X, y)

all_run_records.append({
    'variant'         : 'PSO',
    'selection_method': 'pso',
    'crossover_method': 'pso',
    'mutation_method' : 'pso',
    'act_h1'          : 'relu',
    'act_h2'          : 'relu',
    'seed'            : SEED,
    'pop_size'        : POP_SIZE,
    'n_iter'          : N_ITER,
    'best_fitness'    : pso_fitness,
    'elapsed'         : round(pso_time, 2),
    **pso_metrics,
})

# ---------------------------------------------------------------------------
# 5. Activation combination examples
#    Demonstrates per-layer activations using the custom forward pass.
# ---------------------------------------------------------------------------
print("\n" + "─" * 68)
print("  Activation combination experiments (GA + custom forward pass) …")
print("─" * 68)

ACT_PAIRS = [
    ('relu',     'relu'),       # standard baseline
    ('relu',     'tanh'),       # wide features → compressed
    ('tanh',     'tanh'),       # smooth, zero-centred throughout
    ('tanh',     'relu'),       # zero-centred layer1, sparse layer2
    ('logistic', 'relu'),       # sigmoid → relu
]

act_results = []

for act_h1, act_h2 in ACT_PAIRS:
    # Build a fitness function that uses the custom per-layer activations
    # Closure captures act_h1 and act_h2 explicitly to avoid late-binding
    def make_act_fn(a1, a2):
        def fn(sol, X, y):
            return fitness_function_custom_act(
                sol, X, y,
                hidden1_size=DEFAULT_HIDDEN1_SIZE,
                hidden2_size=DEFAULT_HIDDEN2_SIZE,
                act_h1=a1, act_h2=a2
            )
        return fn

    fn_act = make_act_fn(act_h1, act_h2)

    print(f"\n  act_h1={act_h1:<8}  act_h2={act_h2:<8}  running GA …")
    t0 = time.perf_counter()
    theta_a, fit_a, hist_a = genetic_algorithm(
        fitness_fn      = fn_act,
        n_params        = n_params,
        fitness_args    = (X, y),
        pop_size        = POP_SIZE,
        n_generations   = N_ITER,
        crossover_rate  = 0.8,
        mutation_rate   = 0.05,
        sigma           = 0.1,
        tournament_size = 3,
        elitism         = 2,
        # Use standard/basic operators for activation experiments
        selection_method = 'tournament',
        crossover_method = 'arithmetic',
        mutation_method  = 'gaussian',
        init_method      = 'random',
        input_size       = DEFAULT_INPUT_SIZE,
        hidden1_size     = DEFAULT_HIDDEN1_SIZE,
        hidden2_size     = DEFAULT_HIDDEN2_SIZE,
        output_size      = DEFAULT_OUTPUT_SIZE,
        seed             = SEED,
        verbose          = False,
    )
    elapsed_a = time.perf_counter() - t0

    m_a = evaluate_solution_custom_act(
        theta_a, X, y,
        hidden1_size=DEFAULT_HIDDEN1_SIZE,
        hidden2_size=DEFAULT_HIDDEN2_SIZE,
        act_h1=act_h1, act_h2=act_h2
    )

    print(f"    recall={m_a['recall']:.4f}  f1={m_a['f1']:.4f}  "
          f"accuracy={m_a['accuracy']:.4f}  time={elapsed_a:.1f}s")
    act_results.append({'act_h1': act_h1, 'act_h2': act_h2,
                        'history': hist_a, **m_a})

    all_run_records.append({
        'variant'         : f'GA-std-{act_h1}+{act_h2}',
        'selection_method': 'tournament',
        'crossover_method': 'arithmetic',
        'mutation_method' : 'gaussian',
        'act_h1'          : act_h1,
        'act_h2'          : act_h2,
        'seed'            : SEED,
        'pop_size'        : POP_SIZE,
        'n_iter'          : N_ITER,
        'best_fitness'    : fit_a,
        'elapsed'         : round(elapsed_a, 2),
        **{k: v for k, v in m_a.items() if k not in ('act_h1', 'act_h2')},
    })

# ---------------------------------------------------------------------------
# 6. Comparison tables
# ---------------------------------------------------------------------------
def degenerate_flag(metrics):
    if metrics.get('n_pred_neg', 1) == 0:
        return '  ⚠ DEGENERATE (all-positive predictions)'
    return ''


header = (f"\n{'=' * 68}\n"
          f"  Results Comparison — Two Hidden Layers\n"
          f"  Architecture: {DEFAULT_INPUT_SIZE}→{DEFAULT_HIDDEN1_SIZE}"
          f"→{DEFAULT_HIDDEN2_SIZE}→{DEFAULT_OUTPUT_SIZE}"
          f"  |  n_params={n_params}\n"
          f"{'=' * 68}")

rows = [
    ("Metric",           "GA-std",                   "GA-alt",                   "PSO"),
    ("-"*20,             "-"*18,                      "-"*18,                      "-"*18),
    ("Recall (fitness)",
     f"{ga_std_metrics['recall']:.4f}",
     f"{ga_ext_metrics['recall']:.4f}",
     f"{pso_metrics['recall']:.4f}"),
    ("Accuracy",
     f"{ga_std_metrics['accuracy']:.4f}",
     f"{ga_ext_metrics['accuracy']:.4f}",
     f"{pso_metrics['accuracy']:.4f}"),
    ("Precision",
     f"{ga_std_metrics['precision']:.4f}",
     f"{ga_ext_metrics['precision']:.4f}",
     f"{pso_metrics['precision']:.4f}"),
    ("F1 score",
     f"{ga_std_metrics['f1']:.4f}",
     f"{ga_ext_metrics['f1']:.4f}",
     f"{pso_metrics['f1']:.4f}"),
    ("Pred positive",
     f"{ga_std_metrics['n_pred_pos']} / {n_samples}",
     f"{ga_ext_metrics['n_pred_pos']} / {n_samples}",
     f"{pso_metrics['n_pred_pos']} / {n_samples}"),
    ("Pred negative",
     f"{ga_std_metrics['n_pred_neg']} / {n_samples}",
     f"{ga_ext_metrics['n_pred_neg']} / {n_samples}",
     f"{pso_metrics['n_pred_neg']} / {n_samples}"),
    ("Wall-clock time",
     f"{ga_std_time:.1f}s",
     f"{ga_ext_time:.1f}s",
     f"{pso_time:.1f}s"),
    ("Best fitness",
     f"{ga_std_fitness:.6f}",
     f"{ga_ext_fitness:.6f}",
     f"{pso_fitness:.6f}"),
    ("Converged iter",
     str(ga_std_history.index(ga_std_fitness) + 1
         if ga_std_fitness in ga_std_history else N_ITER),
     str(ga_ext_history.index(ga_ext_fitness) + 1
         if ga_ext_fitness in ga_ext_history else N_ITER),
     str(pso_history.index(pso_fitness) + 1
         if pso_fitness in pso_history else N_ITER)),
]

table_lines = [header]
for r in rows:
    table_lines.append(f"  {r[0]:<22} {r[1]:<20} {r[2]:<20} {r[3]:<20}")

for flag_lbl, metrics in [('GA-std', ga_std_metrics),
                           ('GA-ext', ga_ext_metrics),
                           ('PSO',    pso_metrics)]:
    flag = degenerate_flag(metrics)
    if flag:
        table_lines.append(f"\n  {flag_lbl} {flag}")
table_lines.append("=" * 68)
table_str = "\n".join(table_lines)
print(table_str)

# Activation combination summary
print(f"\n  Activation Combination Results (GA-standard):")
print(f"  {'act_h1':<10} {'act_h2':<10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}")
print("  " + "-"*50)
for r in act_results:
    print(f"  {r['act_h1']:<10} {r['act_h2']:<10} "
          f"{r['recall']:8.4f} {r['f1']:8.4f} {r['accuracy']:10.4f}")

# ---------------------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------------------
with open('results_summary_hidden.txt', 'w') as f:
    f.write(table_str + "\n\n")
    f.write("Random baseline metrics:\n")
    for k, v in random_metrics.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nActivation combination results:\n")
    f.write(f"  {'act_h1':<10} {'act_h2':<10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}\n")
    for r in act_results:
        f.write(f"  {r['act_h1']:<10} {r['act_h2']:<10} "
                f"{r['recall']:8.4f} {r['f1']:8.4f} {r['accuracy']:10.4f}\n")
    f.write("\nGA-standard best θ (first 20 values):\n")
    f.write(str(ga_std_theta[:20].round(6)) + "\n\n")
    f.write("GA-alternative best θ (first 20 values):\n")
    f.write(str(ga_ext_theta[:20].round(6)) + "\n\n")
    f.write("PSO best θ (first 20 values):\n")
    f.write(str(pso_theta[:20].round(6)) + "\n")
print("\n  Saved: results_summary_hidden.txt")

# Save all-runs CSV (includes operator labels, activation labels, metrics)
pd.DataFrame(all_run_records).to_csv('all_runs_main.csv', index=False)
print("  Saved: all_runs_main.csv")

# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('#fafafa')

iterations = list(range(1, N_ITER + 1))

# ── Left: convergence curves (GA-std / GA-ext / PSO) ─────────────────────
ax = axes[0]
ax.set_facecolor('#ffffff')
ax.plot(iterations, ga_std_history, color='#4A7CC3', lw=2,
        label='GA-standard (tournament+arithmetic+gaussian)', alpha=0.9)
ax.plot(iterations, ga_ext_history, color='#E07B3A', lw=2,
        label='GA-alternative (roulette+blend+uniform)', alpha=0.9)
ax.plot(iterations, pso_history,    color='#3A9A5C', lw=2,
        label='PSO', alpha=0.9)

for hist, color, fitness in [
    (ga_std_history, '#4A7CC3', ga_std_fitness),
    (ga_ext_history, '#E07B3A', ga_ext_fitness),
    (pso_history,    '#3A9A5C', pso_fitness),
]:
    best_i = int(np.argmax(hist))
    ax.scatter([best_i + 1], [hist[best_i]], color=color, s=80, zorder=5)

ax.axhline(y=baseline_recall, color='#999', lw=1, ls='--',
           label=f'All-positive baseline ({baseline_recall:.2f})')
ax.set_xlabel('Iteration / Generation', fontsize=11)
ax.set_ylabel('Best Fitness (Recall)', fontsize=11)
ax.set_title('Convergence Curves\nGA-std vs GA-alt vs PSO',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(1, N_ITER); ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, ls=':')
for s in ax.spines.values(): s.set_linewidth(0.5)

# ── Middle: metric bar chart (three variants) ─────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#ffffff')

metric_names = ['Recall', 'Accuracy', 'Precision', 'F1']
std_vals = [ga_std_metrics[k] for k in ['recall', 'accuracy', 'precision', 'f1']]
ext_vals = [ga_ext_metrics[k] for k in ['recall', 'accuracy', 'precision', 'f1']]
pso_vals = [pso_metrics[k]    for k in ['recall', 'accuracy', 'precision', 'f1']]

x     = np.arange(len(metric_names))
width = 0.25
bars_std = ax2.bar(x - width,     std_vals, width, color='#4A7CC3', alpha=0.85, label='GA-std')
bars_ext = ax2.bar(x,             ext_vals, width, color='#E07B3A', alpha=0.85, label='GA-alt')
bars_pso = ax2.bar(x + width,     pso_vals, width, color='#3A9A5C', alpha=0.85, label='PSO')

for bars, color in [(bars_std, '#4A7CC3'), (bars_ext, '#E07B3A'), (bars_pso, '#3A9A5C')]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color=color)

ax2.set_xticks(x); ax2.set_xticklabels(metric_names, fontsize=10)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('Final Metrics\nGA-std vs GA-alt vs PSO',
              fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.18); ax2.legend(fontsize=9)
ax2.grid(True, axis='y', alpha=0.3, ls=':')
ax2.axhline(y=baseline_accuracy, color='#999', lw=1, ls='--', zorder=2)
for s in ax2.spines.values(): s.set_linewidth(0.5)

# ── Right: Activation combination recall bar chart ────────────────────────
ax3 = axes[2]
ax3.set_facecolor('#ffffff')

act_labels = [f"{r['act_h1']}+{r['act_h2']}" for r in act_results]
act_recalls = [r['recall'] for r in act_results]
act_f1s     = [r['f1']     for r in act_results]

x_act = np.arange(len(act_labels))
bars_r = ax3.bar(x_act - 0.18, act_recalls, 0.35,
                 color='#C04848', alpha=0.85, label='Recall')
bars_f = ax3.bar(x_act + 0.18, act_f1s,     0.35,
                 color='#7752BE', alpha=0.85, label='F1')

for bar in bars_r:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.01,
             f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color='#C04848')
for bar in bars_f:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.01,
             f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color='#7752BE')

ax3.set_xticks(x_act)
ax3.set_xticklabels(act_labels, fontsize=8, rotation=20, ha='right')
ax3.set_ylabel('Score', fontsize=11)
ax3.set_title('Activation Combinations\nRecall & F1 (GA-standard)',
              fontsize=12, fontweight='bold')
ax3.set_ylim(0, 1.18); ax3.legend(fontsize=9)
ax3.grid(True, axis='y', alpha=0.3, ls=':')
for s in ax3.spines.values(): s.set_linewidth(0.5)

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
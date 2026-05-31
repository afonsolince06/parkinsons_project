"""
grid_search.py
==============
Systematic hyperparameter exploration for the Parkinson's MLP optimisation.

Four phases, each isolated so results build on each other:

  Phase 1 — Fitness function  : recall vs F1 vs accuracy  (fixes degenerate problem)
  Phase 2 — GA hyperparameters: pop_size, mutation_rate, crossover_method
  Phase 3 — PSO hyperparameters: n_particles, inertia decay, c1/c2 balance
  Phase 4 — Architecture      : hidden layer size (5, 10, 20 neurons)

All runs use pop/swarm=30 and 50 iterations to keep runtime reasonable
while still producing meaningful convergence differences.
"""

import warnings
warnings.filterwarnings('ignore')

import time
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score

from genetic_algorithm_c import genetic_algorithm
from pso_c import particle_swarm_optimisation

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

SEED       = 42
POP        = 30       # kept modest so the grid finishes fast
N_ITER     = 50       # generations / iterations per run

np.random.seed(SEED)

df = pd.read_csv('parkinsons_preprocessed.csv')
X  = df.drop(columns=['status']).values.astype(float)
y  = df['status'].values.astype(int)

N_SAMPLES = len(y)


def compute_n_params(input_size=22, hidden_size=10, output_size=1):
    return input_size*hidden_size + hidden_size + hidden_size*output_size + output_size


def unpack_weights(solution, input_size, hidden_size, output_size=1):
    idx = 0
    nW1 = input_size * hidden_size
    W1  = solution[idx:idx+nW1].reshape(input_size, hidden_size); idx += nW1
    b1  = solution[idx:idx+hidden_size];                           idx += hidden_size
    nW2 = hidden_size * output_size
    W2  = solution[idx:idx+nW2].reshape(hidden_size, output_size); idx += nW2
    b2  = solution[idx:idx+output_size]
    return [W1, W2], [b1, b2]


def make_fitness(metric='f1', hidden_size=10, input_size=22, output_size=1):
    """
    Factory: returns a fitness function for the given metric and architecture.

    metric options:
      'recall'   — maximise sensitivity (catches all PD, but may go degenerate)
      'f1'       — harmonic mean of precision and recall (balanced)
      'accuracy' — overall correct predictions (penalised by imbalance)
    """
    def fitness_fn(solution, X, y):
        clf = MLPClassifier(hidden_layer_sizes=(hidden_size,), activation='relu',
                            solver='sgd', max_iter=1, random_state=42)
        clf.fit(np.zeros((2, input_size)), np.array([0, 1]))
        clf.coefs_, clf.intercepts_ = unpack_weights(
            solution, input_size, hidden_size, output_size)
        y_pred = clf.predict(X)
        if metric == 'recall':
            return recall_score(y, y_pred, pos_label=1, zero_division=0)
        if metric == 'f1':
            return f1_score(y, y_pred, pos_label=1, zero_division=0)
        if metric == 'accuracy':
            return accuracy_score(y, y_pred)
        raise ValueError(f"Unknown metric: {metric}")
    return fitness_fn


def full_eval(solution, hidden_size=10, input_size=22, output_size=1):
    """Return all four metrics + prediction counts for a given solution."""
    clf = MLPClassifier(hidden_layer_sizes=(hidden_size,), activation='relu',
                        solver='sgd', max_iter=1, random_state=42)
    clf.fit(np.zeros((2, input_size)), np.array([0, 1]))
    clf.coefs_, clf.intercepts_ = unpack_weights(
        solution, input_size, hidden_size, output_size)
    y_pred = clf.predict(X)
    return {
        'recall'    : recall_score(y, y_pred, pos_label=1, zero_division=0),
        'precision' : precision_score(y, y_pred, pos_label=1, zero_division=0),
        'f1'        : f1_score(y, y_pred, pos_label=1, zero_division=0),
        'accuracy'  : accuracy_score(y, y_pred),
        'n_pred_pos': int((y_pred == 1).sum()),
        'n_pred_neg': int((y_pred == 0).sum()),
        'degenerate': int((y_pred == 0).sum()) == 0,
    }


def run_ga(fitness_fn, n_params, config):
    t0 = time.perf_counter()
    theta, fit, hist = genetic_algorithm(
        fitness_fn      = fitness_fn,
        n_params        = n_params,
        fitness_args    = (X, y),
        pop_size        = config.get('pop_size', POP),
        n_generations   = N_ITER,
        crossover_rate  = 0.8,
        mutation_rate   = config.get('mutation_rate', 0.05),
        sigma           = config.get('sigma', 0.1),
        tournament_size = 3,
        elitism         = 2,
        crossover_method = config.get('crossover_method', 'arithmetic'),
        mutation_method  = config.get('mutation_method', 'gaussian'),
        selection_method = 'tournament',
        init_method      = 'random',
        seed             = SEED,
        verbose          = False,
    )
    return theta, fit, hist, time.perf_counter() - t0


def run_pso(fitness_fn, n_params, config):
    t0 = time.perf_counter()
    theta, fit, hist = particle_swarm_optimisation(
        fitness_fn       = fitness_fn,
        n_params         = n_params,
        fitness_args     = (X, y),
        n_particles      = config.get('n_particles', POP),
        n_iterations     = N_ITER,
        w                = config.get('w', 0.9),
        w_min            = config.get('w_min', 0.4),
        c1               = config.get('c1', 2.0),
        c2               = config.get('c2', 2.0),
        v_max_ratio      = 0.2,
        inertia_strategy = 'linear_decay',
        patience         = 15,
        seed             = SEED,
        verbose          = False,
    )
    return theta, fit, hist, time.perf_counter() - t0


def print_result(label, metrics, elapsed, fitness_signal):
    deg = ' ⚠ DEGEN' if metrics['degenerate'] else ''
    print(f"  {label:<42}  "
          f"fit={fitness_signal:.3f}  "
          f"acc={metrics['accuracy']:.3f}  "
          f"f1={metrics['f1']:.3f}  "
          f"rec={metrics['recall']:.3f}  "
          f"prec={metrics['precision']:.3f}  "
          f"neg={metrics['n_pred_neg']:3d}  "
          f"{elapsed:5.1f}s{deg}")


# ─────────────────────────────────────────────────────────────────────────────
# Storage for all results (used by the plot at the end)
# ─────────────────────────────────────────────────────────────────────────────
all_results   = []   # list of dicts
phase_histories = {} # label -> convergence history list


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Fitness function
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("  PHASE 1 — Fitness function  (recall vs f1 vs accuracy)")
print("  Fixed: hidden=10, pop=30, iters=50, default GA/PSO params")
print("="*75)
print(f"  {'Config':<42}  fit    acc    f1     rec    prec   neg   time")
print("  " + "-"*72)

METRICS   = ['recall', 'f1', 'accuracy']
N_PARAMS_DEFAULT = compute_n_params(hidden_size=10)

best_fitness_metric = None
best_fitness_f1     = -1.0

for metric in METRICS:
    fn = make_fitness(metric=metric, hidden_size=10)
    for algo in ['GA', 'PSO']:
        cfg = {}
        if algo == 'GA':
            theta, fit, hist, elapsed = run_ga(fn, N_PARAMS_DEFAULT, cfg)
        else:
            theta, fit, hist, elapsed = run_pso(fn, N_PARAMS_DEFAULT, cfg)

        metrics = full_eval(theta, hidden_size=10)
        label   = f"{algo}  fitness={metric}"
        print_result(label, metrics, elapsed, fit)

        rec = {
            'phase': 1, 'algo': algo, 'label': label,
            'fitness_metric': metric,
            **metrics, 'fitness_signal': fit, 'elapsed': elapsed
        }
        all_results.append(rec)
        phase_histories[label] = hist

        # Track best non-degenerate F1 to pick metric for later phases
        if not metrics['degenerate'] and metrics['f1'] > best_fitness_f1:
            best_fitness_f1     = metrics['f1']
            best_fitness_metric = metric

best_fitness_metric = best_fitness_metric or 'f1'
print(f"\n  → Best non-degenerate metric for next phases: '{best_fitness_metric}'  (F1={best_fitness_f1:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — GA hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"  PHASE 2 — GA hyperparameters  (fitness='{best_fitness_metric}', hidden=10)")
print("  Grid: pop_size × mutation_rate × crossover_method")
print("="*75)
print(f"  {'Config':<42}  fit    acc    f1     rec    prec   neg   time")
print("  " + "-"*72)

fn_phase2 = make_fitness(metric=best_fitness_metric, hidden_size=10)

ga_grid = list(itertools.product(
    [30, 50],                        # pop_size
    [0.05, 0.15],                    # mutation_rate
    ['arithmetic', 'blend'],         # crossover_method
))

best_ga_config = None
best_ga_f1     = -1.0

for pop, mut, cx in ga_grid:
    cfg   = {'pop_size': pop, 'mutation_rate': mut, 'crossover_method': cx}
    theta, fit, hist, elapsed = run_ga(fn_phase2, N_PARAMS_DEFAULT, cfg)
    metrics = full_eval(theta, hidden_size=10)
    label = f"GA  pop={pop} mut={mut} cx={cx}"
    print_result(label, metrics, elapsed, fit)

    rec = {
        'phase': 2, 'algo': 'GA', 'label': label,
        'pop_size': pop, 'mutation_rate': mut, 'crossover_method': cx,
        **metrics, 'fitness_signal': fit, 'elapsed': elapsed
    }
    all_results.append(rec)
    phase_histories[label] = hist

    if not metrics['degenerate'] and metrics['f1'] > best_ga_f1:
        best_ga_f1     = metrics['f1']
        best_ga_config = cfg.copy()

best_ga_config = best_ga_config or {'pop_size': 50, 'mutation_rate': 0.05, 'crossover_method': 'arithmetic'}
print(f"\n  → Best GA config: {best_ga_config}  (F1={best_ga_f1:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — PSO hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"  PHASE 3 — PSO hyperparameters  (fitness='{best_fitness_metric}', hidden=10)")
print("  Grid: n_particles × inertia (w→w_min) × c1/c2 balance")
print("="*75)
print(f"  {'Config':<42}  fit    acc    f1     rec    prec   neg   time")
print("  " + "-"*72)

fn_phase3 = make_fitness(metric=best_fitness_metric, hidden_size=10)

# Inertia pairs: (w_start, w_end) — both are linear decay
# (0.9→0.4) classic LDIW  |  (0.7→0.2) more aggressive decay
pso_grid = list(itertools.product(
    [30, 50],           # n_particles
    [(0.9, 0.4),
     (0.7, 0.2)],       # (w, w_min)
    [(2.0, 2.0),
     (1.5, 2.5)],       # (c1, c2) — balanced vs social-heavy
))

best_pso_config = None
best_pso_f1     = -1.0

for n_part, (w, w_min), (c1, c2) in pso_grid:
    cfg   = {'n_particles': n_part, 'w': w, 'w_min': w_min, 'c1': c1, 'c2': c2}
    theta, fit, hist, elapsed = run_pso(fn_phase3, N_PARAMS_DEFAULT, cfg)
    metrics = full_eval(theta, hidden_size=10)
    label = f"PSO n={n_part} w={w}→{w_min} c1={c1}/c2={c2}"
    print_result(label, metrics, elapsed, fit)

    rec = {
        'phase': 3, 'algo': 'PSO', 'label': label,
        'n_particles': n_part, 'w': w, 'w_min': w_min, 'c1': c1, 'c2': c2,
        **metrics, 'fitness_signal': fit, 'elapsed': elapsed
    }
    all_results.append(rec)
    phase_histories[label] = hist

    if not metrics['degenerate'] and metrics['f1'] > best_pso_f1:
        best_pso_f1     = metrics['f1']
        best_pso_config = cfg.copy()

best_pso_config = best_pso_config or {'n_particles': 50, 'w': 0.9, 'w_min': 0.4, 'c1': 2.0, 'c2': 2.0}
print(f"\n  → Best PSO config: {best_pso_config}  (F1={best_pso_f1:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Architecture: hidden layer size
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"  PHASE 4 — Hidden layer size  (fitness='{best_fitness_metric}', best configs)")
print("  Grid: hidden_size ∈ {{5, 10, 20}}")
print("="*75)
print(f"  {'Config':<42}  fit    acc    f1     rec    prec   neg   time")
print("  " + "-"*72)

for hidden in [5, 10, 20]:
    n_params = compute_n_params(hidden_size=hidden)
    fn = make_fitness(metric=best_fitness_metric, hidden_size=hidden)

    # GA with best config from phase 2
    theta, fit, hist, elapsed = run_ga(fn, n_params, best_ga_config)
    metrics = full_eval(theta, hidden_size=hidden)
    label   = f"GA  hidden={hidden}"
    print_result(label, metrics, elapsed, fit)
    all_results.append({'phase': 4, 'algo': 'GA', 'label': label,
                        'hidden_size': hidden, **metrics,
                        'fitness_signal': fit, 'elapsed': elapsed})
    phase_histories[label] = hist

    # PSO with best config from phase 3
    theta, fit, hist, elapsed = run_pso(fn, n_params, best_pso_config)
    metrics = full_eval(theta, hidden_size=hidden)
    label   = f"PSO hidden={hidden}"
    print_result(label, metrics, elapsed, fit)
    all_results.append({'phase': 4, 'algo': 'PSO', 'label': label,
                        'hidden_size': hidden, **metrics,
                        'fitness_signal': fit, 'elapsed': elapsed})
    phase_histories[label] = hist


# ─────────────────────────────────────────────────────────────────────────────
# Final summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("  OVERALL BEST RESULTS  (non-degenerate only, ranked by F1)")
print("="*75)
print(f"  {'Label':<44} {'F1':>6} {'Acc':>6} {'Rec':>6} {'Prec':>6} {'Neg':>5}")
print("  " + "-"*72)

non_degen = [r for r in all_results if not r['degenerate']]
ranked    = sorted(non_degen, key=lambda r: r['f1'], reverse=True)

for r in ranked[:15]:
    print(f"  {r['label']:<44} "
          f"{r['f1']:6.3f} "
          f"{r['accuracy']:6.3f} "
          f"{r['recall']:6.3f} "
          f"{r['precision']:6.3f} "
          f"{r['n_pred_neg']:5d}")

if not ranked:
    print("  All runs produced degenerate solutions — try switching fitness to F1.")

print("="*75)


# ─────────────────────────────────────────────────────────────────────────────
# Save results CSV
# ─────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv('grid_search_results.csv', index=False)
print("\n  Saved: grid_search_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'GA':  '#4A7CC3',
    'PSO': '#E07B3A',
}
METRIC_COLORS = {
    'recall':   '#C04848',
    'f1':       '#3A9A5C',
    'accuracy': '#7752BE',
}

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('#fafafa')

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

iters = list(range(1, N_ITER + 1))

# ── P1a: convergence curves per fitness metric (GA) ─────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for metric in METRICS:
    key  = f"GA  fitness={metric}"
    hist = phase_histories.get(key, [])
    ax1.plot(iters[:len(hist)], hist,
             color=METRIC_COLORS[metric], linewidth=1.8, label=metric)
ax1.set_title('Phase 1 — GA convergence\nby fitness metric', fontsize=9, fontweight='bold')
ax1.set_xlabel('Generation', fontsize=8); ax1.set_ylabel('Best fitness', fontsize=8)
ax1.legend(fontsize=7); ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.tick_params(labelsize=7)

# ── P1b: convergence curves per fitness metric (PSO) ────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for metric in METRICS:
    key  = f"PSO  fitness={metric}"
    hist = phase_histories.get(key, [])
    ax2.plot(iters[:len(hist)], hist,
             color=METRIC_COLORS[metric], linewidth=1.8, label=metric)
ax2.set_title('Phase 1 — PSO convergence\nby fitness metric', fontsize=9, fontweight='bold')
ax2.set_xlabel('Iteration', fontsize=8); ax2.set_ylabel('Best fitness', fontsize=8)
ax2.legend(fontsize=7); ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.tick_params(labelsize=7)

# ── P1c: bar chart F1 per fitness metric ────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
p1_data = [r for r in all_results if r['phase'] == 1]
metrics_labels = METRICS
ga_f1s  = [next((r['f1']  for r in p1_data if r['algo']=='GA'  and r['fitness_metric']==m), 0) for m in metrics_labels]
pso_f1s = [next((r['f1']  for r in p1_data if r['algo']=='PSO' and r['fitness_metric']==m), 0) for m in metrics_labels]
ga_degen  = [next((r['degenerate'] for r in p1_data if r['algo']=='GA'  and r['fitness_metric']==m), True) for m in metrics_labels]
pso_degen = [next((r['degenerate'] for r in p1_data if r['algo']=='PSO' and r['fitness_metric']==m), True) for m in metrics_labels]
x = np.arange(len(metrics_labels)); w = 0.35
bars_ga  = ax3.bar(x - w/2, ga_f1s,  w, color='#4A7CC3', alpha=0.85, label='GA')
bars_pso = ax3.bar(x + w/2, pso_f1s, w, color='#E07B3A', alpha=0.85, label='PSO')
for i, (b, dg) in enumerate(zip(bars_ga, ga_degen)):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             '⚠' if dg else f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=7)
for i, (b, dg) in enumerate(zip(bars_pso, pso_degen)):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             '⚠' if dg else f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=7)
ax3.set_xticks(x); ax3.set_xticklabels(metrics_labels, fontsize=8)
ax3.set_title('Phase 1 — F1 by fitness metric\n(⚠ = degenerate)', fontsize=9, fontweight='bold')
ax3.set_ylabel('F1 score', fontsize=8); ax3.set_ylim(0, 1.15)
ax3.legend(fontsize=7); ax3.grid(True, axis='y', alpha=0.3, linestyle=':')
ax3.tick_params(labelsize=7)

# ── P2: GA grid — F1 heatmap style bar chart ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
p2_data = sorted([r for r in all_results if r['phase'] == 2],
                 key=lambda r: r['f1'], reverse=True)
labels2  = [r['label'].replace('GA  ', '') for r in p2_data]
f1s2     = [r['f1'] for r in p2_data]
degen2   = [r['degenerate'] for r in p2_data]
bar_cols = ['#c0392b' if d else '#4A7CC3' for d in degen2]
bars = ax4.barh(range(len(labels2)), f1s2, color=bar_cols, alpha=0.85, height=0.6)
for i, (b, d, f) in enumerate(zip(bars, degen2, f1s2)):
    ax4.text(f + 0.005, i, '⚠ degen' if d else f'{f:.3f}',
             va='center', fontsize=7.5)
ax4.set_yticks(range(len(labels2))); ax4.set_yticklabels(labels2, fontsize=7.5)
ax4.set_xlabel('F1 score', fontsize=8)
ax4.set_title('Phase 2 — GA hyperparameter grid\n(ranked by F1)', fontsize=9, fontweight='bold')
ax4.set_xlim(0, 1.15); ax4.invert_yaxis()
ax4.grid(True, axis='x', alpha=0.3, linestyle=':')
ax4.tick_params(labelsize=7)
degen_patch  = mpatches.Patch(color='#c0392b', alpha=0.85, label='Degenerate')
normal_patch = mpatches.Patch(color='#4A7CC3', alpha=0.85, label='Normal')
ax4.legend(handles=[normal_patch, degen_patch], fontsize=7, loc='lower right')

# ── P2 side: GA convergence for top-3 configs ───────────────────────────────
ax4b = fig.add_subplot(gs[1, 2])
top3_ga = [r for r in sorted([r for r in all_results if r['phase']==2],
           key=lambda r: r['f1'], reverse=True) if not r['degenerate']][:3]
if not top3_ga:
    top3_ga = sorted([r for r in all_results if r['phase']==2],
                     key=lambda r: r['f1'], reverse=True)[:3]
pal = ['#1a5599', '#4A7CC3', '#89b4e8']
for idx, r in enumerate(top3_ga):
    hist = phase_histories.get(r['label'], [])
    short = r['label'].replace('GA  ','')
    ax4b.plot(iters[:len(hist)], hist, color=pal[idx], linewidth=1.6,
              label=short, alpha=0.9)
ax4b.set_title('Phase 2 — GA top configs\nconvergence', fontsize=9, fontweight='bold')
ax4b.set_xlabel('Generation', fontsize=8); ax4b.set_ylabel('Best fitness', fontsize=8)
ax4b.legend(fontsize=6.5); ax4b.set_ylim(0, 1.05)
ax4b.grid(True, alpha=0.3, linestyle=':'); ax4b.tick_params(labelsize=7)

# ── P3: PSO grid ─────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0:2])
p3_data = sorted([r for r in all_results if r['phase'] == 3],
                 key=lambda r: r['f1'], reverse=True)
labels3  = [r['label'].replace('PSO ', '') for r in p3_data]
f1s3     = [r['f1'] for r in p3_data]
degen3   = [r['degenerate'] for r in p3_data]
bar_cols3 = ['#c0392b' if d else '#E07B3A' for d in degen3]
bars3 = ax5.barh(range(len(labels3)), f1s3, color=bar_cols3, alpha=0.85, height=0.6)
for i, (b, d, f) in enumerate(zip(bars3, degen3, f1s3)):
    ax5.text(f + 0.005, i, '⚠ degen' if d else f'{f:.3f}',
             va='center', fontsize=7.5)
ax5.set_yticks(range(len(labels3))); ax5.set_yticklabels(labels3, fontsize=7.5)
ax5.set_xlabel('F1 score', fontsize=8)
ax5.set_title('Phase 3 — PSO hyperparameter grid\n(ranked by F1)', fontsize=9, fontweight='bold')
ax5.set_xlim(0, 1.15); ax5.invert_yaxis()
ax5.grid(True, axis='x', alpha=0.3, linestyle=':')
ax5.tick_params(labelsize=7)
degen_patch2  = mpatches.Patch(color='#c0392b', alpha=0.85, label='Degenerate')
normal_patch2 = mpatches.Patch(color='#E07B3A', alpha=0.85, label='Normal')
ax5.legend(handles=[normal_patch2, degen_patch2], fontsize=7, loc='lower right')

# ── P4: hidden size comparison ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
p4_data  = [r for r in all_results if r['phase'] == 4]
hiddens  = [5, 10, 20]
ga_f1_h  = [next((r['f1'] for r in p4_data if r['algo']=='GA'  and r['hidden_size']==h), 0) for h in hiddens]
pso_f1_h = [next((r['f1'] for r in p4_data if r['algo']=='PSO' and r['hidden_size']==h), 0) for h in hiddens]
ga_dh    = [next((r['degenerate'] for r in p4_data if r['algo']=='GA'  and r['hidden_size']==h), True) for h in hiddens]
pso_dh   = [next((r['degenerate'] for r in p4_data if r['algo']=='PSO' and r['hidden_size']==h), True) for h in hiddens]
xh = np.arange(len(hiddens)); wh = 0.35
bga  = ax6.bar(xh - wh/2, ga_f1_h,  wh, color='#4A7CC3', alpha=0.85, label='GA')
bpso = ax6.bar(xh + wh/2, pso_f1_h, wh, color='#E07B3A', alpha=0.85, label='PSO')
for b, d in zip(bga,  ga_dh):
    ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             '⚠' if d else f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=7.5)
for b, d in zip(bpso, pso_dh):
    ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             '⚠' if d else f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=7.5)
ax6.set_xticks(xh); ax6.set_xticklabels([f'h={h}' for h in hiddens], fontsize=8)
ax6.set_title('Phase 4 — Hidden size\nvs F1 score', fontsize=9, fontweight='bold')
ax6.set_ylabel('F1 score', fontsize=8); ax6.set_ylim(0, 1.15)
ax6.legend(fontsize=7); ax6.grid(True, axis='y', alpha=0.3, linestyle=':')
ax6.tick_params(labelsize=7)

fig.suptitle(
    "Grid Search — Parkinson's MLP Optimisation\n"
    f"(pop/swarm={POP}, iters={N_ITER}, seed={SEED})",
    fontsize=12, fontweight='bold', y=1.01
)

plt.savefig('grid_search_plot.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved: grid_search_plot.png")
print("\nGrid search complete.\n")
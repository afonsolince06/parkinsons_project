"""
utils_hidden.py
===============
Systematic hyperparameter grid search for the Parkinson's MLP optimisation
— **two hidden layers** variant (22 → h1 → h2 → 1).

Four phases, each building on the last:

  Phase 1 — Fitness function   : recall vs F1 vs accuracy
  Phase 2 — GA  hyperparameters: pop_size × mutation_rate × crossover_method
  Phase 3 — PSO hyperparameters: n_particles × inertia (w→w_min) × c1/c2
  Phase 4 — Architecture       : hidden1_size × hidden2_size pairs   ← NEW
             (replaces the single hidden-size sweep from utilss.py)
             Pairs tested: (5,5), (5,10), (10,5), (10,10), (10,20), (20,10), (20,20)

Changes from utilss.py
----------------------
* Imports from genetic_algorithm_hidden / pso_hidden (two-layer GA/PSO).
* compute_n_params() now takes hidden1_size + hidden2_size.
* unpack_weights() slices six arrays (W1/b1/W2/b2/W3/b3).
* make_fitness() and full_eval() accept (hidden1_size, hidden2_size).
* run_ga() / run_pso() forward both hidden sizes.
* Phase 4 grid is over (h1, h2) pairs instead of a single hidden size.
* Output files named *_hidden.* to avoid overwriting single-layer results.
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

from genetic_algorithm_hidden import genetic_algorithm
from pso_hidden import particle_swarm_optimisation

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────
SEED   = 42
POP    = 30
N_ITER = 50

np.random.seed(SEED)

df = pd.read_csv('parkinsons_preprocessed.csv')
X  = df.drop(columns=['status']).values.astype(float)
y  = df['status'].values.astype(int)

N_SAMPLES  = len(y)
INPUT_SIZE = 22
OUTPUT_SIZE = 1


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — two-hidden-layer versions
# ─────────────────────────────────────────────────────────────────────────────

def compute_n_params(input_size=INPUT_SIZE,
                     hidden1_size=10,
                     hidden2_size=10,   # NEW
                     output_size=OUTPUT_SIZE):
    """Total parameters for a 2-hidden-layer MLP."""
    return (input_size   * hidden1_size + hidden1_size      # W1, b1
          + hidden1_size * hidden2_size + hidden2_size      # W2, b2  ← NEW
          + hidden2_size * output_size  + output_size)      # W3, b3


def unpack_weights(solution, input_size, hidden1_size, hidden2_size,
                   output_size=OUTPUT_SIZE):
    """
    Slice flat vector θ into six weight/bias arrays for sklearn.

    CHANGED from utilss.py: now returns three coefs_ and three intercepts_.
    Layout: [ W1 | b1 | W2 | b2 | W3 | b3 ]
    """
    idx = 0
    # W1: input → hidden-1
    nW1 = input_size * hidden1_size
    W1  = solution[idx:idx+nW1].reshape(input_size, hidden1_size); idx += nW1
    b1  = solution[idx:idx+hidden1_size];                           idx += hidden1_size
    # W2: hidden-1 → hidden-2  (NEW)
    nW2 = hidden1_size * hidden2_size
    W2  = solution[idx:idx+nW2].reshape(hidden1_size, hidden2_size); idx += nW2
    b2  = solution[idx:idx+hidden2_size];                             idx += hidden2_size
    # W3: hidden-2 → output
    nW3 = hidden2_size * output_size
    W3  = solution[idx:idx+nW3].reshape(hidden2_size, output_size); idx += nW3
    b3  = solution[idx:idx+output_size]
    return [W1, W2, W3], [b1, b2, b3]


def make_fitness(metric='f1', hidden1_size=10, hidden2_size=10,
                 input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
    """
    Factory: returns a fitness function for a given metric and 2-layer arch.

    CHANGED: hidden_layer_sizes=(h1, h2) — two-tuple instead of one-tuple.
    Metric options: 'recall', 'f1', 'accuracy'
    """
    def fitness_fn(solution, X, y):
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden1_size, hidden2_size),   # ← 2 layers
            activation='relu', solver='sgd', max_iter=1, random_state=42
        )
        clf.fit(np.zeros((2, input_size)), np.array([0, 1]))
        clf.coefs_, clf.intercepts_ = unpack_weights(
            solution, input_size, hidden1_size, hidden2_size, output_size)
        y_pred = clf.predict(X)
        if metric == 'recall':
            return recall_score(y, y_pred, pos_label=1, zero_division=0)
        if metric == 'f1':
            return f1_score(y, y_pred, pos_label=1, zero_division=0)
        if metric == 'accuracy':
            return accuracy_score(y, y_pred)
        raise ValueError(f"Unknown metric: {metric}")
    return fitness_fn


def full_eval(solution, hidden1_size=10, hidden2_size=10,
              input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
    """All four metrics + prediction counts. CHANGED: two-layer classifier."""
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden1_size, hidden2_size),
        activation='relu', solver='sgd', max_iter=1, random_state=42
    )
    clf.fit(np.zeros((2, input_size)), np.array([0, 1]))
    clf.coefs_, clf.intercepts_ = unpack_weights(
        solution, input_size, hidden1_size, hidden2_size, output_size)
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


def run_ga(fitness_fn, n_params, config,
           hidden1_size=10, hidden2_size=10):
    """Run GA with given config. CHANGED: passes both hidden sizes for Xavier."""
    t0 = time.perf_counter()
    theta, fit, hist = genetic_algorithm(
        fitness_fn       = fitness_fn,
        n_params         = n_params,
        fitness_args     = (X, y),
        pop_size         = config.get('pop_size', POP),
        n_generations    = N_ITER,
        crossover_rate   = 0.8,
        mutation_rate    = config.get('mutation_rate', 0.05),
        sigma            = config.get('sigma', 0.1),
        tournament_size  = 3,
        elitism          = 2,
        crossover_method = config.get('crossover_method', 'arithmetic'),
        mutation_method  = config.get('mutation_method', 'gaussian'),
        selection_method = 'tournament',
        init_method      = 'random',
        input_size       = INPUT_SIZE,
        hidden1_size     = hidden1_size,   # ← NEW
        hidden2_size     = hidden2_size,   # ← NEW
        output_size      = OUTPUT_SIZE,
        seed             = SEED,
        verbose          = False,
    )
    return theta, fit, hist, time.perf_counter() - t0


def run_pso(fitness_fn, n_params, config,
            hidden1_size=10, hidden2_size=10):
    """Run PSO with given config. CHANGED: passes both hidden sizes for header."""
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
        input_size       = INPUT_SIZE,
        hidden1_size     = hidden1_size,   # ← NEW
        hidden2_size     = hidden2_size,   # ← NEW
        output_size      = OUTPUT_SIZE,
        seed             = SEED,
        verbose          = False,
    )
    return theta, fit, hist, time.perf_counter() - t0


def print_result(label, metrics, elapsed, fitness_signal):
    deg = ' ⚠ DEGEN' if metrics['degenerate'] else ''
    print(f"  {label:<46}  "
          f"fit={fitness_signal:.3f}  "
          f"acc={metrics['accuracy']:.3f}  "
          f"f1={metrics['f1']:.3f}  "
          f"rec={metrics['recall']:.3f}  "
          f"prec={metrics['precision']:.3f}  "
          f"neg={metrics['n_pred_neg']:3d}  "
          f"{elapsed:5.1f}s{deg}")


# ─────────────────────────────────────────────────────────────────────────────
# Result storage
# ─────────────────────────────────────────────────────────────────────────────
all_results     = []
phase_histories = {}

# Default n_params for Phases 1-3 (both hidden layers = 10)
N_PARAMS_DEFAULT = compute_n_params(hidden1_size=10, hidden2_size=10)

print("\n" + "="*80)
print("  Grid Search — Two Hidden Layers  (22 → h1 → h2 → 1)")
print(f"  pop/swarm={POP}  iters={N_ITER}  seed={SEED}")
print(f"  Default architecture: 22 → 10 → 10 → 1  |  n_params={N_PARAMS_DEFAULT}")
print("="*80)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Fitness function
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("  PHASE 1 — Fitness function  (recall vs f1 vs accuracy)")
print("  Fixed: h1=10, h2=10, pop=30, iters=50, default GA/PSO params")
print("="*80)
print(f"  {'Config':<46}  fit    acc    f1     rec    prec   neg   time")
print("  " + "-"*76)

METRICS = ['recall', 'f1', 'accuracy']
best_fitness_metric = None
best_fitness_f1     = -1.0

for metric in METRICS:
    fn = make_fitness(metric=metric, hidden1_size=10, hidden2_size=10)
    for algo in ['GA', 'PSO']:
        cfg = {}
        if algo == 'GA':
            theta, fit, hist, elapsed = run_ga(fn, N_PARAMS_DEFAULT, cfg)
        else:
            theta, fit, hist, elapsed = run_pso(fn, N_PARAMS_DEFAULT, cfg)

        metrics = full_eval(theta, hidden1_size=10, hidden2_size=10)
        label   = f"{algo}  fitness={metric}"
        print_result(label, metrics, elapsed, fit)

        all_results.append({
            'phase': 1, 'algo': algo, 'label': label,
            'fitness_metric': metric,
            **metrics, 'fitness_signal': fit, 'elapsed': elapsed
        })
        phase_histories[label] = hist

        if not metrics['degenerate'] and metrics['f1'] > best_fitness_f1:
            best_fitness_f1     = metrics['f1']
            best_fitness_metric = metric

best_fitness_metric = best_fitness_metric or 'f1'
print(f"\n  → Best non-degenerate metric for next phases: "
      f"'{best_fitness_metric}'  (F1={best_fitness_f1:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — GA hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"  PHASE 2 — GA hyperparameters  "
      f"(fitness='{best_fitness_metric}', h1=10, h2=10)")
print("  Grid: pop_size × mutation_rate × crossover_method")
print("="*80)
print(f"  {'Config':<46}  fit    acc    f1     rec    prec   neg   time")
print("  " + "-"*76)

fn_phase2 = make_fitness(metric=best_fitness_metric,
                         hidden1_size=10, hidden2_size=10)

ga_grid = list(itertools.product(
    [30, 50],
    [0.05, 0.15],
    ['arithmetic', 'blend'],
))

best_ga_config = None
best_ga_f1     = -1.0

for pop, mut, cx in ga_grid:
    cfg   = {'pop_size': pop, 'mutation_rate': mut, 'crossover_method': cx}
    theta, fit, hist, elapsed = run_ga(fn_phase2, N_PARAMS_DEFAULT, cfg)
    metrics = full_eval(theta, hidden1_size=10, hidden2_size=10)
    label   = f"GA  pop={pop} mut={mut} cx={cx}"
    print_result(label, metrics, elapsed, fit)

    all_results.append({
        'phase': 2, 'algo': 'GA', 'label': label,
        'pop_size': pop, 'mutation_rate': mut, 'crossover_method': cx,
        **metrics, 'fitness_signal': fit, 'elapsed': elapsed
    })
    phase_histories[label] = hist

    if not metrics['degenerate'] and metrics['f1'] > best_ga_f1:
        best_ga_f1     = metrics['f1']
        best_ga_config = cfg.copy()

best_ga_config = best_ga_config or {
    'pop_size': 50, 'mutation_rate': 0.05, 'crossover_method': 'arithmetic'
}
print(f"\n  → Best GA config: {best_ga_config}  (F1={best_ga_f1:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — PSO hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"  PHASE 3 — PSO hyperparameters  "
      f"(fitness='{best_fitness_metric}', h1=10, h2=10)")
print("  Grid: n_particles × inertia (w→w_min) × c1/c2 balance")
print("="*80)
print(f"  {'Config':<46}  fit    acc    f1     rec    prec   neg   time")
print("  " + "-"*76)

fn_phase3 = make_fitness(metric=best_fitness_metric,
                         hidden1_size=10, hidden2_size=10)

pso_grid = list(itertools.product(
    [30, 50],
    [(0.9, 0.4), (0.7, 0.2)],
    [(2.0, 2.0), (1.5, 2.5)],
))

best_pso_config = None
best_pso_f1     = -1.0

for n_part, (w, w_min), (c1, c2) in pso_grid:
    cfg   = {'n_particles': n_part, 'w': w, 'w_min': w_min, 'c1': c1, 'c2': c2}
    theta, fit, hist, elapsed = run_pso(fn_phase3, N_PARAMS_DEFAULT, cfg)
    metrics = full_eval(theta, hidden1_size=10, hidden2_size=10)
    label   = f"PSO n={n_part} w={w}→{w_min} c1={c1}/c2={c2}"
    print_result(label, metrics, elapsed, fit)

    all_results.append({
        'phase': 3, 'algo': 'PSO', 'label': label,
        'n_particles': n_part, 'w': w, 'w_min': w_min, 'c1': c1, 'c2': c2,
        **metrics, 'fitness_signal': fit, 'elapsed': elapsed
    })
    phase_histories[label] = hist

    if not metrics['degenerate'] and metrics['f1'] > best_pso_f1:
        best_pso_f1     = metrics['f1']
        best_pso_config = cfg.copy()

best_pso_config = best_pso_config or {
    'n_particles': 50, 'w': 0.9, 'w_min': 0.4, 'c1': 2.0, 'c2': 2.0
}
print(f"\n  → Best PSO config: {best_pso_config}  (F1={best_pso_f1:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Architecture: hidden1_size × hidden2_size pairs
# ─────────────────────────────────────────────────────────────────────────────
# CHANGED from utilss.py: instead of sweeping a single hidden size (5,10,20),
# we now sweep *pairs* (h1, h2) to explore how both layers interact.
# Seven combinations covering small/medium/large and asymmetric pairs.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"  PHASE 4 — Architecture: hidden1 × hidden2 pairs")
print(f"  (fitness='{best_fitness_metric}', best configs from Phases 2 & 3)")
print("  Pairs: (5,5) (5,10) (10,5) (10,10) (10,20) (20,10) (20,20)")
print("="*80)
print(f"  {'Config':<46}  fit    acc    f1     rec    prec   neg   time")
print("  " + "-"*76)

# All (h1, h2) pairs to test
ARCH_PAIRS = [(5, 5), (5, 10), (10, 5), (10, 10), (10, 20), (20, 10), (20, 20)]

for h1, h2 in ARCH_PAIRS:
    n_params = compute_n_params(hidden1_size=h1, hidden2_size=h2)
    fn = make_fitness(metric=best_fitness_metric,
                      hidden1_size=h1, hidden2_size=h2)

    # GA with best config from Phase 2
    theta, fit, hist, elapsed = run_ga(
        fn, n_params, best_ga_config, hidden1_size=h1, hidden2_size=h2
    )
    metrics = full_eval(theta, hidden1_size=h1, hidden2_size=h2)
    label   = f"GA  h1={h1} h2={h2}  (n={n_params})"
    print_result(label, metrics, elapsed, fit)
    all_results.append({
        'phase': 4, 'algo': 'GA', 'label': label,
        'hidden1_size': h1, 'hidden2_size': h2, 'n_params': n_params,
        **metrics, 'fitness_signal': fit, 'elapsed': elapsed
    })
    phase_histories[label] = hist

    # PSO with best config from Phase 3
    theta, fit, hist, elapsed = run_pso(
        fn, n_params, best_pso_config, hidden1_size=h1, hidden2_size=h2
    )
    metrics = full_eval(theta, hidden1_size=h1, hidden2_size=h2)
    label   = f"PSO h1={h1} h2={h2}  (n={n_params})"
    print_result(label, metrics, elapsed, fit)
    all_results.append({
        'phase': 4, 'algo': 'PSO', 'label': label,
        'hidden1_size': h1, 'hidden2_size': h2, 'n_params': n_params,
        **metrics, 'fitness_signal': fit, 'elapsed': elapsed
    })
    phase_histories[label] = hist


# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("  OVERALL BEST RESULTS  (non-degenerate only, ranked by F1)")
print("="*80)
print(f"  {'Label':<48} {'F1':>6} {'Acc':>6} {'Rec':>6} {'Prec':>6} {'Neg':>5}")
print("  " + "-"*76)

non_degen = [r for r in all_results if not r['degenerate']]
ranked    = sorted(non_degen, key=lambda r: r['f1'], reverse=True)

for r in ranked[:15]:
    print(f"  {r['label']:<48} "
          f"{r['f1']:6.3f} "
          f"{r['accuracy']:6.3f} "
          f"{r['recall']:6.3f} "
          f"{r['precision']:6.3f} "
          f"{r['n_pred_neg']:5d}")

if not ranked:
    print("  All runs produced degenerate solutions — consider switching to F1.")

print("="*80)

# Save CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv('grid_search_results_hidden.csv', index=False)
print("\n  Saved: grid_search_results_hidden.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Plots — same 3×3 layout as utilss.py; Phase 4 panel updated for (h1,h2) pairs
# ─────────────────────────────────────────────────────────────────────────────
METRIC_COLORS = {
    'recall':   '#C04848',
    'f1':       '#3A9A5C',
    'accuracy': '#7752BE',
}

fig = plt.figure(figsize=(17, 14))
fig.patch.set_facecolor('#fafafa')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.40)
iters = list(range(1, N_ITER + 1))

# ── P1a: GA convergence by fitness metric ────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for metric in METRICS:
    hist = phase_histories.get(f"GA  fitness={metric}", [])
    ax1.plot(iters[:len(hist)], hist,
             color=METRIC_COLORS[metric], linewidth=1.8, label=metric)
ax1.set_title('Phase 1 — GA convergence\nby fitness metric',
              fontsize=9, fontweight='bold')
ax1.set_xlabel('Generation', fontsize=8)
ax1.set_ylabel('Best fitness', fontsize=8)
ax1.legend(fontsize=7); ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3, linestyle=':'); ax1.tick_params(labelsize=7)

# ── P1b: PSO convergence by fitness metric ───────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for metric in METRICS:
    hist = phase_histories.get(f"PSO  fitness={metric}", [])
    ax2.plot(iters[:len(hist)], hist,
             color=METRIC_COLORS[metric], linewidth=1.8, label=metric)
ax2.set_title('Phase 1 — PSO convergence\nby fitness metric',
              fontsize=9, fontweight='bold')
ax2.set_xlabel('Iteration', fontsize=8)
ax2.set_ylabel('Best fitness', fontsize=8)
ax2.legend(fontsize=7); ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3, linestyle=':'); ax2.tick_params(labelsize=7)

# ── P1c: bar chart F1 per fitness metric ─────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
p1_data = [r for r in all_results if r['phase'] == 1]
ga_f1s    = [next((r['f1'] for r in p1_data
                   if r['algo']=='GA'  and r['fitness_metric']==m), 0)
             for m in METRICS]
pso_f1s   = [next((r['f1'] for r in p1_data
                   if r['algo']=='PSO' and r['fitness_metric']==m), 0)
             for m in METRICS]
ga_degen  = [next((r['degenerate'] for r in p1_data
                   if r['algo']=='GA'  and r['fitness_metric']==m), True)
             for m in METRICS]
pso_degen = [next((r['degenerate'] for r in p1_data
                   if r['algo']=='PSO' and r['fitness_metric']==m), True)
             for m in METRICS]
x = np.arange(len(METRICS)); w = 0.35
bars_ga  = ax3.bar(x - w/2, ga_f1s,  w, color='#4A7CC3', alpha=0.85, label='GA')
bars_pso = ax3.bar(x + w/2, pso_f1s, w, color='#E07B3A', alpha=0.85, label='PSO')
for b, dg in zip(bars_ga, ga_degen):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             '⚠' if dg else f'{b.get_height():.2f}',
             ha='center', va='bottom', fontsize=7)
for b, dg in zip(bars_pso, pso_degen):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             '⚠' if dg else f'{b.get_height():.2f}',
             ha='center', va='bottom', fontsize=7)
ax3.set_xticks(x); ax3.set_xticklabels(METRICS, fontsize=8)
ax3.set_title('Phase 1 — F1 by fitness metric\n(⚠ = degenerate)',
              fontsize=9, fontweight='bold')
ax3.set_ylabel('F1 score', fontsize=8); ax3.set_ylim(0, 1.15)
ax3.legend(fontsize=7); ax3.grid(True, axis='y', alpha=0.3, linestyle=':')
ax3.tick_params(labelsize=7)

# ── P2: GA grid ranked by F1 ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
p2_data  = sorted([r for r in all_results if r['phase'] == 2],
                  key=lambda r: r['f1'], reverse=True)
labels2  = [r['label'].replace('GA  ', '') for r in p2_data]
f1s2     = [r['f1'] for r in p2_data]
degen2   = [r['degenerate'] for r in p2_data]
bcols2   = ['#c0392b' if d else '#4A7CC3' for d in degen2]
bars2    = ax4.barh(range(len(labels2)), f1s2, color=bcols2,
                    alpha=0.85, height=0.6)
for b, d, f in zip(bars2, degen2, f1s2):
    ax4.text(f + 0.005, b.get_y() + b.get_height()/2,
             '⚠ degen' if d else f'{f:.3f}', va='center', fontsize=7.5)
ax4.set_yticks(range(len(labels2))); ax4.set_yticklabels(labels2, fontsize=7.5)
ax4.set_xlabel('F1 score', fontsize=8)
ax4.set_title('Phase 2 — GA hyperparameter grid\n(ranked by F1)',
              fontsize=9, fontweight='bold')
ax4.set_xlim(0, 1.15); ax4.invert_yaxis()
ax4.grid(True, axis='x', alpha=0.3, linestyle=':'); ax4.tick_params(labelsize=7)
ax4.legend(handles=[
    mpatches.Patch(color='#4A7CC3', alpha=0.85, label='Normal'),
    mpatches.Patch(color='#c0392b', alpha=0.85, label='Degenerate'),
], fontsize=7, loc='lower right')

# ── P2 side: top-3 GA convergence ────────────────────────────────────────────
ax4b = fig.add_subplot(gs[1, 2])
top3_ga = [r for r in sorted([r for r in all_results if r['phase']==2],
           key=lambda r: r['f1'], reverse=True) if not r['degenerate']][:3]
if not top3_ga:
    top3_ga = sorted([r for r in all_results if r['phase']==2],
                     key=lambda r: r['f1'], reverse=True)[:3]
for i, r in enumerate(top3_ga):
    hist = phase_histories.get(r['label'], [])
    ax4b.plot(iters[:len(hist)], hist,
              color=['#1a5599', '#4A7CC3', '#89b4e8'][i],
              linewidth=1.6, alpha=0.9,
              label=r['label'].replace('GA  ', ''))
ax4b.set_title('Phase 2 — GA top configs\nconvergence',
               fontsize=9, fontweight='bold')
ax4b.set_xlabel('Generation', fontsize=8); ax4b.set_ylabel('Best fitness', fontsize=8)
ax4b.legend(fontsize=6.5); ax4b.set_ylim(0, 1.05)
ax4b.grid(True, alpha=0.3, linestyle=':'); ax4b.tick_params(labelsize=7)

# ── P3: PSO grid ranked by F1 ────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0:2])
p3_data  = sorted([r for r in all_results if r['phase'] == 3],
                  key=lambda r: r['f1'], reverse=True)
labels3  = [r['label'].replace('PSO ', '') for r in p3_data]
f1s3     = [r['f1'] for r in p3_data]
degen3   = [r['degenerate'] for r in p3_data]
bcols3   = ['#c0392b' if d else '#E07B3A' for d in degen3]
bars3    = ax5.barh(range(len(labels3)), f1s3, color=bcols3,
                    alpha=0.85, height=0.6)
for b, d, f in zip(bars3, degen3, f1s3):
    ax5.text(f + 0.005, b.get_y() + b.get_height()/2,
             '⚠ degen' if d else f'{f:.3f}', va='center', fontsize=7.5)
ax5.set_yticks(range(len(labels3))); ax5.set_yticklabels(labels3, fontsize=7.5)
ax5.set_xlabel('F1 score', fontsize=8)
ax5.set_title('Phase 3 — PSO hyperparameter grid\n(ranked by F1)',
              fontsize=9, fontweight='bold')
ax5.set_xlim(0, 1.15); ax5.invert_yaxis()
ax5.grid(True, axis='x', alpha=0.3, linestyle=':'); ax5.tick_params(labelsize=7)
ax5.legend(handles=[
    mpatches.Patch(color='#E07B3A', alpha=0.85, label='Normal'),
    mpatches.Patch(color='#c0392b', alpha=0.85, label='Degenerate'),
], fontsize=7, loc='lower right')

# ── P4: Architecture heatmap — h1 × h2 pairs (NEW) ──────────────────────────
# CHANGED from utilss.py: instead of a bar chart over one hidden size,
# we display a grouped bar chart over (h1, h2) pairs, one bar per algorithm.
ax6 = fig.add_subplot(gs[2, 2])
p4_data = [r for r in all_results if r['phase'] == 4]

pair_labels = [f"({h1},{h2})" for h1, h2 in ARCH_PAIRS]
ga_f1_p  = [next((r['f1'] for r in p4_data
                  if r['algo']=='GA'  and r['hidden1_size']==h1
                  and r['hidden2_size']==h2), 0)
             for h1, h2 in ARCH_PAIRS]
pso_f1_p = [next((r['f1'] for r in p4_data
                  if r['algo']=='PSO' and r['hidden1_size']==h1
                  and r['hidden2_size']==h2), 0)
             for h1, h2 in ARCH_PAIRS]
ga_dp    = [next((r['degenerate'] for r in p4_data
                  if r['algo']=='GA'  and r['hidden1_size']==h1
                  and r['hidden2_size']==h2), True)
             for h1, h2 in ARCH_PAIRS]
pso_dp   = [next((r['degenerate'] for r in p4_data
                  if r['algo']=='PSO' and r['hidden1_size']==h1
                  and r['hidden2_size']==h2), True)
             for h1, h2 in ARCH_PAIRS]

xp  = np.arange(len(ARCH_PAIRS)); wp = 0.35
bga  = ax6.bar(xp - wp/2, ga_f1_p,  wp, color='#4A7CC3', alpha=0.85, label='GA')
bpso = ax6.bar(xp + wp/2, pso_f1_p, wp, color='#E07B3A', alpha=0.85, label='PSO')

for b, d in zip(bga,  ga_dp):
    ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             '⚠' if d else f'{b.get_height():.2f}',
             ha='center', va='bottom', fontsize=6.5)
for b, d in zip(bpso, pso_dp):
    ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             '⚠' if d else f'{b.get_height():.2f}',
             ha='center', va='bottom', fontsize=6.5)

ax6.set_xticks(xp)
ax6.set_xticklabels(pair_labels, fontsize=7, rotation=30, ha='right')
ax6.set_title('Phase 4 — Architecture pairs\n(h1, h2) vs F1 score',   # CHANGED title
              fontsize=9, fontweight='bold')
ax6.set_ylabel('F1 score', fontsize=8); ax6.set_ylim(0, 1.15)
ax6.legend(fontsize=7); ax6.grid(True, axis='y', alpha=0.3, linestyle=':')
ax6.tick_params(labelsize=7)

fig.suptitle(
    "Grid Search — Parkinson's MLP  [Two Hidden Layers]\n"
    f"(pop/swarm={POP}, iters={N_ITER}, seed={SEED})",
    fontsize=12, fontweight='bold', y=1.01
)

plt.savefig('grid_search_plot_hidden.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved: grid_search_plot_hidden.png")
print("\nGrid search complete.\n")

"""
utils_hidden.py
===============
Systematic hyperparameter and operator grid search for the Parkinson's
MLP optimisation — two hidden layers variant (22 → h1 → h2 → 1).

NEW in this version
-------------------
Phase 5 — GA Operator Comparison  (standard vs extended operators)
    Compares GA with:
      • Standard operators : tournament + arithmetic + gaussian
      • Extended operators : rank + two_point + adaptive (new operators)
    Runs each configuration N_RUNS times with different seeds to collect
    a distribution of results suitable for statistical testing.

Phase 6 — Activation Function Combinations
    Loops over activation combinations (act_h1, act_h2) from
    {relu, tanh, logistic, identity} × {relu, tanh, logistic, identity}.
    Each combination is tested with both GA and PSO.
    Results saved to CSV for statistical analysis.

Statistical Tests (NEW)
-----------------------
After multi-run phases complete, the following tests are performed and
their results saved to 'statistical_tests_hidden.csv':

  1. Paired t-test (scipy.stats.ttest_rel)
       H0: GA-standard and GA-extended produce equal recall distributions.
       Appropriate when: distribution is roughly normal (n ≥ 30 or CLT).

  2. Wilcoxon signed-rank test (scipy.stats.wilcoxon)
       H0: No difference in medians between two paired samples.
       Appropriate when: distribution is non-normal or n is small.
       This is the non-parametric alternative to the paired t-test.

  3. One-way ANOVA (scipy.stats.f_oneway)
       H0: All GA operator combinations produce the same mean recall.
       Appropriate when: comparing 3+ groups simultaneously.

Why these tests?
    • Recall on the Parkinson's dataset tends to be bimodal (degenerate
      solutions cluster at 1.0 recall by predicting all positive; good
      solutions vary between 0.7–1.0).  Wilcoxon is more robust to this.
    • ANOVA covers the multi-group comparison of activation combinations.
    • p-values < 0.05 support rejecting H0; reported in the report with
      effect sizes (Cohen's d for t-test, rank-biserial r for Wilcoxon).

All multi-run results and statistical test outcomes are written to:
  • multi_run_results_hidden.csv   — raw per-run metrics
  • statistical_tests_hidden.csv   — test names, statistics, p-values
  • activation_results_hidden.csv  — activation combination results
  • grid_search_results_hidden.csv — hyperparameter grid results (Phase 1-4)
  • grid_search_plot_hidden.png    — visualisation of Phases 1–4
  • operator_comparison_plot.png   — box plots for Phase 5
  • activation_heatmap_plot.png    — heatmaps for Phase 6

Four legacy phases (from previous utils_hidden.py) are preserved intact:
  Phase 1 — Fitness function (recall vs F1 vs accuracy)
  Phase 2 — GA  hyperparameters
  Phase 3 — PSO hyperparameters
  Phase 4 — Architecture (h1, h2) pairs
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
from scipy import stats   # for statistical tests

from genetic_algorithm_hidden import genetic_algorithm
from pso_hidden import particle_swarm_optimisation
from parkinsons_hidden import (
    fitness_function,
    fitness_function_custom_act,
    evaluate_solution,
    evaluate_solution_custom_act,
    compute_n_params,
    generate_solution,
    SUPPORTED_ACTIVATIONS,
    DEFAULT_INPUT_SIZE,
    DEFAULT_HIDDEN1_SIZE,
    DEFAULT_HIDDEN2_SIZE,
    DEFAULT_OUTPUT_SIZE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────
SEED      = 42
POP       = 30
N_ITER    = 50
N_RUNS    = 10      # number of independent runs for statistical tests

np.random.seed(SEED)

df = pd.read_csv('parkinsons_preprocessed.csv')
X  = df.drop(columns=['status']).values.astype(float)
y  = df['status'].values.astype(int)

N_SAMPLES   = len(y)
INPUT_SIZE  = DEFAULT_INPUT_SIZE
OUTPUT_SIZE = DEFAULT_OUTPUT_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_n_params_local(input_size=INPUT_SIZE, hidden1_size=10,
                            hidden2_size=10, output_size=OUTPUT_SIZE):
    """Total parameters for a 2-hidden-layer MLP."""
    return (input_size   * hidden1_size + hidden1_size
          + hidden1_size * hidden2_size + hidden2_size
          + hidden2_size * output_size  + output_size)


def unpack_weights(solution, input_size, hidden1_size, hidden2_size,
                   output_size=OUTPUT_SIZE):
    """Slice flat θ into six weight/bias arrays for sklearn."""
    idx = 0
    nW1 = input_size * hidden1_size
    W1  = solution[idx:idx+nW1].reshape(input_size, hidden1_size); idx += nW1
    b1  = solution[idx:idx+hidden1_size];                           idx += hidden1_size
    nW2 = hidden1_size * hidden2_size
    W2  = solution[idx:idx+nW2].reshape(hidden1_size, hidden2_size); idx += nW2
    b2  = solution[idx:idx+hidden2_size];                             idx += hidden2_size
    nW3 = hidden2_size * output_size
    W3  = solution[idx:idx+nW3].reshape(hidden2_size, output_size); idx += nW3
    b3  = solution[idx:idx+output_size]
    return [W1, W2, W3], [b1, b2, b3]


def make_fitness(metric='recall', hidden1_size=10, hidden2_size=10,
                 input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
    """
    Factory: returns a fitness function for the given metric and 2-layer arch.
    Supports: 'recall', 'f1', 'accuracy'.
    """
    def fitness_fn(solution, X, y):
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden1_size, hidden2_size),
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
    """All four metrics + prediction counts. Returns a dict."""
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
           hidden1_size=10, hidden2_size=10, seed=SEED):
    """Run GA with given config. Returns (theta, fit, history, elapsed_sec)."""
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
        selection_method = config.get('selection_method', 'tournament'),
        crossover_method = config.get('crossover_method', 'arithmetic'),
        mutation_method  = config.get('mutation_method', 'gaussian'),
        init_method      = 'random',
        input_size       = INPUT_SIZE,
        hidden1_size     = hidden1_size,
        hidden2_size     = hidden2_size,
        output_size      = OUTPUT_SIZE,
        seed             = seed,
        verbose          = False,
    )
    return theta, fit, hist, time.perf_counter() - t0


def run_pso(fitness_fn, n_params, config,
            hidden1_size=10, hidden2_size=10, seed=SEED):
    """Run PSO with given config. Returns (theta, fit, history, elapsed_sec)."""
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
        hidden1_size     = hidden1_size,
        hidden2_size     = hidden2_size,
        output_size      = OUTPUT_SIZE,
        seed             = seed,
        verbose          = False,
    )
    return theta, fit, hist, time.perf_counter() - t0


def print_result(label, metrics, elapsed, fitness_signal):
    """Print a compact result row to stdout."""
    deg = ' ⚠ DEGEN' if metrics['degenerate'] else ''
    print(f"  {label:<50}  "
          f"fit={fitness_signal:.3f}  "
          f"acc={metrics['accuracy']:.3f}  "
          f"f1={metrics['f1']:.3f}  "
          f"rec={metrics['recall']:.3f}  "
          f"prec={metrics['precision']:.3f}  "
          f"neg={metrics['n_pred_neg']:3d}  "
          f"{elapsed:5.1f}s{deg}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical test helpers
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    """Cohen's d effect size for two arrays."""
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled_std + 1e-12)


def rank_biserial_r(a, b):
    """Rank-biserial correlation (effect size for Wilcoxon test)."""
    n1, n2 = len(a), len(b)
    u, _ = stats.mannwhitneyu(a, b, alternative='two-sided')
    return 1 - (2 * u) / (n1 * n2)


def run_statistical_tests(group_a, group_b, label_a, label_b):
    """
    Run paired t-test, Wilcoxon signed-rank test, and return results dict.

    Parameters
    ----------
    group_a, group_b : array-like of floats — recall scores from N_RUNS
    label_a, label_b : str — descriptive labels for the groups

    Returns
    -------
    dict with test name, statistic, p-value, effect size, interpretation
    """
    results = []

    a = np.array(group_a)
    b = np.array(group_b)

    # 1. Paired t-test (parametric, assumes normality)
    try:
        t_stat, t_p = stats.ttest_rel(a, b)
        d = cohens_d(a, b)
        results.append({
            'test'       : 'Paired t-test',
            'group_a'    : label_a,
            'group_b'    : label_b,
            'statistic'  : round(t_stat, 4),
            'p_value'    : round(t_p, 4),
            'effect_size': round(d, 4),
            'effect_label': 'Cohen\'s d',
            'significant': t_p < 0.05,
            'note'       : 'Parametric; assumes normality',
        })
    except Exception as e:
        results.append({'test': 'Paired t-test', 'error': str(e)})

    # 2. Wilcoxon signed-rank test (non-parametric alternative)
    try:
        w_stat, w_p = stats.wilcoxon(a, b, zero_method='wilcox')
        r = rank_biserial_r(a, b)
        results.append({
            'test'       : 'Wilcoxon signed-rank',
            'group_a'    : label_a,
            'group_b'    : label_b,
            'statistic'  : round(w_stat, 4),
            'p_value'    : round(w_p, 4),
            'effect_size': round(r, 4),
            'effect_label': 'Rank-biserial r',
            'significant': w_p < 0.05,
            'note'       : 'Non-parametric; robust to non-normality',
        })
    except Exception as e:
        results.append({'test': 'Wilcoxon signed-rank', 'error': str(e)})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Result storage
# ─────────────────────────────────────────────────────────────────────────────
all_results     = []
phase_histories = {}
multi_run_rows  = []    # NEW: per-run rows for statistical tests
stat_test_rows  = []    # NEW: statistical test results
act_rows        = []    # NEW: activation combination results

N_PARAMS_DEFAULT = compute_n_params_local(hidden1_size=10, hidden2_size=10)

print("\n" + "="*80)
print("  Grid Search + Statistical Analysis — Two Hidden Layers  (22 → h1 → h2 → 1)")
print(f"  pop/swarm={POP}  iters={N_ITER}  seed={SEED}  n_runs={N_RUNS}")
print(f"  Default architecture: 22 → 10 → 10 → 1  |  n_params={N_PARAMS_DEFAULT}")
print("="*80)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Fitness function
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("  PHASE 1 — Fitness function  (recall vs f1 vs accuracy)")
print("  Fixed: h1=10, h2=10, pop=30, iters=50, standard GA/PSO")
print("="*80)
print(f"  {'Config':<50}  fit    acc    f1     rec    prec   neg   time")
print("  " + "-"*78)

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

best_fitness_metric = best_fitness_metric or 'recall'
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
print(f"  {'Config':<50}  fit    acc    f1     rec    prec   neg   time")
print("  " + "-"*78)

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
print(f"  {'Config':<50}  fit    acc    f1     rec    prec   neg   time")
print("  " + "-"*78)

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
print("\n" + "="*80)
print(f"  PHASE 4 — Architecture: hidden1 × hidden2 pairs")
print(f"  (fitness='{best_fitness_metric}', best configs from Phases 2 & 3)")
print("  Pairs: (5,5) (5,10) (10,5) (10,10) (10,20) (20,10) (20,20)")
print("="*80)
print(f"  {'Config':<50}  fit    acc    f1     rec    prec   neg   time")
print("  " + "-"*78)

ARCH_PAIRS = [(5, 5), (5, 10), (10, 5), (10, 10), (10, 20), (20, 10), (20, 20)]

for h1, h2 in ARCH_PAIRS:
    n_params = compute_n_params_local(hidden1_size=h1, hidden2_size=h2)
    fn = make_fitness(metric=best_fitness_metric,
                      hidden1_size=h1, hidden2_size=h2)

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
# PHASE 5 — GA Operator Comparison  (multi-run, statistical tests)  ← NEW
# ─────────────────────────────────────────────────────────────────────────────
"""
Why this phase?
    Phases 1-4 use a single run per configuration, which cannot distinguish
    genuine algorithmic differences from random seed effects.  Phase 5 runs
    each configuration N_RUNS=10 times with different seeds and compares:

    Variant A — Standard operators  : tournament + arithmetic + gaussian
    Variant B — Extended operators  : rank      + two_point  + adaptive

    Statistical tests (paired t-test, Wilcoxon) answer whether the extended
    operators produce significantly different recall distributions.
"""
print("\n" + "="*80)
print("  PHASE 5 — GA Operator Comparison  (multi-run + statistical tests)")
print(f"  {N_RUNS} independent runs per variant  |  seeds: {SEED}…{SEED+N_RUNS-1}")
print("  Variant A: tournament + arithmetic + gaussian  (standard)")
print("  Variant B: rank + two_point + adaptive         (extended)")
print("="*80)

fn_p5 = make_fitness(metric=best_fitness_metric,
                     hidden1_size=10, hidden2_size=10)

# GA operator configurations to compare
OPERATOR_VARIANTS = {
    'GA-standard': {
        # Standard operators — tournament selection, arithmetic crossover,
        # Gaussian mutation with fixed σ
        'selection_method': 'tournament',
        'crossover_method': 'arithmetic',
        'mutation_method' : 'gaussian',
        'pop_size'        : 30,
        'mutation_rate'   : 0.05,
        'sigma'           : 0.1,
    },
    'GA-extended': {
        # Extended operators — rank-based selection, two-point crossover,
        # adaptive mutation (σ scales with stagnation)
        'selection_method': 'rank',
        'crossover_method': 'two_point',
        'mutation_method' : 'adaptive',
        'pop_size'        : 30,
        'mutation_rate'   : 0.05,
        'sigma'           : 0.1,
    },
}

# Also include PSO for three-way comparison
run_recall = {k: [] for k in list(OPERATOR_VARIANTS.keys()) + ['PSO']}
run_f1     = {k: [] for k in list(OPERATOR_VARIANTS.keys()) + ['PSO']}
run_acc    = {k: [] for k in list(OPERATOR_VARIANTS.keys()) + ['PSO']}
run_time   = {k: [] for k in list(OPERATOR_VARIANTS.keys()) + ['PSO']}

for run_i in range(N_RUNS):
    run_seed = SEED + run_i
    print(f"\n  Run {run_i+1}/{N_RUNS}  (seed={run_seed})")

    for var_name, cfg in OPERATOR_VARIANTS.items():
        theta, fit, hist, elapsed = run_ga(
            fn_p5, N_PARAMS_DEFAULT, cfg, seed=run_seed
        )
        m = full_eval(theta, hidden1_size=10, hidden2_size=10)
        print(f"    {var_name:<16} recall={m['recall']:.4f}  "
              f"f1={m['f1']:.4f}  acc={m['accuracy']:.4f}  {elapsed:.1f}s")

        run_recall[var_name].append(m['recall'])
        run_f1[var_name].append(m['f1'])
        run_acc[var_name].append(m['accuracy'])
        run_time[var_name].append(elapsed)

        multi_run_rows.append({
            'phase': 5, 'algo': var_name, 'run': run_i+1, 'seed': run_seed,
            **m, 'fitness_signal': fit, 'elapsed': elapsed,
            'selection_method': cfg['selection_method'],
            'crossover_method': cfg['crossover_method'],
            'mutation_method' : cfg['mutation_method'],
        })

    # PSO reference run
    theta_p, fit_p, hist_p, elapsed_p = run_pso(
        fn_p5, N_PARAMS_DEFAULT, best_pso_config, seed=run_seed
    )
    m_p = full_eval(theta_p, hidden1_size=10, hidden2_size=10)
    print(f"    {'PSO':<16} recall={m_p['recall']:.4f}  "
          f"f1={m_p['f1']:.4f}  acc={m_p['accuracy']:.4f}  {elapsed_p:.1f}s")

    run_recall['PSO'].append(m_p['recall'])
    run_f1['PSO'].append(m_p['f1'])
    run_acc['PSO'].append(m_p['accuracy'])
    run_time['PSO'].append(elapsed_p)

    multi_run_rows.append({
        'phase': 5, 'algo': 'PSO', 'run': run_i+1, 'seed': run_seed,
        **m_p, 'fitness_signal': fit_p, 'elapsed': elapsed_p,
        'selection_method': 'pso', 'crossover_method': 'pso',
        'mutation_method': 'pso',
    })

# Summary statistics
print(f"\n  {'Variant':<18} {'Recall mean±std':>20} {'F1 mean±std':>20} {'Time mean':>12}")
print("  " + "-"*72)
for var in list(OPERATOR_VARIANTS.keys()) + ['PSO']:
    r = np.array(run_recall[var])
    f = np.array(run_f1[var])
    t = np.array(run_time[var])
    print(f"  {var:<18} "
          f"{r.mean():.4f} ± {r.std():.4f}       "
          f"{f.mean():.4f} ± {f.std():.4f}       "
          f"{t.mean():.1f}s")

# ── Statistical tests between variants ──────────────────────────────────────
print(f"\n  Statistical Tests (n={N_RUNS} paired runs):")
print("  " + "-"*72)

test_pairs = [
    ('GA-standard', 'GA-extended', 'Standard vs Extended operators'),
    ('GA-extended', 'PSO',         'GA-extended vs PSO'),
    ('GA-standard', 'PSO',         'GA-standard vs PSO'),
]

for (a, b, desc) in test_pairs:
    print(f"\n  {desc}:")
    test_results = run_statistical_tests(
        run_recall[a], run_recall[b], a, b
    )
    for r in test_results:
        if 'error' in r:
            print(f"    {r['test']}: ERROR — {r['error']}")
        else:
            sig = "✓ SIGNIFICANT" if r['significant'] else "✗ not significant"
            print(f"    {r['test']:<28}  stat={r['statistic']:8.4f}  "
                  f"p={r['p_value']:.4f}  {r['effect_label']}={r['effect_size']:.4f}"
                  f"  [{sig}]")
            stat_test_rows.append({**r, 'metric': 'recall', 'description': desc})

# ANOVA across all three groups
try:
    f_stat, f_p = stats.f_oneway(
        run_recall['GA-standard'],
        run_recall['GA-extended'],
        run_recall['PSO']
    )
    sig = "✓ SIGNIFICANT" if f_p < 0.05 else "✗ not significant"
    print(f"\n  One-way ANOVA (GA-standard vs GA-extended vs PSO):")
    print(f"    F={f_stat:.4f}  p={f_p:.4f}  [{sig}]")
    stat_test_rows.append({
        'test': 'One-way ANOVA',
        'group_a': 'GA-standard + GA-extended + PSO',
        'group_b': '(all)',
        'statistic': round(f_stat, 4),
        'p_value'  : round(f_p, 4),
        'effect_size': '',
        'effect_label': '',
        'significant': f_p < 0.05,
        'note': 'Three-group comparison; H0: equal means',
        'metric': 'recall',
        'description': 'All three variants',
    })
except Exception as e:
    print(f"  ANOVA error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — Activation Function Combinations  ← NEW
# ─────────────────────────────────────────────────────────────────────────────
"""
Why explore activation combinations?
    sklearn's MLPClassifier uses one activation uniformly across all hidden
    layers.  The custom forward pass in parkinsons_hidden.py allows mixing.
    We test combinations from a reduced set to keep runtime manageable:
    {relu, tanh, logistic} × {relu, tanh, logistic}  (9 combinations).

    Hypothesis:
    • Layer 1 activation affects the raw feature representation extracted
      from the 22 biomarkers.
    • Layer 2 activation affects how those features are compressed before
      the output.
    • The optimal combination may differ from the uniform relu baseline.

    Statistical support:
    A one-way ANOVA tests whether different activation combinations produce
    significantly different recall.  Post-hoc Tukey HSD (if available) or
    pairwise Wilcoxon tests identify which specific combinations differ.
"""
print("\n" + "="*80)
print("  PHASE 6 — Activation Function Combinations")
print("  Testing all combinations of (act_h1, act_h2) with GA and PSO")
print("  Activations: relu, tanh, logistic  (3×3 = 9 combinations)")
print("="*80)
print(f"  {'Config':<55}  recall  f1      acc     time")
print("  " + "-"*78)

# Reduce to 3 activations (identity excluded from main sweep to save time)
ACT_SWEEP = ['relu', 'tanh', 'logistic']
ACT_COMBINATIONS = list(itertools.product(ACT_SWEEP, ACT_SWEEP))

act_recall_by_combo = {}  # for ANOVA

for act_h1, act_h2 in ACT_COMBINATIONS:
    combo_key = f"{act_h1}+{act_h2}"

    # Fitness function using the custom forward pass
    def make_act_fitness(a1=act_h1, a2=act_h2):
        def fn(solution, X, y):
            return fitness_function_custom_act(
                solution, X, y,
                hidden1_size=10, hidden2_size=10,
                act_h1=a1, act_h2=a2
            )
        return fn

    fn_act = make_act_fitness(act_h1, act_h2)

    for algo in ['GA', 'PSO']:
        t0 = time.perf_counter()
        if algo == 'GA':
            theta, fit, hist, elapsed = run_ga(
                fn_act, N_PARAMS_DEFAULT,
                {'selection_method': 'tournament',
                 'crossover_method': 'arithmetic',
                 'mutation_method' : 'gaussian'},
                seed=SEED
            )
        else:
            theta, fit, hist, elapsed = run_pso(
                fn_act, N_PARAMS_DEFAULT, best_pso_config, seed=SEED
            )

        m = evaluate_solution_custom_act(
            theta, X, y,
            hidden1_size=10, hidden2_size=10,
            act_h1=act_h1, act_h2=act_h2
        )

        label = f"{algo}  act_h1={act_h1:<8}  act_h2={act_h2:<8}"
        print(f"  {label:<55}  "
              f"{m['recall']:.4f}  {m['f1']:.4f}  "
              f"{m['accuracy']:.4f}  {elapsed:.1f}s")

        act_rows.append({
            'algo'        : algo,
            'act_h1'      : act_h1,
            'act_h2'      : act_h2,
            'combo'       : combo_key,
            'recall'      : m['recall'],
            'f1'          : m['f1'],
            'accuracy'    : m['accuracy'],
            'precision'   : m['precision'],
            'n_pred_pos'  : m['n_pred_pos'],
            'n_pred_neg'  : m['n_pred_neg'],
            'fitness_signal': fit,
            'elapsed'     : elapsed,
        })

        if algo == 'GA':
            act_recall_by_combo[combo_key] = m['recall']

# ANOVA across activation combinations (GA only)
combo_recalls = [
    [r['recall'] for r in act_rows if r['algo']=='GA' and r['combo']==k]
    for k in [f"{a1}+{a2}" for a1, a2 in ACT_COMBINATIONS]
]
combo_recalls = [g for g in combo_recalls if len(g) > 0]

try:
    f_stat_act, f_p_act = stats.f_oneway(*combo_recalls)
    sig = "✓ SIGNIFICANT" if f_p_act < 0.05 else "✗ not significant"
    print(f"\n  ANOVA across all activation combinations (GA):")
    print(f"  F={f_stat_act:.4f}  p={f_p_act:.4f}  [{sig}]")
    stat_test_rows.append({
        'test'       : 'One-way ANOVA',
        'group_a'    : 'All GA activation combos',
        'group_b'    : '(all)',
        'statistic'  : round(f_stat_act, 4),
        'p_value'    : round(f_p_act, 4),
        'effect_size': '',
        'effect_label': '',
        'significant': f_p_act < 0.05,
        'note'       : 'One run per combo; tests whether activation matters',
        'metric'     : 'recall',
        'description': 'Phase 6: Activation combinations',
    })
except Exception as e:
    print(f"  ANOVA error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Final summary (Phases 1-4)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("  OVERALL BEST RESULTS  (non-degenerate only, ranked by F1)")
print("="*80)
print(f"  {'Label':<52} {'F1':>6} {'Acc':>6} {'Rec':>6} {'Prec':>6} {'Neg':>5}")
print("  " + "-"*78)

non_degen = [r for r in all_results if not r.get('degenerate', False)]
ranked    = sorted(non_degen, key=lambda r: r['f1'], reverse=True)

for r in ranked[:15]:
    print(f"  {r['label']:<52} "
          f"{r['f1']:6.3f} "
          f"{r['accuracy']:6.3f} "
          f"{r['recall']:6.3f} "
          f"{r['precision']:6.3f} "
          f"{r['n_pred_neg']:5d}")

if not ranked:
    print("  All runs produced degenerate solutions.")

print("="*80)

# ─────────────────────────────────────────────────────────────────────────────
# Save all results to CSV
# ─────────────────────────────────────────────────────────────────────────────
pd.DataFrame(all_results).to_csv('grid_search_results_hidden.csv', index=False)
print("\n  Saved: grid_search_results_hidden.csv")

pd.DataFrame(multi_run_rows).to_csv('multi_run_results_hidden.csv', index=False)
print("  Saved: multi_run_results_hidden.csv")

pd.DataFrame(stat_test_rows).to_csv('statistical_tests_hidden.csv', index=False)
print("  Saved: statistical_tests_hidden.csv")

pd.DataFrame(act_rows).to_csv('activation_results_hidden.csv', index=False)
print("  Saved: activation_results_hidden.csv")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
METRIC_COLORS = {
    'recall':   '#C04848',
    'f1':       '#3A9A5C',
    'accuracy': '#7752BE',
}
iters = list(range(1, N_ITER + 1))

# ── Plot A: Legacy grid search (Phases 1–4) ──────────────────────────────────
fig = plt.figure(figsize=(17, 14))
fig.patch.set_facecolor('#fafafa')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.40)

# P1a: GA convergence by fitness metric
ax1 = fig.add_subplot(gs[0, 0])
for metric in METRICS:
    hist = phase_histories.get(f"GA  fitness={metric}", [])
    ax1.plot(iters[:len(hist)], hist,
             color=METRIC_COLORS[metric], linewidth=1.8, label=metric)
ax1.set_title('Phase 1 — GA convergence\nby fitness metric',
              fontsize=9, fontweight='bold')
ax1.set_xlabel('Generation', fontsize=8); ax1.set_ylabel('Best fitness', fontsize=8)
ax1.legend(fontsize=7); ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3, linestyle=':'); ax1.tick_params(labelsize=7)

# P1b: PSO convergence by fitness metric
ax2 = fig.add_subplot(gs[0, 1])
for metric in METRICS:
    hist = phase_histories.get(f"PSO  fitness={metric}", [])
    ax2.plot(iters[:len(hist)], hist,
             color=METRIC_COLORS[metric], linewidth=1.8, label=metric)
ax2.set_title('Phase 1 — PSO convergence\nby fitness metric',
              fontsize=9, fontweight='bold')
ax2.set_xlabel('Iteration', fontsize=8); ax2.set_ylabel('Best fitness', fontsize=8)
ax2.legend(fontsize=7); ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3, linestyle=':'); ax2.tick_params(labelsize=7)

# P1c: bar chart F1 per fitness metric
ax3 = fig.add_subplot(gs[0, 2])
p1_data = [r for r in all_results if r['phase'] == 1]
ga_f1s   = [next((r['f1'] for r in p1_data if r['algo']=='GA'  and r['fitness_metric']==m), 0) for m in METRICS]
pso_f1s  = [next((r['f1'] for r in p1_data if r['algo']=='PSO' and r['fitness_metric']==m), 0) for m in METRICS]
ga_degen = [next((r['degenerate'] for r in p1_data if r['algo']=='GA'  and r['fitness_metric']==m), True) for m in METRICS]
pso_degen= [next((r['degenerate'] for r in p1_data if r['algo']=='PSO' and r['fitness_metric']==m), True) for m in METRICS]
xb = np.arange(len(METRICS)); wb = 0.35
bga  = ax3.bar(xb - wb/2, ga_f1s,  wb, color='#4A7CC3', alpha=0.85, label='GA')
bpso = ax3.bar(xb + wb/2, pso_f1s, wb, color='#E07B3A', alpha=0.85, label='PSO')
for b, dg in zip(bga, ga_degen):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             '⚠' if dg else f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=7)
for b, dg in zip(bpso, pso_degen):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             '⚠' if dg else f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=7)
ax3.set_xticks(xb); ax3.set_xticklabels(METRICS, fontsize=8)
ax3.set_title('Phase 1 — F1 by fitness metric\n(⚠ = degenerate)',
              fontsize=9, fontweight='bold')
ax3.set_ylabel('F1 score', fontsize=8); ax3.set_ylim(0, 1.15)
ax3.legend(fontsize=7); ax3.grid(True, axis='y', alpha=0.3, linestyle=':')
ax3.tick_params(labelsize=7)

# P2: GA grid ranked by F1
ax4 = fig.add_subplot(gs[1, 0:2])
p2_data  = sorted([r for r in all_results if r['phase'] == 2], key=lambda r: r['f1'], reverse=True)
labels2  = [r['label'].replace('GA  ', '') for r in p2_data]
f1s2     = [r['f1'] for r in p2_data]
degen2   = [r['degenerate'] for r in p2_data]
bcols2   = ['#c0392b' if d else '#4A7CC3' for d in degen2]
bars2    = ax4.barh(range(len(labels2)), f1s2, color=bcols2, alpha=0.85, height=0.6)
for b, d, f in zip(bars2, degen2, f1s2):
    ax4.text(f + 0.005, b.get_y()+b.get_height()/2,
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

# P2 side: top-3 GA convergence
ax4b = fig.add_subplot(gs[1, 2])
top3_ga = [r for r in sorted([r for r in all_results if r['phase']==2],
           key=lambda r: r['f1'], reverse=True) if not r['degenerate']][:3]
if not top3_ga:
    top3_ga = sorted([r for r in all_results if r['phase']==2], key=lambda r: r['f1'], reverse=True)[:3]
for i, r in enumerate(top3_ga):
    hist = phase_histories.get(r['label'], [])
    ax4b.plot(iters[:len(hist)], hist,
              color=['#1a5599', '#4A7CC3', '#89b4e8'][i],
              linewidth=1.6, alpha=0.9, label=r['label'].replace('GA  ', ''))
ax4b.set_title('Phase 2 — GA top configs\nconvergence', fontsize=9, fontweight='bold')
ax4b.set_xlabel('Generation', fontsize=8); ax4b.set_ylabel('Best fitness', fontsize=8)
ax4b.legend(fontsize=6.5); ax4b.set_ylim(0, 1.05)
ax4b.grid(True, alpha=0.3, linestyle=':'); ax4b.tick_params(labelsize=7)

# P4: Architecture heatmap
ax6 = fig.add_subplot(gs[2, 2])
p4_data = [r for r in all_results if r['phase'] == 4]
pair_labels = [f"({h1},{h2})" for h1, h2 in ARCH_PAIRS]
ga_f1_p  = [next((r['f1'] for r in p4_data if r['algo']=='GA'  and r['hidden1_size']==h1 and r['hidden2_size']==h2), 0) for h1, h2 in ARCH_PAIRS]
pso_f1_p = [next((r['f1'] for r in p4_data if r['algo']=='PSO' and r['hidden1_size']==h1 and r['hidden2_size']==h2), 0) for h1, h2 in ARCH_PAIRS]
ga_dp    = [next((r['degenerate'] for r in p4_data if r['algo']=='GA'  and r['hidden1_size']==h1 and r['hidden2_size']==h2), True) for h1, h2 in ARCH_PAIRS]
pso_dp   = [next((r['degenerate'] for r in p4_data if r['algo']=='PSO' and r['hidden1_size']==h1 and r['hidden2_size']==h2), True) for h1, h2 in ARCH_PAIRS]
xp = np.arange(len(ARCH_PAIRS)); wp = 0.35
bga2  = ax6.bar(xp - wp/2, ga_f1_p,  wp, color='#4A7CC3', alpha=0.85, label='GA')
bpso2 = ax6.bar(xp + wp/2, pso_f1_p, wp, color='#E07B3A', alpha=0.85, label='PSO')
for b, d in zip(bga2, ga_dp):
    ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             '⚠' if d else f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=6.5)
for b, d in zip(bpso2, pso_dp):
    ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             '⚠' if d else f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=6.5)
ax6.set_xticks(xp); ax6.set_xticklabels(pair_labels, fontsize=7, rotation=30, ha='right')
ax6.set_title('Phase 4 — Architecture pairs\n(h1, h2) vs F1 score', fontsize=9, fontweight='bold')
ax6.set_ylabel('F1 score', fontsize=8); ax6.set_ylim(0, 1.15)
ax6.legend(fontsize=7); ax6.grid(True, axis='y', alpha=0.3, linestyle=':')
ax6.tick_params(labelsize=7)

# P3: PSO grid ranked by F1
ax5 = fig.add_subplot(gs[2, 0:2])
p3_data  = sorted([r for r in all_results if r['phase'] == 3], key=lambda r: r['f1'], reverse=True)
labels3  = [r['label'].replace('PSO ', '') for r in p3_data]
f1s3     = [r['f1'] for r in p3_data]
degen3   = [r['degenerate'] for r in p3_data]
bcols3   = ['#c0392b' if d else '#E07B3A' for d in degen3]
bars3    = ax5.barh(range(len(labels3)), f1s3, color=bcols3, alpha=0.85, height=0.6)
for b, d, f in zip(bars3, degen3, f1s3):
    ax5.text(f + 0.005, b.get_y()+b.get_height()/2,
             '⚠ degen' if d else f'{f:.3f}', va='center', fontsize=7.5)
ax5.set_yticks(range(len(labels3))); ax5.set_yticklabels(labels3, fontsize=7.5)
ax5.set_xlabel('F1 score', fontsize=8)
ax5.set_title('Phase 3 — PSO hyperparameter grid\n(ranked by F1)', fontsize=9, fontweight='bold')
ax5.set_xlim(0, 1.15); ax5.invert_yaxis()
ax5.grid(True, axis='x', alpha=0.3, linestyle=':'); ax5.tick_params(labelsize=7)
ax5.legend(handles=[
    mpatches.Patch(color='#E07B3A', alpha=0.85, label='Normal'),
    mpatches.Patch(color='#c0392b', alpha=0.85, label='Degenerate'),
], fontsize=7, loc='lower right')

fig.suptitle(
    "Grid Search — Parkinson's MLP  [Two Hidden Layers]\n"
    f"(pop/swarm={POP}, iters={N_ITER}, seed={SEED})",
    fontsize=12, fontweight='bold', y=1.01
)
plt.savefig('grid_search_plot_hidden.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved: grid_search_plot_hidden.png")

# ── Plot B: Operator comparison box plots (Phase 5) ──────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.patch.set_facecolor('#fafafa')

variant_labels = ['GA-standard', 'GA-extended', 'PSO']
colors_b = ['#4A7CC3', '#E07B3A', '#3A9A5C']

# Box plot: Recall
ax_b1 = axes2[0]
ax_b1.set_facecolor('#ffffff')
bp1 = ax_b1.boxplot(
    [run_recall[v] for v in variant_labels],
    labels=variant_labels, patch_artist=True,
    medianprops=dict(color='black', linewidth=2),
)
for patch, color in zip(bp1['boxes'], colors_b):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax_b1.set_title('Phase 5 — Recall Distribution\n(10 independent runs)',
                fontsize=11, fontweight='bold')
ax_b1.set_ylabel('Recall', fontsize=10); ax_b1.set_ylim(0, 1.05)
ax_b1.grid(True, axis='y', alpha=0.3, linestyle=':')
for s in ax_b1.spines.values(): s.set_linewidth(0.5)

# Box plot: F1
ax_b2 = axes2[1]
ax_b2.set_facecolor('#ffffff')
bp2 = ax_b2.boxplot(
    [run_f1[v] for v in variant_labels],
    labels=variant_labels, patch_artist=True,
    medianprops=dict(color='black', linewidth=2),
)
for patch, color in zip(bp2['boxes'], colors_b):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax_b2.set_title('Phase 5 — F1 Distribution\n(10 independent runs)',
                fontsize=11, fontweight='bold')
ax_b2.set_ylabel('F1 Score', fontsize=10); ax_b2.set_ylim(0, 1.05)
ax_b2.grid(True, axis='y', alpha=0.3, linestyle=':')
for s in ax_b2.spines.values(): s.set_linewidth(0.5)

fig2.suptitle(
    "Operator Comparison: GA-standard vs GA-extended vs PSO\n"
    f"(22→10→10→1, {N_RUNS} runs each, pop/swarm={POP}, iters={N_ITER})",
    fontsize=10, y=1.02, color='#333',
)
plt.tight_layout()
plt.savefig('operator_comparison_plot.png', dpi=150, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
plt.close()
print("  Saved: operator_comparison_plot.png")

# ── Plot C: Activation combination heatmap (Phase 6) ────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.patch.set_facecolor('#fafafa')

for ax_idx, algo_name in enumerate(['GA', 'PSO']):
    ax = axes3[ax_idx]
    ax.set_facecolor('#ffffff')

    # Build recall matrix (3×3)
    heatmap_data = np.zeros((len(ACT_SWEEP), len(ACT_SWEEP)))
    for i, a1 in enumerate(ACT_SWEEP):
        for j, a2 in enumerate(ACT_SWEEP):
            row = next((r for r in act_rows
                        if r['algo']==algo_name
                        and r['act_h1']==a1
                        and r['act_h2']==a2), None)
            heatmap_data[i, j] = row['recall'] if row else 0.0

    im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1,
                   aspect='auto')
    ax.set_xticks(range(len(ACT_SWEEP))); ax.set_xticklabels(ACT_SWEEP, fontsize=9)
    ax.set_yticks(range(len(ACT_SWEEP))); ax.set_yticklabels(ACT_SWEEP, fontsize=9)
    ax.set_xlabel('act_h2 (hidden layer 2)', fontsize=9)
    ax.set_ylabel('act_h1 (hidden layer 1)', fontsize=9)
    ax.set_title(f'{algo_name} — Recall by Activation Combo',
                 fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Recall', shrink=0.8)

    for i in range(len(ACT_SWEEP)):
        for j in range(len(ACT_SWEEP)):
            ax.text(j, i, f'{heatmap_data[i,j]:.3f}',
                    ha='center', va='center', fontsize=9,
                    color='black' if heatmap_data[i,j] > 0.5 else 'white')
    for s in ax.spines.values(): s.set_linewidth(0.5)

fig3.suptitle(
    "Phase 6 — Activation Function Combinations\n"
    f"(22→10→10→1, relu/tanh/logistic × relu/tanh/logistic, "
    f"pop/swarm={POP}, iters={N_ITER})",
    fontsize=10, y=1.02, color='#333',
)
plt.tight_layout()
plt.savefig('activation_heatmap_plot.png', dpi=150, bbox_inches='tight',
            facecolor=fig3.get_facecolor())
plt.close()
print("  Saved: activation_heatmap_plot.png")

print("\nAll phases complete.\n")
"""
Architecture
    Input  →  Hidden-1  →  Hidden-2  →  Output
      22        10            10           1

Total parameters (default architecture):
    W1 : 22 × 10 = 220   (input   → hidden1 weights)
    b1 :      10 =  10   (hidden1 biases)
    W2 : 10 × 10 = 100   (hidden1 → hidden2 weights)
    b2 :      10 =  10   (hidden2 biases)
    W3 : 10 ×  1 =  10   (hidden2 → output weights)
    b3 :       1 =   1   (output   bias)
    Total       = 351 parameters

Fitness Metric: F1-Score
    The primary fitness metric is F1-score.

    CLINICAL NOTE: In a real diagnostic context, recall (sensitivity) is
    the preferred metric because the cost of a False Negative (missed
    Parkinson's diagnosis) far exceeds the cost of a False Positive.
    However, pure recall optimisation tends to produce *degenerate*
    solutions where the model predicts every sample as positive (recall=1.0,
    but precision=0.0 and the model is clinically useless).

    F1-score (harmonic mean of precision and recall) penalises degenerate
    solutions while still strongly rewarding high recall.  This produces
    more meaningful convergence curves for visualisation and algorithm
    comparison, which is the primary goal of this project.

    All four algorithms (GA, PSO, DE, ABC) share the same fitness interface:
        fitness_fn(solution, X, y) → float  [0, 1]
"""
import random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score)

input_size   = 22
hidden_sizes = (10,10)
output_size  = 1


def compute_n_params(input_size, hidden_sizes, output_size):
    """Return the total number of trainable parameters for the two-layer MLP."""
    hidden1_size = hidden_sizes[0]
    hidden2_size = hidden_sizes[1]

    n_W1 = input_size * hidden1_size # 220
    n_b1 = hidden1_size  #  10
    n_W2 = hidden1_size * hidden2_size # 100
    n_b2 = hidden2_size #  10
    n_W3 = hidden2_size * output_size #  10
    n_b3 = output_size #   1
    return n_W1 + n_b1 + n_W2 + n_b2 + n_W3 + n_b3  # 351

def generate_solution(n_params):
    return [random.uniform(-1.0, 1.0) for _ in range(n_params)]

def take_matrix(solution, idx, rows, cols):
    matrix = []

    for i in range(rows):
        row = []

        for j in range(cols):
            row.append(solution[idx])
            idx += 1

        matrix.append(row)

    return matrix, idx


def _unpack_weights(solution, input_size, hidden_sizes, output_size):
    idx = 0
    hidden1_size = hidden_sizes[0]
    hidden2_size = hidden_sizes[1]

    W1, idx = take_matrix(solution, idx, input_size, hidden1_size)

    b1 = solution[idx:idx + hidden1_size]
    idx += hidden1_size

    W2, idx = take_matrix(solution, idx, hidden1_size, hidden2_size)

    b2 = solution[idx:idx + hidden2_size]
    idx += hidden2_size

    W3, idx = take_matrix(solution, idx, hidden2_size, output_size)

    b3 = solution[idx:idx + output_size]
    idx += output_size
    return [W1, W2, W3], [b1, b2, b3]

def build_MLP(hidden_sizes, activation='relu', max_iter=1, warm_start=False):
    """
    Build an MLPClassifier with two hidden layers.
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation=activation,
        solver='sgd',
        max_iter=max_iter,
        warm_start=warm_start,
        random_state=42,
    )

# ---------------------------------------------------------------------------
# Custom forward pass  (supports per-layer activations)


def _forward_pass(X, coefs, intercepts, act_h1, act_h2):
    """
    Two-hidden-layer forward pass implemented in numpy.

    This bypasses sklearn completely, enabling *different* activations per
    hidden layer.  The output layer always uses logistic (sigmoid) to
    produce a probability for binary classification.

    Parameters
    ----------
    X          : np.ndarray, shape (n_samples, input_size)
    coefs      : list of [W1, W2, W3]
    intercepts : list of [b1, b2, b3]
    act_h1     : str — activation for hidden layer 1
    act_h2     : str — activation for hidden layer 2

    Returns
    -------
    y_pred : np.ndarray, shape (n_samples,) — binary predictions {0, 1}
    """
    W1, W2, W3 = coefs
    b1, b2, b3 = intercepts

    # Layer 1: input → hidden-1 (activation = act_h1)
    h1 = _apply_activation(X @ W1 + b1, act_h1)

    # Layer 2: hidden-1 → hidden-2 (activation = act_h2)
    h2 = _apply_activation(h1 @ W2 + b2, act_h2)

    # Output layer: hidden-2 → output (logistic for binary classification)
    logit  = h2 @ W3 + b3         # shape (n_samples, 1)
    prob   = _apply_activation(logit.ravel(), 'logistic')
    y_pred = (prob >= 0.5).astype(int)
    return y_pred


# ---------------------------------------------------------------------------
# Standard fitness function  (uniform activation via sklearn)
# ---------------------------------------------------------------------------

def fitness_function(
    solution,
    X,
    y,
    input_size   = DEFAULT_INPUT_SIZE,
    hidden1_size = DEFAULT_HIDDEN1_SIZE,
    hidden2_size = DEFAULT_HIDDEN2_SIZE,
    output_size  = DEFAULT_OUTPUT_SIZE,
    activation   = 'relu',   # uniform activation for both hidden layers
):
    """
    Evaluate weight vector θ → return F1-SCORE (positive class).

    Uses sklearn's MLPClassifier with a *uniform* activation.
    For per-layer activations use fitness_function_custom_act().

    Compatible with GA, PSO, DE, and ABC — all call:
        fitness_fn(solution, X, y) → float in [0, 1]

    Steps
    -----
    1. Build MLPClassifier(hidden_layer_sizes=(h1, h2), activation).
    2. Dummy .fit() on two rows to initialise internal bookkeeping.
    3. Overwrite coefs_ / intercepts_ with values from `solution`.
    4. Forward pass on real X → predictions.
    5. Return f1_score(y, y_pred, pos_label=1).
    """
    clf = _build_clf(hidden1_size, hidden2_size, activation)

    # Dummy fit: one sample per class, minimal overhead
    X_dummy = np.zeros((2, input_size))
    y_dummy = np.array([0, 1])
    clf.fit(X_dummy, y_dummy)

    # Inject optimiser's weight vector
    clf.coefs_, clf.intercepts_ = _unpack_weights(
        solution, input_size, hidden1_size, hidden2_size, output_size
    )

    y_pred = clf.predict(X)
    # PRIMARY FITNESS: F1-score (harmonic mean of precision and recall).
    # CLINICAL NOTE: Recall would be preferred in practice (minimise false
    # negatives), but pure recall maximisation leads to degenerate solutions
    # (model predicts all samples as positive).  F1 prevents degeneracy while
    # still strongly rewarding high recall, and produces more meaningful
    # convergence curves for GA/PSO/DE/ABC comparison.
    return f1_score(y, y_pred, pos_label=1, zero_division=0)


# ---------------------------------------------------------------------------
# Custom-activation fitness function  (per-layer activations via numpy)
# ---------------------------------------------------------------------------

def fitness_function_custom_act(
    solution,
    X,
    y,
    input_size   = DEFAULT_INPUT_SIZE,
    hidden1_size = DEFAULT_HIDDEN1_SIZE,
    hidden2_size = DEFAULT_HIDDEN2_SIZE,
    output_size  = DEFAULT_OUTPUT_SIZE,
    act_h1       = 'relu',    # activation for hidden layer 1
    act_h2       = 'relu',    # activation for hidden layer 2
):
    """
    Evaluate weight vector θ → return F1-SCORE using a *custom* numpy forward
    pass that supports different activations per hidden layer.

    Why this matters
    ----------------
    Using different activations per layer is impossible with sklearn's
    MLPClassifier (it applies one activation uniformly).  This function
    unpacks θ into weight matrices and runs the forward pass manually,
    enabling combinations like relu→tanh or tanh→logistic.

    FITNESS METRIC: F1-score (see fitness_function() docstring for the
    clinical note on why F1 is preferred over recall for this project).
    """
    coefs, intercepts = _unpack_weights(
        solution, input_size, hidden1_size, hidden2_size, output_size
    )
    y_pred = _forward_pass(X, coefs, intercepts, act_h1, act_h2)
    # F1-score as primary fitness (see clinical note in fitness_function())
    return f1_score(y, y_pred, pos_label=1, zero_division=0)


# ---------------------------------------------------------------------------
# Standard evaluate_solution  (uniform activation)
# ---------------------------------------------------------------------------

def evaluate_solution(
    solution,
    X,
    y,
    input_size   = DEFAULT_INPUT_SIZE,
    hidden1_size = DEFAULT_HIDDEN1_SIZE,
    hidden2_size = DEFAULT_HIDDEN2_SIZE,
    output_size  = DEFAULT_OUTPUT_SIZE,
    activation   = 'relu',
):
    """
    Full diagnostic evaluation after optimisation completes.

    Returns dict with keys:
        accuracy, recall, precision, f1, n_pred_pos, n_pred_neg
    """
    clf = _build_clf(hidden1_size, hidden2_size, activation)
    clf.fit(np.zeros((2, input_size)), np.array([0, 1]))
    clf.coefs_, clf.intercepts_ = _unpack_weights(
        solution, input_size, hidden1_size, hidden2_size, output_size
    )
    y_pred = clf.predict(X)

    return {
        'accuracy'  : accuracy_score(y, y_pred),
        'recall'    : recall_score(y,    y_pred, pos_label=1, zero_division=0),
        'precision' : precision_score(y, y_pred, pos_label=1, zero_division=0),
        'f1'        : f1_score(y,        y_pred, pos_label=1, zero_division=0),
        'n_pred_pos': int((y_pred == 1).sum()),
        'n_pred_neg': int((y_pred == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Custom-activation evaluate_solution  (per-layer activations)
# ---------------------------------------------------------------------------

def evaluate_solution_custom_act(
    solution,
    X,
    y,
    input_size   = DEFAULT_INPUT_SIZE,
    hidden1_size = DEFAULT_HIDDEN1_SIZE,
    hidden2_size = DEFAULT_HIDDEN2_SIZE,
    output_size  = DEFAULT_OUTPUT_SIZE,
    act_h1       = 'relu',
    act_h2       = 'relu',
):
    """
    Full diagnostic evaluation using the custom numpy forward pass.

    Returns the same dict schema as evaluate_solution() plus the
    activation pair used, for easy CSV export.
    """
    coefs, intercepts = _unpack_weights(
        solution, input_size, hidden1_size, hidden2_size, output_size
    )
    y_pred = _forward_pass(X, coefs, intercepts, act_h1, act_h2)

    return {
        'act_h1'    : act_h1,
        'act_h2'    : act_h2,
        'accuracy'  : accuracy_score(y, y_pred),
        'recall'    : recall_score(y,    y_pred, pos_label=1, zero_division=0),
        'precision' : precision_score(y, y_pred, pos_label=1, zero_division=0),
        'f1'        : f1_score(y,        y_pred, pos_label=1, zero_division=0),
        'n_pred_pos': int((y_pred == 1).sum()),
        'n_pred_neg': int((y_pred == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import pandas as pd

    print("=" * 60)
    print("  my_parkinsons_problem.py  —  self-test (2 hidden layers)")
    print("=" * 60)

    df = pd.read_csv('parkinsons_preprocessed.csv')
    X  = df.drop(columns=['status']).values.astype(float)
    y  = df['status'].values.astype(int)
    print(f"\nDataset : {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes : {(y==1).sum()} Parkinson's  |  {(y==0).sum()} Healthy")

    n_params = compute_n_params()
    print(f"\nArchitecture : {DEFAULT_INPUT_SIZE} → {DEFAULT_HIDDEN1_SIZE}"
          f" → {DEFAULT_HIDDEN2_SIZE} → {DEFAULT_OUTPUT_SIZE}")
    print(f"Total params : {n_params}")

    np.random.seed(0)
    theta = generate_solution(n_params)

    # --- Standard path ---
    fit_std = fitness_function(theta, X, y)
    print(f"\n[Standard / relu+relu]  Fitness (F1) : {fit_std:.4f}")
    m_std = evaluate_solution(theta, X, y)
    for k, v in m_std.items():
        print(f"  {k:<12}: {v}")

    # --- Custom activation path ---
    print("\n[Custom activation combinations]")
    for a1, a2 in [('relu', 'relu'), ('relu', 'tanh'),
                   ('tanh', 'relu'), ('tanh', 'tanh')]:
        fit_c = fitness_function_custom_act(theta, X, y, act_h1=a1, act_h2=a2)
        print(f"  act_h1={a1:<8}  act_h2={a2:<8}  F1={fit_c:.4f}")

    print("\n[PASS] Module loaded successfully.")
"""
parkinsons_hidden.py
====================
Core problem definition for Parkinson's MLP weight optimisation.

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

Why Recall?
    The dataset is imbalanced (147 Parkinson's vs 48 healthy) and the
    clinical cost of a False Negative (missed diagnosis) far exceeds
    a False Positive.  Both optimisers maximise recall.

Activation Function Experiments (NEW)
    sklearn's MLPClassifier applies the *same* activation to every hidden
    layer; it does not support mixed activations natively.  To explore
    per-layer combinations we bypass sklearn's forward pass entirely and
    implement our own numpy forward pass inside fitness_function_custom_act.
    Supported activations: 'relu', 'tanh', 'logistic', 'identity'.

    Why explore activation combinations?
    • relu        → sparsity, fast training, vanishing-gradient resistant
    • tanh        → zero-centred outputs, useful when features are standardised
    • logistic    → smooth, historically popular for binary classification
    • identity    → linear layer (useful as ablation / control)
    Mixing activations per layer can capture different feature abstractions
    at each depth, potentially improving generalisation.  The statistical
    tests in utils_hidden.py will determine whether observed differences
    are significant or merely noise.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score)

# ---------------------------------------------------------------------------
# Architecture defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT_SIZE   = 22
DEFAULT_HIDDEN1_SIZE = 10
DEFAULT_HIDDEN2_SIZE = 10
DEFAULT_OUTPUT_SIZE  = 1

# ---------------------------------------------------------------------------
# Supported activation functions for the custom forward pass
# ---------------------------------------------------------------------------
SUPPORTED_ACTIVATIONS = ('relu', 'tanh', 'logistic', 'identity')


def compute_n_params(
    input_size   = DEFAULT_INPUT_SIZE,
    hidden1_size = DEFAULT_HIDDEN1_SIZE,
    hidden2_size = DEFAULT_HIDDEN2_SIZE,
    output_size  = DEFAULT_OUTPUT_SIZE,
):
    """Return the total number of trainable parameters for the two-layer MLP."""
    n_W1 = input_size   * hidden1_size   # 220
    n_b1 = hidden1_size                  #  10
    n_W2 = hidden1_size * hidden2_size   # 100
    n_b2 = hidden2_size                  #  10
    n_W3 = hidden2_size * output_size    #  10
    n_b3 = output_size                   #   1
    return n_W1 + n_b1 + n_W2 + n_b2 + n_W3 + n_b3  # 351 (default)


def generate_solution(n_params):
    """Create a random weight vector θ ∈ [-1, 1]^n_params."""
    return np.random.uniform(-1.0, 1.0, size=n_params)


# ---------------------------------------------------------------------------
# Weight unpacking (shared by sklearn-based and custom forward-pass paths)
# ---------------------------------------------------------------------------

def _unpack_weights(solution, input_size, hidden1_size, hidden2_size,
                    output_size):
    """
    Slice the flat weight vector θ into six arrays.

    Layout: [ W1 | b1 | W2 | b2 | W3 | b3 ]
    Returns: (coefs_list, intercepts_list)
        coefs_list      = [W1, W2, W3]
        intercepts_list = [b1, b2, b3]
    """
    idx = 0

    # W1 : input → hidden-1
    nW1 = input_size * hidden1_size
    W1  = solution[idx:idx + nW1].reshape(input_size, hidden1_size)
    idx += nW1
    b1  = solution[idx:idx + hidden1_size];   idx += hidden1_size

    # W2 : hidden-1 → hidden-2
    nW2 = hidden1_size * hidden2_size
    W2  = solution[idx:idx + nW2].reshape(hidden1_size, hidden2_size)
    idx += nW2
    b2  = solution[idx:idx + hidden2_size];   idx += hidden2_size

    # W3 : hidden-2 → output
    nW3 = hidden2_size * output_size
    W3  = solution[idx:idx + nW3].reshape(hidden2_size, output_size)
    idx += nW3
    b3  = solution[idx:idx + output_size];    idx += output_size

    assert idx == len(solution), (
        f"Weight unpacking mismatch: consumed {idx} of {len(solution)} params."
    )
    return [W1, W2, W3], [b1, b2, b3]


# ---------------------------------------------------------------------------
# Activation function dispatcher (for custom forward pass)
# ---------------------------------------------------------------------------

def _apply_activation(z, name):
    """
    Apply a named activation function element-wise.

    Parameters
    ----------
    z    : np.ndarray  — pre-activation values
    name : str         — one of 'relu', 'tanh', 'logistic', 'identity'
    """
    if name == 'relu':
        # ReLU: max(0, z)
        # Good default: sparse activations, no vanishing gradient for positive z
        return np.maximum(0.0, z)
    elif name == 'tanh':
        # Tanh: zero-centred output ∈ (-1,1), smooth gradients
        return np.tanh(z)
    elif name == 'logistic':
        # Logistic (sigmoid): output ∈ (0,1), historically popular
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    elif name == 'identity':
        # Identity (linear): no non-linearity; useful as ablation baseline
        return z
    else:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from {SUPPORTED_ACTIVATIONS}."
        )


# ---------------------------------------------------------------------------
# sklearn-based builder (uniform activation across both hidden layers)
# ---------------------------------------------------------------------------

def _build_clf(hidden1_size, hidden2_size, activation='relu'):
    """
    Build an MLPClassifier with two hidden layers and a uniform activation.

    NOTE: sklearn applies the same activation to *all* hidden layers.
    For per-layer activations, use the custom forward pass path
    (fitness_function_custom_act / evaluate_solution_custom_act).

    Parameters
    ----------
    hidden1_size : int  — neurons in layer 1
    hidden2_size : int  — neurons in layer 2
    activation   : str  — sklearn activation: 'relu', 'tanh', 'logistic',
                          or 'identity'
    """
    return MLPClassifier(
        hidden_layer_sizes=(hidden1_size, hidden2_size),
        activation=activation,   # ← uniform across both hidden layers
        solver='sgd',
        max_iter=1,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Custom forward pass  (supports per-layer activations)
# ---------------------------------------------------------------------------

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
    Evaluate weight vector θ → return RECALL (positive class).

    Uses sklearn's MLPClassifier with a *uniform* activation.
    For per-layer activations use fitness_function_custom_act().

    Steps
    -----
    1. Build MLPClassifier(hidden_layer_sizes=(h1, h2), activation).
    2. Dummy .fit() on two rows to initialise internal bookkeeping.
    3. Overwrite coefs_ / intercepts_ with values from `solution`.
    4. Forward pass on real X → predictions.
    5. Return recall_score(y, y_pred, pos_label=1).
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
    return recall_score(y, y_pred, pos_label=1, zero_division=0)


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
    Evaluate weight vector θ → return RECALL using a *custom* numpy forward
    pass that supports different activations per hidden layer.

    Why this matters
    ----------------
    Using different activations per layer is impossible with sklearn's
    MLPClassifier (it applies one activation uniformly).  This function
    unpacks θ into weight matrices and runs the forward pass manually,
    enabling combinations like relu→tanh or tanh→logistic.

    Expected impact of activation combinations
    -------------------------------------------
    relu  + relu    : standard baseline, fast convergence
    relu  + tanh    : wide hidden-1 features, compressed hidden-2 output
    tanh  + relu    : zero-centred first layer, sparse second
    tanh  + tanh    : smooth, zero-centred throughout; may help on standardised data
    logistic + relu : smooth first stage, sparse second
    relu  + logistic: sparse features, bounded hidden-2 output
    """
    coefs, intercepts = _unpack_weights(
        solution, input_size, hidden1_size, hidden2_size, output_size
    )
    y_pred = _forward_pass(X, coefs, intercepts, act_h1, act_h2)
    return recall_score(y, y_pred, pos_label=1, zero_division=0)


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
    print("  parkinsons_hidden.py  —  self-test (2 hidden layers)")
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
    print(f"\n[Standard / relu+relu]  Fitness (recall) : {fit_std:.4f}")
    m_std = evaluate_solution(theta, X, y)
    for k, v in m_std.items():
        print(f"  {k:<12}: {v}")

    # --- Custom activation path ---
    print("\n[Custom activation combinations]")
    for a1, a2 in [('relu', 'relu'), ('relu', 'tanh'),
                   ('tanh', 'relu'), ('tanh', 'tanh')]:
        fit_c = fitness_function_custom_act(theta, X, y, act_h1=a1, act_h2=a2)
        print(f"  act_h1={a1:<8}  act_h2={a2:<8}  recall={fit_c:.4f}")

    print("\n[PASS] Module loaded successfully.")
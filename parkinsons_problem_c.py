"""
parkinsons_problem.py
=====================
Problem definition module for the Parkinson's Disease Classification project.

This module is the **shared foundation** for every optimization algorithm
in the project.  Both the Genetic Algorithm and the swarm/nature-inspired
algorithm import `generate_solution` and `fitness_function` from here, so
neither algorithm knows or cares about Parkinson's disease — they simply ask
"how good is this vector?" and this module answers.

Architecture
------------
We use a one-hidden-layer Multi-Layer Perceptron (MLP):

    Input layer  →  Hidden layer  →  Output layer
       22 neurons      10 neurons       1 neuron
       (features)    (ReLU / default)  (sigmoid → binary)

Total parameters (weights + biases):
    W1 : 22 × 10 = 220   (input  → hidden weight matrix)
    b1 :      10 =  10   (hidden layer bias vector)
    W2 : 10 ×  1 =  10   (hidden → output weight matrix)
    b2 :       1 =   1   (output layer bias scalar)
    ────────────────────
    Total       = 241 parameters  ← this is n_params

The optimizer searches R^241 for the vector θ that maximises RECALL
for the positive class (Parkinson's, label = 1).

Why Recall instead of Accuracy?
--------------------------------
The dataset is imbalanced: 147 Parkinson's patients vs 48 healthy controls
(75.4% positive rate).  A naive classifier that predicts *everyone* as
positive would score 75.4% accuracy while having zero clinical value.

In medical screening, the cost of a False Negative (missing a sick patient)
vastly outweighs the cost of a False Positive (unnecessary follow-up).
Recall directly minimises False Negatives:

    Recall = TP / (TP + FN)

By optimising recall, both GA and PSO are forced to find weights that
correctly identify as many Parkinson's patients as possible — which is
precisely the clinical objective.

    Metric          Formula                 What it misses
    ──────────────────────────────────────────────────────
    Accuracy        (TP+TN)/(P+N)           Hides imbalance, rewards lazy models
    Recall (ours)   TP/(TP+FN)              May sacrifice precision
    Precision       TP/(TP+FP)              Ignores false negatives
    F1              2·P·R/(P+R)             Balanced alternative (used in report)

Recall is used as the optimization fitness signal.
The full report will also present accuracy, precision, and F1 for context.

Why MLPClassifier instead of a custom network?
----------------------------------------------
sklearn's MLPClassifier lets us *inject* external weights via the
`coefs_` and `intercepts_` attributes without calling `.fit()`.
This decouples the network's forward pass from gradient-based training
so any population-based optimizer can drive it.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# ---------------------------------------------------------------------------
# Architecture constants
# These are module-level defaults so they stay in one place.
# Every function that needs them accepts explicit arguments, keeping
# functions fully self-contained and testable.
# ---------------------------------------------------------------------------
DEFAULT_INPUT_SIZE  = 22   # 22 biomedical voice features
DEFAULT_HIDDEN_SIZE = 10   # one hidden layer, 10 neurons
DEFAULT_OUTPUT_SIZE = 1    # binary: 0 = healthy, 1 = Parkinson's


def compute_n_params(input_size: int = DEFAULT_INPUT_SIZE,
                     hidden_size: int = DEFAULT_HIDDEN_SIZE,
                     output_size: int = DEFAULT_OUTPUT_SIZE) -> int:
    """
    Return the total number of trainable parameters for the MLP.

    For a network  input_size → hidden_size → output_size  the count is:

        W1 weights : input_size  * hidden_size
        b1 biases  :               hidden_size
        W2 weights : hidden_size * output_size
        b2 biases  :               output_size

    Parameters
    ----------
    input_size  : int  — number of input features
    hidden_size : int  — neurons in the single hidden layer
    output_size : int  — neurons in the output layer (1 for binary)

    Returns
    -------
    int — total number of real-valued parameters in θ
    """
    n_W1 = input_size  * hidden_size   # 22 * 10 = 220
    n_b1 = hidden_size                 #       10
    n_W2 = hidden_size * output_size   # 10 *  1 =  10
    n_b2 = output_size                 #        1
    return n_W1 + n_b1 + n_W2 + n_b2  # 241


def generate_solution(n_params: int,
                      low: float = -1.0,
                      high: float = 1.0) -> np.ndarray:
    """
    Create a random weight vector (a candidate solution, θ).

    This is the **solution representation** required by the project spec.
    Each optimizer calls this function to populate its initial population.

    The vector is drawn uniformly from [low, high].  This range works well
    as a starting point because:
      • Xavier / Glorot initialisation (standard in DL) also keeps initial
        weights small to avoid saturating activation functions.
      • The dataset is already standardised, so feature magnitudes are
        comparable and small weights are appropriate.

    Parameters
    ----------
    n_params : int   — length of θ, i.e. compute_n_params(...)
    low      : float — lower bound of the uniform draw  (default -1.0)
    high     : float — upper bound of the uniform draw  (default +1.0)

    Returns
    -------
    np.ndarray, shape (n_params,)  — one random candidate solution
    """
    return np.random.uniform(low, high, size=n_params)


def _unpack_weights(solution: np.ndarray,
                    input_size: int,
                    hidden_size: int,
                    output_size: int):
    """
    Slice the flat weight vector θ into the four separate weight/bias arrays
    that sklearn's MLPClassifier expects.

    MLPClassifier stores weights in two lists:
        coefs_       : list of 2D arrays  [W1, W2]
        intercepts_  : list of 1D arrays  [b1, b2]

    The flat vector is laid out as:
        [ W1 (flattened, row-major) | b1 | W2 (flattened, row-major) | b2 ]

    Parameters
    ----------
    solution    : np.ndarray, shape (n_params,)
    input_size  : int
    hidden_size : int
    output_size : int

    Returns
    -------
    coefs_      : list of np.ndarray  — [W1 (input×hidden), W2 (hidden×output)]
    intercepts_ : list of np.ndarray  — [b1 (hidden,),      b2 (output,)      ]
    """
    idx = 0  # sliding index into solution

    # --- W1: input_size rows, hidden_size columns ---
    n_W1 = input_size * hidden_size
    W1 = solution[idx: idx + n_W1].reshape(input_size, hidden_size)
    idx += n_W1

    # --- b1: one bias per hidden neuron ---
    n_b1 = hidden_size
    b1 = solution[idx: idx + n_b1]
    idx += n_b1

    # --- W2: hidden_size rows, output_size columns ---
    n_W2 = hidden_size * output_size
    W2 = solution[idx: idx + n_W2].reshape(hidden_size, output_size)
    idx += n_W2

    # --- b2: one bias per output neuron ---
    n_b2 = output_size
    b2 = solution[idx: idx + n_b2]
    idx += n_b2

    assert idx == len(solution), (
        f"Weight unpacking mismatch: consumed {idx} of {len(solution)} params."
    )

    return [W1, W2], [b1, b2]


def fitness_function(solution: np.ndarray,
                     X: np.ndarray,
                     y: np.ndarray,
                     input_size:  int = DEFAULT_INPUT_SIZE,
                     hidden_size: int = DEFAULT_HIDDEN_SIZE,
                     output_size: int = DEFAULT_OUTPUT_SIZE) -> float:
    """
    Evaluate a weight vector θ and return its RECALL for the positive class.

    This is the **fitness function** required by the project spec.
    It is the only link between an optimizer and the machine-learning task:
    the optimizer knows nothing about neural networks — it just maximises
    whatever number this function returns.

    Fitness = Recall (sensitivity) for class 1 (Parkinson's):

        Recall = TP / (TP + FN)

    A recall of 1.0 means the model caught every single Parkinson's patient.
    A recall of 0.0 means it missed all of them.

    Clinical justification
    ----------------------
    False Negatives in Parkinson's screening mean a patient goes undiagnosed
    and untreated.  Early intervention significantly improves quality of life,
    so we must minimise FN even at the cost of some False Positives (which
    simply lead to further clinical testing, not harm).

    Optimising recall directly aligns the search with this clinical priority.

    Edge case
    ---------
    If the model predicts all samples as class 0 (healthy), recall = 0.0.
    This is the worst possible fitness and will be eliminated by selection.
    The `zero_division=0` parameter handles the case where TP + FN = 0
    (impossible here since y always contains positive samples, but included
    for robustness).

    Steps
    -----
    1. Build an MLPClassifier with the right architecture but **no training**.
    2. Call a dummy `.fit()` with two dummy rows so sklearn initialises
       its internal bookkeeping (layer sizes, activation, etc.).
    3. Overwrite `coefs_` and `intercepts_` with the values from `solution`.
    4. Run `.predict()` on the real dataset X.
    5. Return recall_score(y, y_pred, pos_label=1) as the fitness value.

    Parameters
    ----------
    solution    : np.ndarray, shape (n_params,) — the weight vector θ
    X           : np.ndarray, shape (n_samples, input_size) — feature matrix
    y           : np.ndarray, shape (n_samples,)            — labels (0 or 1)
    input_size  : int — must match X.shape[1]
    hidden_size : int — hidden layer width
    output_size : int — output layer width (1 for binary classification)

    Returns
    -------
    float — recall in [0, 1] for class 1  (maximised by the optimizer)
    """
    # ------------------------------------------------------------------
    # 1. Build the classifier shell (no training yet).
    # ------------------------------------------------------------------
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_size,),
        activation='relu',
        solver='sgd',        # solver choice doesn't matter — we never train
        max_iter=1,
        random_state=42,
    )

    # ------------------------------------------------------------------
    # 2. Minimal dummy fit to force sklearn to initialise its weight
    #    attributes (coefs_, intercepts_, n_layers_, etc.).
    #    We use exactly two samples — one per class — to satisfy sklearn's
    #    requirement that at least n_classes samples exist in y.
    #    These dummy weights are overwritten immediately after.
    # ------------------------------------------------------------------
    X_dummy = np.zeros((2, input_size))
    y_dummy = np.array([0, 1])
    clf.fit(X_dummy, y_dummy)

    # ------------------------------------------------------------------
    # 3. Inject the optimizer's weight vector into the network.
    # ------------------------------------------------------------------
    clf.coefs_, clf.intercepts_ = _unpack_weights(
        solution, input_size, hidden_size, output_size
    )

    # ------------------------------------------------------------------
    # 4. Forward pass on the real data — no gradients, no training.
    # ------------------------------------------------------------------
    y_pred = clf.predict(X)

    # ------------------------------------------------------------------
    # 5. Fitness = Recall for the positive class (Parkinson's = 1).
    # ------------------------------------------------------------------
    return recall_score(y, y_pred, pos_label=1, zero_division=0)


def evaluate_solution(solution: np.ndarray,
                      X: np.ndarray,
                      y: np.ndarray,
                      input_size:  int = DEFAULT_INPUT_SIZE,
                      hidden_size: int = DEFAULT_HIDDEN_SIZE,
                      output_size: int = DEFAULT_OUTPUT_SIZE) -> dict:
    """
    Full diagnostic evaluation of a weight vector θ.

    Returns all four key metrics so the report can present a complete
    picture beyond the single fitness signal used during optimization.

    This function is called AFTER optimization completes — not during it.
    The optimizer only ever sees `fitness_function` (recall).

    Metrics returned
    ----------------
    accuracy  : (TP+TN) / (P+N)          — overall correctness
    recall    : TP / (TP+FN)             — sensitivity; the fitness signal
    precision : TP / (TP+FP)             — how trustworthy positive predictions are
    f1        : 2·precision·recall / (precision+recall)  — harmonic mean
    n_pred_pos: number of samples predicted as positive (Parkinson's)
    n_pred_neg: number of samples predicted as negative (Healthy)

    Parameters
    ----------
    solution    : np.ndarray, shape (n_params,)
    X           : np.ndarray, shape (n_samples, input_size)
    y           : np.ndarray, shape (n_samples,)
    input_size, hidden_size, output_size : ints — architecture

    Returns
    -------
    dict with keys: accuracy, recall, precision, f1, n_pred_pos, n_pred_neg
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_size,),
        activation='relu',
        solver='sgd',
        max_iter=1,
        random_state=42,
    )
    X_dummy = np.zeros((2, input_size))
    y_dummy = np.array([0, 1])
    clf.fit(X_dummy, y_dummy)

    clf.coefs_, clf.intercepts_ = _unpack_weights(
        solution, input_size, hidden_size, output_size
    )
    y_pred = clf.predict(X)

    return {
        'accuracy' : accuracy_score(y, y_pred),
        'recall'   : recall_score(y, y_pred,    pos_label=1, zero_division=0),
        'precision': precision_score(y, y_pred, pos_label=1, zero_division=0),
        'f1'       : f1_score(y, y_pred,        pos_label=1, zero_division=0),
        'n_pred_pos': int((y_pred == 1).sum()),
        'n_pred_neg': int((y_pred == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Quick self-test — run this file directly to verify everything works.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import pandas as pd

    print("=" * 55)
    print("  parkinsons_problem.py  —  self-test")
    print("=" * 55)

    # Load data
    df = pd.read_csv('parkinsons_preprocessed.csv')
    X = df.drop(columns=['status']).values
    y = df['status'].values
    print(f"\nDataset : {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes : {(y==1).sum()} Parkinson's  |  {(y==0).sum()} Healthy")
    print(f"Positive rate (majority baseline recall): {(y==1).mean():.4f}")

    # Architecture
    n_params = compute_n_params()
    print(f"\nArchitecture : {DEFAULT_INPUT_SIZE} → {DEFAULT_HIDDEN_SIZE}"
          f" → {DEFAULT_OUTPUT_SIZE}")
    print(f"Total params : {n_params}")

    # Generate one random solution and evaluate it
    np.random.seed(0)
    theta = generate_solution(n_params)
    print(f"\nRandom θ  :  shape={theta.shape},  "
          f"min={theta.min():.4f},  max={theta.max():.4f}")

    # Fitness signal (recall)
    fitness = fitness_function(theta, X, y)
    print(f"\nFitness (recall, pos=1) : {fitness:.4f}  ({fitness*100:.1f}%)")

    # Full diagnostic
    metrics = evaluate_solution(theta, X, y)
    print(f"\nFull evaluation on random θ:")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}   ← fitness signal")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Predicted positive : {metrics['n_pred_pos']}")
    print(f"  Predicted negative : {metrics['n_pred_neg']}")
    print("\n[PASS] Module loaded and executed successfully.")

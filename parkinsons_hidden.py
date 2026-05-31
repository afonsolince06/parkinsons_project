"""
parkinsons_hidden.py
====================
Problem definition module for the Parkinson's Disease Classification project
— **two hidden layers** variant.

Architecture (CHANGED from single-layer version)
-------------------------------------------------
    Input  →  Hidden-1  →  Hidden-2  →  Output
      22        10            10           1

Total parameters:
    W1 : 22 × 10 = 220   (input   → hidden-1 weights)
    b1 :      10 =  10   (hidden-1 biases)
    W2 : 10 × 10 = 100   (hidden-1 → hidden-2 weights)   ← NEW
    b2 :      10 =  10   (hidden-2 biases)                ← NEW
    W3 : 10 ×  1 =  10   (hidden-2 → output  weights)
    b3 :       1 =   1   (output   bias)
    ─────────────────────
    Total       = 351 parameters

Why Recall?
-----------
Same reasoning as the single-layer module: the dataset is imbalanced
(147 PD vs 48 healthy) and the clinical cost of a False Negative is far
greater than a False Positive.  Both optimisers maximise recall.

sklearn trick
-------------
MLPClassifier accepts `hidden_layer_sizes=(h1, h2)` for a two-layer
network.  We inject weights via `coefs_` / `intercepts_` after a dummy
`.fit()` exactly as before — no backpropagation is used.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score)

# ---------------------------------------------------------------------------
# Architecture defaults  (CHANGED: two hidden layers)
# ---------------------------------------------------------------------------
DEFAULT_INPUT_SIZE   = 22
DEFAULT_HIDDEN1_SIZE = 10   # first  hidden layer  ← kept from original
DEFAULT_HIDDEN2_SIZE = 10   # second hidden layer  ← NEW
DEFAULT_OUTPUT_SIZE  = 1


def compute_n_params(
    input_size:   int = DEFAULT_INPUT_SIZE,
    hidden1_size: int = DEFAULT_HIDDEN1_SIZE,
    hidden2_size: int = DEFAULT_HIDDEN2_SIZE,   # NEW argument
    output_size:  int = DEFAULT_OUTPUT_SIZE,
) -> int:
    """
    Return the total number of trainable parameters for the two-layer MLP.

    CHANGED: added W2/b2 for the second hidden layer.

        W1 : input_size   * hidden1_size
        b1 :                hidden1_size
        W2 : hidden1_size * hidden2_size   ← NEW
        b2 :                hidden2_size   ← NEW
        W3 : hidden2_size * output_size
        b3 :                output_size
    """
    n_W1 = input_size   * hidden1_size   # 220
    n_b1 = hidden1_size                  #  10
    n_W2 = hidden1_size * hidden2_size   # 100  ← NEW
    n_b2 = hidden2_size                  #  10  ← NEW
    n_W3 = hidden2_size * output_size    #  10
    n_b3 = output_size                   #   1
    return n_W1 + n_b1 + n_W2 + n_b2 + n_W3 + n_b3  # 351


def generate_solution(n_params: int,
                      low: float = -1.0,
                      high: float = 1.0) -> np.ndarray:
    """
    Create a random weight vector θ of length n_params.

    Unchanged from the single-layer version — the solution representation
    is simply a longer flat vector (351 vs 241 values).
    """
    return np.random.uniform(low, high, size=n_params)


def _unpack_weights(
    solution:     np.ndarray,
    input_size:   int,
    hidden1_size: int,
    hidden2_size: int,   # NEW
    output_size:  int,
):
    """
    Slice the flat weight vector θ into the six arrays that sklearn's
    MLPClassifier expects for a two-hidden-layer network.

    CHANGED: now unpacks six arrays instead of four.

    Layout of the flat vector:
        [ W1 | b1 | W2 | b2 | W3 | b3 ]

    MLPClassifier stores them as:
        coefs_      : [W1, W2, W3]
        intercepts_ : [b1, b2, b3]
    """
    idx = 0

    # --- W1: input → hidden-1 ---
    n_W1 = input_size * hidden1_size
    W1 = solution[idx: idx + n_W1].reshape(input_size, hidden1_size)
    idx += n_W1

    # --- b1: hidden-1 biases ---
    b1 = solution[idx: idx + hidden1_size]
    idx += hidden1_size

    # --- W2: hidden-1 → hidden-2 (NEW) ---
    n_W2 = hidden1_size * hidden2_size
    W2 = solution[idx: idx + n_W2].reshape(hidden1_size, hidden2_size)
    idx += n_W2

    # --- b2: hidden-2 biases (NEW) ---
    b2 = solution[idx: idx + hidden2_size]
    idx += hidden2_size

    # --- W3: hidden-2 → output ---
    n_W3 = hidden2_size * output_size
    W3 = solution[idx: idx + n_W3].reshape(hidden2_size, output_size)
    idx += n_W3

    # --- b3: output bias ---
    b3 = solution[idx: idx + output_size]
    idx += output_size

    assert idx == len(solution), (
        f"Weight unpacking mismatch: consumed {idx} of {len(solution)} params."
    )

    # coefs_ expects one matrix per *transition*, intercepts_ one vector per layer
    return [W1, W2, W3], [b1, b2, b3]


def _build_clf(hidden1_size: int, hidden2_size: int) -> MLPClassifier:
    """
    Helper: build a skeleton MLPClassifier with the two-layer architecture.
    CHANGED: hidden_layer_sizes is now a 2-tuple.
    """
    return MLPClassifier(
        hidden_layer_sizes=(hidden1_size, hidden2_size),   # ← 2 hidden layers
        activation='relu',
        solver='sgd',
        max_iter=1,
        random_state=42,
    )


def fitness_function(
    solution:     np.ndarray,
    X:            np.ndarray,
    y:            np.ndarray,
    input_size:   int = DEFAULT_INPUT_SIZE,
    hidden1_size: int = DEFAULT_HIDDEN1_SIZE,
    hidden2_size: int = DEFAULT_HIDDEN2_SIZE,   # NEW
    output_size:  int = DEFAULT_OUTPUT_SIZE,
) -> float:
    """
    Evaluate a weight vector θ and return its RECALL for the positive class.

    CHANGED: dummy fit and weight injection now handle the three-matrix
    architecture (W1/b1, W2/b2, W3/b3).

    Steps
    -----
    1. Build an MLPClassifier with hidden_layer_sizes=(h1, h2) — no training.
    2. Dummy .fit() on two rows so sklearn initialises internal bookkeeping.
    3. Overwrite coefs_ / intercepts_ with the values from `solution`.
    4. Forward pass on real data X → predictions.
    5. Return recall_score(y, y_pred, pos_label=1).
    """
    clf = _build_clf(hidden1_size, hidden2_size)

    # Dummy fit — only two samples needed, one per class
    X_dummy = np.zeros((2, input_size))
    y_dummy = np.array([0, 1])
    clf.fit(X_dummy, y_dummy)

    # Inject optimiser's weight vector
    clf.coefs_, clf.intercepts_ = _unpack_weights(
        solution, input_size, hidden1_size, hidden2_size, output_size
    )

    y_pred = clf.predict(X)
    return recall_score(y, y_pred, pos_label=1, zero_division=0)


def evaluate_solution(
    solution:     np.ndarray,
    X:            np.ndarray,
    y:            np.ndarray,
    input_size:   int = DEFAULT_INPUT_SIZE,
    hidden1_size: int = DEFAULT_HIDDEN1_SIZE,
    hidden2_size: int = DEFAULT_HIDDEN2_SIZE,   # NEW
    output_size:  int = DEFAULT_OUTPUT_SIZE,
) -> dict:
    """
    Full diagnostic evaluation after optimisation completes.

    CHANGED: builds two-hidden-layer classifier; otherwise identical
    in structure and return keys to the single-layer version.

    Returns
    -------
    dict with keys: accuracy, recall, precision, f1, n_pred_pos, n_pred_neg
    """
    clf = _build_clf(hidden1_size, hidden2_size)
    clf.fit(np.zeros((2, input_size)), np.array([0, 1]))

    clf.coefs_, clf.intercepts_ = _unpack_weights(
        solution, input_size, hidden1_size, hidden2_size, output_size
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
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import pandas as pd

    print("=" * 55)
    print("  parkinsons_hidden.py  —  self-test (2 hidden layers)")
    print("=" * 55)

    df = pd.read_csv('parkinsons_preprocessed.csv')
    X  = df.drop(columns=['status']).values
    y  = df['status'].values
    print(f"\nDataset : {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes : {(y==1).sum()} Parkinson's  |  {(y==0).sum()} Healthy")

    n_params = compute_n_params()
    print(f"\nArchitecture : {DEFAULT_INPUT_SIZE} → {DEFAULT_HIDDEN1_SIZE}"
          f" → {DEFAULT_HIDDEN2_SIZE} → {DEFAULT_OUTPUT_SIZE}")
    print(f"Total params : {n_params}  (was 241 with one hidden layer)")

    np.random.seed(0)
    theta = generate_solution(n_params)
    fitness = fitness_function(theta, X, y)
    print(f"\nRandom θ  :  shape={theta.shape}")
    print(f"Fitness (recall) : {fitness:.4f}")

    metrics = evaluate_solution(theta, X, y)
    print(f"\nFull evaluation:")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")
    print("\n[PASS] Module loaded successfully.")

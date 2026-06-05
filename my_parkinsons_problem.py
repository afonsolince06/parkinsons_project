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


def unpack_weights(solution, input_size, hidden_sizes, output_size):
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

    coefs = [
        np.array(W1),
        np.array(W2),
        np.array(W3)
    ]

    intercepts = [
        np.array(b1),
        np.array(b2),
        np.array(b3)
    ]
    return coefs, intercepts

def fitness_function(solution, X, y, input_size, hidden_sizes, output_size):
    """
    Evaluate one solution and return its fitness.

    In this project, a solution is a vector with all neural network weights.
    The function loads those weights into the MLP and returns the F1-score.
    """

    # Create the MLP structure
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='sgd',
        max_iter=1,
        random_state=42
    )

    # Dummy fit only to make sklearn create the internal structure
    X_dummy = np.zeros((2, input_size))
    y_dummy = np.array([0, 1])
    clf.fit(X_dummy, y_dummy)

    # Transform the solution vector into weights and biases
    coefs, intercepts = unpack_weights(
        solution,
        input_size,
        hidden_sizes,
        output_size
    )
    # Put our GA weights inside the neural network
    clf.coefs_ = coefs
    clf.intercepts_ = intercepts

    y_pred = clf.predict(X)

    return f1_score(y, y_pred, zero_division=0)

def evaluate_solution(solution, X, y, input_size, hidden_sizes, output_size):
    """
    Evaluate the final solution using several metrics.
    This is used after the optimization, not inside the GA loop.
    """

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='sgd',
        max_iter=1,
        random_state=42
    )

    # Dummy fit to initialize sklearn's internal structure
    X_dummy = np.zeros((2, input_size))
    y_dummy = np.array([0, 1])
    clf.fit(X_dummy, y_dummy)

    # Convert solution vector into MLP weights
    coefs, intercepts = unpack_weights(
        solution,
        input_size,
        hidden_sizes,
        output_size
    )

    clf.coefs_ = coefs
    clf.intercepts_ = intercepts

    y_pred = clf.predict(X)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred, pos_label=1, zero_division=0),
        "precision": precision_score(y, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "predicted_positive": int((y_pred == 1).sum()),
        "predicted_negative": int((y_pred == 0).sum())
    }
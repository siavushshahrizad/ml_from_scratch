"""
File: utils.py
Date: 7 July 25
--------------------
SUMMARY
--------------------
The file contains the functions the 
main programme uses.
"""


import numpy as np
import pandas as pd


FILE = "./data/breast-cancer-wisconsin.data"
MISSING_VALUE = "?"

def normalise_data(X):
    mean = np.mean(X, axis=0)       # Technically inefficient as mean recomputed in std
    std = np.std(X, axis=0)
    return (X - mean) / std


def load_and_clean_data(file):
    """
    Loads and cleans data, i.e. documentation says 
    there are 16 missing values somewhere.
    See documentation what exactly these vars are, 
    """
    data = pd.read_csv(file, header=None)           # Contains no headers
    data = data.replace(MISSING_VALUE, np.nan)
    data = data.dropna()
    data = data.to_numpy()
    data = data.astype(int)     # Doc tells us that all numbers are 1-10, but some get converted into str
    X = data[:, 1:-1]        # Last col is the target; first col is ID
    y = data[:, -1:]
    return X, y

def create_data_split(X, y, ratio):
    """
    Returns a split of two datasets when one is inputted.     
    The split is random and the point of the func is to split
    datasets into train, test, validation.
    """
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    cutoff = int(n * ratio)
    indices_set1 = indices[:cutoff]
    indices_set2 = indices[cutoff:]

    X_1 = X[indices_set1]
    X_2 = X[indices_set2]
    y_1 = y[indices_set1]
    y_2 = y[indices_set2]

    return X_1, X_2, y_1, y_2

def forward_pass(X, w):
    """
    This version returns logits and predictions. 
    Logits are returned for cross-entropy func.    
    """
    z = X @ w
    y_hat = 1 / (1 + np.exp(-z))
    y_hat.reshape(-1, 1)        # Or letter shape mismatch in gradient descent
    return y_hat, z

def mean_logistic_cross_entropy(logits, y, w, l=0.01):
    """
    L1 regularisation is applied to this loss func.
    The way this is implemented, e.g. logits are consumed
    rather than y_hat and then logaddexp, is supposed to 
    be more stable according to Grosse's notes.
    """
    loss = y.T @ np.logaddexp(0, -logits) + (1 - y).T @ np.logaddexp(0, logits)
    mean_loss_with_l1 = (loss +  l * np.sum(np.abs(w))) / y.shape[0]    
    return mean_loss_with_l1.item()     # So scalar is actually returned

def simple_gradient_descent(
        w, 
        X, 
        y, 
        num_epochs=50, 
        l=0.01,
        alpha=0.01 
        ):
    """
    This is a sanity check function to see if the
    most basic version of gradient descent is working.
    """
    trained_w = np.copy(w)
    for _ in range(num_epochs):
        y_hat, _ = forward_pass(X, trained_w)
        gradient = (X.T @ (y_hat - y)) / len(y)
        l1_gradient = l * (trained_w / np.abs(trained_w))       # Should cause problems if div by 0; np.sign func needed?
        final_gradient = gradient + l1_gradient
        trained_w -= alpha * final_gradient
    
    return trained_w

def early_stopping_gradient_descent(
        w, 
        X_train, 
        y_train,
        X_val,
        y_val,
        min_epochs=5,
        l=0.01,
        threshold=1.0,
        alpha=0.01
        ):
    """
    Func applies and early_stopping variant of 
    gradient descent to prevent overfitting based on
    criterion parameter which is by default 5. This 
    means that the algorithm stops gradients descent if 
    the current validation loss is 5% higher or more
    than the best validation loss. Note that training
    runs for a minimum amount of time.
    """
    trained_w = np.copy(w)
    checkpoint_w = np.copy(w)
    optimum_loss = float("inf")
    epochs = 0
    
    # Gradient descent loop
    while True:
        epochs += 1 
        # Training via gradient descent
        y_hat, _ = forward_pass(X_train, trained_w)
        gradient = (X_train.T @ (y_hat - y_train)) / len(y_train)
        l1_gradient = l * (trained_w / np.abs(trained_w))
        final_gradient = gradient + l1_gradient
        trained_w -= alpha * final_gradient
        
        # Potential early stopping
        _, z_val = forward_pass(X_val, trained_w)
        validation_loss = mean_logistic_cross_entropy(
            z_val, 
            y_val, 
            trained_w
        )
        generalisation_loss = (validation_loss / optimum_loss - 1) * 100
        stopping_criterion = generalisation_loss > threshold

        if stopping_criterion and epochs >= min_epochs:
            break
        if validation_loss < optimum_loss: 
            optimum_loss = validation_loss
            checkpoint_w = np.copy(trained_w)

    return checkpoint_w, epochs

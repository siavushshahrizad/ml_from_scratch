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

def process_y_labels(y):
    """
    The data set uses 2 for benign and 4 for 
    malignant cases. This causes a bunch of 
    errors if not changed.

    After processing 0 means benign and 1
    malignant.
    """
    return (y == 4).astype(int)

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
    y_hat = y_hat.reshape(-1, 1)        # Or letter shape mismatch in gradient descent
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
    for i in range(num_epochs):
        y_hat, logits = forward_pass(X, trained_w)      # Remove logits later
        gradient = (X.T @ (y_hat - y)) / len(y)
        l1_gradient = l * (trained_w / np.abs(trained_w))       # Should cause problems if div by 0; np.sign func needed?
        final_gradient = gradient + l1_gradient
        trained_w -= alpha * final_gradient

        # if i % 1000 == 0:
        #     print("Epoch: ", i)
        #     print("Avg weights: ", trained_w.mean())
        #     loss = mean_logistic_cross_entropy(logits, y, trained_w)
        #     print("Train loss: ", loss)

    return trained_w

def early_stopping_gradient_descent(
        w, 
        X_train, 
        y_train,
        X_val,
        y_val,
        patience=5,
        l=0.01,
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
    epochs_trained = 0
    epochs_worse = 0
    
    # Gradient descent loop
    while epochs_worse < patience:
        epochs_trained += 1
        # print(epochs_trained)
        # if epochs_trained == 10:
        #     break

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
        # print("Val: ", validation_loss)
        # print("Best: ", optimum_loss)
        if validation_loss > optimum_loss:
            epochs_worse += 1
        else:
            epochs_worse = 0
            optimum_loss = validation_loss
            checkpoint_w = np.copy(trained_w)

    return checkpoint_w, epochs_trained

def adam(
        w,
        X,
        y,
        decay1=0.9,
        decay2=0.999,
        eps=1e-8,
        time_steps=50,
        alpha=0.01,
        l=0.01
        ):
    """
    This func implements the Adam optimiser (simple version)
    from the Klingma and Lei Ba (2015) paper. For what 
    the exact params are, check their paper:     
    https://arxiv.org/abs/1412.6980
    """
    trained_w = np.copy(w)
    mavg1 = np.zeros(w.shape[0]).reshape(-1, 1)
    mavg2 = np.zeros(w.shape[0]).reshape(-1, 1)

    for step in range(1, time_steps + 1):
        y_hat, _ = forward_pass(X, trained_w)
        gradient = (X.T @ (y_hat - y)) / len(y)
        l1_gradient = l * (trained_w / np.abs(trained_w))       # Sample problem as in simple GD func
        final_gradient = gradient + l1_gradient
        mavg1 = decay1 * mavg1 + (1 - decay1) * final_gradient
        mavg2 = decay2 * mavg2 + (1 - decay2) * np.square(final_gradient)
        mavg1_cor = mavg1 / (1 - decay1 ** step)
        mavg2_cor = mavg2 / (1 - decay2 ** step)
        trained_w -= alpha  * mavg1_cor / (np.sqrt(mavg2_cor) + eps)

    return trained_w

def early_stopping_adam(
    w,
    X_train,
    y_train,
    X_val,
    y_val,
    decay1=0.9,
    decay2=0.999,
    eps=1e-8,
    patience=5,
    alpha=0.01,
    l=0.01,
    ):
    """
    Func combines early stopping with Adam.
    See previous funcs above for more detail.
    """
    trained_w = np.copy(w)
    checkpoint_w = np.copy(w)
    optimum_loss = float("inf")
    epochs_worse = 0
    steps = 0

    mavg1 = np.zeros(w.shape[0]).reshape(-1, 1)
    mavg2 = np.zeros(w.shape[0]).reshape(-1, 1)
    
    # Adam loop
    while epochs_worse < patience:
        steps += 1 

        # Training         
        y_hat, _ = forward_pass(X_train, trained_w)
        gradient = (X_train.T @ (y_hat - y_train)) / len(y_train)
        l1_gradient = l * (trained_w / np.abs(trained_w))
        final_gradient = gradient + l1_gradient

        mavg1 = decay1 * mavg1 + (1 - decay1) * final_gradient
        mavg2 = decay2 * mavg2 + (1 - decay2) * np.square(final_gradient)
        mavg1_cor = mavg1 / (1 - decay1 ** steps)
        mavg2_cor = mavg2 / (1 - decay2 ** steps)
        trained_w -= alpha  * mavg1_cor / (np.sqrt(mavg2_cor) + eps)
        
        # Potential early stopping
        _, z_val = forward_pass(X_val, trained_w)
        validation_loss = mean_logistic_cross_entropy(
            z_val, 
            y_val, 
            trained_w
        )

        if validation_loss > optimum_loss:
            epochs_worse += 1
        else:
            epochs_worse = 0
            optimum_loss = validation_loss
            checkpoint_w = np.copy(trained_w)

    return checkpoint_w, steps

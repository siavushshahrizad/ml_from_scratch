"""
File: utils.py
Date: 30.6.25
--------------------
SUMMARY
--------------------
The file contains helper functions.
"""


import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


# NUM_PREDICTORS_TO_USE = [1, 8, 50, 100, 200, 500]
NUM_PREDICTORS_TO_USE = [1, 8]
MAX_PREDICTORS = 500
TRAIN_TEST_RATIO = 0.8
LEARNING_RATE = 0.0001
MAX_EPOCHS = 5 
CONVERGENCE_THRESHHOLD = 0.1


def add_bias(X):
    """
    Adds a bias term (column) to each row of the input input matrix X.
    """
    bias_terms = np.ones(X.shape[0])
    return  np.column_stack([bias_terms, X])

def load_and_clean_data():
    """
    Loads data which is pre-cleaned.
    The first col is the median income variable.
    More can be found out with data.DESCR.
    Also adds synthetic features for simulations.
    The data is returned as two numpy arrays.
    """
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names).to_numpy()

    # Add synthetic features
    additions = np.random.randn(X.shape[0], MAX_PREDICTORS - X.shape[1])
    X = np.concatenate((X, additions), axis=1)
    
    X = add_bias(X)
    y = data.target
    return X, y

def forward_pass(X, w):
    return X @ w

def mean_squared_error(y, y_hat):
    m = y.shape[0]
    return 0.5 * (np.sum(np.square(y - y_hat)) / m)

def closed_form_solution(X, y):
    """
    This fun calculates the optimal weights in one 
    go by solving a system of linear equations.
    See the accompanying PDF for details    
    """
    w_optimal = np.linalg.inv(X.T @ X) @ X.T @ y            # optimal_w results from solving gradient for 0
    return w_optimal

def measure_time_for_weights(func, X, y):
    """
    Returns time in seconds. Wrapper function
    to see how long each method of calculating
    weights takes.
    """
    start = time.time()
    weights = func(X, y)
    end = time.time()
    time_needed = end - start
    return time_needed, weights

def simulate_closed_form_solution(X_train, X_test, y_train, y_test):
    """
    Returns the time needed to compute weights
    and the loss achieved on the test set.
    """
    time_needed, w = measure_time_for_weights(
        closed_form_solution,
        X_train,
        y_train
    )
    y_hat_test = forward_pass(X_test, w)
    test_loss = mean_squared_error(y_test, y_hat_test)
    return time_needed, test_loss

def gradient_descent(X, y):
    """
    Calculates updates to weights
    via gradient descent. Implementation
    does a certain number of max loops but 
    generally to aims when there is 
    "convergence".
    """
    w = np.random.randn(X.shape[1])     
    n = X.shape[0]
    y_hat_prev = forward_pass(X, w)
    loss_prev = mean_squared_error(y, y_hat_prev)

    for _ in range(MAX_EPOCHS):
        y_hat_curr = forward_pass(X, w)
        gradient =  (X.T @ (y_hat_curr  - y)) / n
        w -= (LEARNING_RATE * gradient)
        loss_curr = mean_squared_error(y, y_hat_curr)

        if abs(loss_curr - loss_prev) <= CONVERGENCE_THRESHHOLD:
            break
        loss_prev = loss_curr

    return w 

def simulate_gradient_descent(X_train, X_test, y_train, y_test):
    time_needed, w = measure_time_for_weights(
        gradient_descent, 
        X_train,
        y_train
    )
    y_hat_test = forward_pass(X_test, w)
    test_loss = mean_squared_error(y_test, y_hat_test)
    return time_needed, test_loss
        
def run_comparison(X, y):
    """
    Func takes numpy arrays of features and the targets.
    It then simulates two optimisation methods, returning
    two arrays with the results for each method.
    """
    # Create simple train test split
    n = X.shape[0]
    cutoff = int(n * TRAIN_TEST_RATIO)
    results_closed_form = []
    results_gradient_descent = []
    
    for num in NUM_PREDICTORS_TO_USE:
        X_train =  X[:cutoff, 0:num]
        X_test = X[cutoff:, 0:num]
        y_train = y[:cutoff]
        y_test = y[cutoff:]
        
        results_closed_form.append(simulate_closed_form_solution(
            X_train,
            X_test,
            y_train,
            y_test
        ))

        results_gradient_descent.append(simulate_gradient_descent(
            X_train,
            X_test,
            y_train,
            y_test
        ))

    return results_closed_form, results_gradient_descent

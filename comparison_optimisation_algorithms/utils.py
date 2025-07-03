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


MAX_PREDICTORS = 500

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



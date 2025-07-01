"""
File: utils.py
Date: 30.6.25
--------------------
SUMMARY
--------------------
The file contains helper functions.
"""


import numpy as np


def add_bias(X):
    """
    Adds a bias term (column) to each row of the input input matrix X.
    """
    bias_terms = np.ones(X.shape[0])
    return  np.column_stack([bias_terms, X])

def forward_pass(X, w):
    return X @ w

def mean_squared_error(y, y_hat):
    m = y.shape[0]
    return 0.5 * (np.sum(np.square(y - y_hat)) / m)

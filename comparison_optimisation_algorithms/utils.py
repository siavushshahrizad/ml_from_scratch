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
    print(bias_terms)
    combined = np.column_stack([bias_terms, X])
    return combined


def forward_pass(X, w):
    return X @ w

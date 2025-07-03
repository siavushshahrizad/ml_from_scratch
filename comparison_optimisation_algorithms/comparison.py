"""
File: comparison.py
Date: 30.6.25
--------------------
SUMMARY
--------------------
The file does basic data exploration.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    load_and_clean_data,
    forward_pass,
    mean_squared_error
)


# NUM_PREDICTORS_TO_USE = [1, 8, 50, 100, 200, 500]
NUM_PREDICTORS_TO_USE = [1, 8]

def run_comparison(X, y):
    """
    Func takes numpy arrays of featurs and the targets.
    It then simulates different optimisation methods.
    """
    results = []
    for num in NUM_PREDICTORS_TO_USE:
        curr_df =  X[:, 0:num]

        w = np.random.randn(curr_df.shape[1])     # Accounts for added bias term
        y_hat = forward_pass(y, w)
        cost = mean_squared_error(y, y_hat)


        print(w)

    return results


def main():
    X, y = load_and_clean_data()
    results = run_comparison(X, y)


if __name__ == "__main__":
    main()

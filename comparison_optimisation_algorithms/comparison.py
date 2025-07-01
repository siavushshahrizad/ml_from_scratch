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
from sklearn.datasets import fetch_california_housing
from utils import (
    add_bias,
    forward_pass,
    mean_squared_error
)


OUTCOME = "price"
# NUM_PREDICTORS_TO_USE = [1, 8, 50, 100, 200, 500]
NUM_PREDICTORS_TO_USE = [1, 8]


def load_and_clean_data():
    """
    Loads data which is pre-cleaned.
    The first col is the median income variable.
    More can be found out with data.DESCR.
    The data is returned as two numpy arrays.
    """
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names).to_numpy()
    y = data.target
    return X, y

def run_comparison(X, y):
    """
    Func takes numpy arrays of featurs and the targets.
    It then simulates different optimisation methods.
    """
    results = []
    for num in NUM_PREDICTORS_TO_USE:
        curr_df =  X[:, 0:num]
        curr_df = add_bias(curr_df)

        w = np.random.random(curr_df.shape[1])     # Accounts for added bias term
        y_hat = forward_pass(y, w)
        cost = mean_squared_error(y, y_hat)


        print(w)

    return results


def main():
    X, y = load_and_clean_data()
    results = run_comparison(X, y)


if __name__ == "__main__":
    main()

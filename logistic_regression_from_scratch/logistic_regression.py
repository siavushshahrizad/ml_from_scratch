"""
File: logistic_regression.py
Date: 7 July 25
--------------------
SUMMARY
--------------------
This is the main file that runs
the logistic regression on the breast
cancer data.
"""


import numpy as np
from utils import (
    FILE,
    load_and_clean_data,
    normalise_data,
    create_data_split,
    forward_pass,
    mean_logistic_cross_entropy,
    simple_gradient_descent,
    early_stopping_gradient_descent
)


TRAIN_TO_OTHER_SET_RATIO = 0.8
VAL_TO_TEST_SET_RATIO = 0.5


def main():
    np.random.seed(42)

    # Get data, process, and split it
    X, y = load_and_clean_data(FILE)
    X = normalise_data(X)
    X_train, X_other, y_train, y_other = create_data_split(X, y, TRAIN_TO_OTHER_SET_RATIO)
    X_val, X_test, y_val, y_test = create_data_split(X_other, y_other, VAL_TO_TEST_SET_RATIO)

    # Initialise weights and train on data
    w = np.random.randn(X.shape[1], 1)      # Important to initialise with right shape

    _, z = forward_pass(X_train, w)
    initial_loss = mean_logistic_cross_entropy(z, y_train, w)
    print("Initial loss: ", initial_loss)
    w_trained = simple_gradient_descent(w, X_train, y_train)

    _, z = forward_pass(X_train, w_trained)
    trained_loss = mean_logistic_cross_entropy(z, y_train, w_trained)
    print("Trained loss: ", trained_loss)
    
    w_trained, epochs_trained = early_stopping_gradient_descent(
        w, 
        X_train, 
        y_train,
        X_val,
        y_val
        )
    _, z = forward_pass(X_train, w_trained)
    trained_loss = mean_logistic_cross_entropy(z, y_train, w_trained)
    print("Trained loss early stopping: ", trained_loss)
    print("Epochs trained: ", epochs_trained)









if __name__ == "__main__":
    main()

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
    early_stopping_gradient_descent,
    adam,
    early_stopping_adam
)


TRAIN_TO_OTHER_SET_RATIO = 0.8
VAL_TO_TEST_SET_RATIO = 0.5
NUM_TRIALS = 4


np.random.seed(42)      # Set for reproducability


def main():
    """
    The main func runs four logistic regression 
    variants on the breast cancer set, comparing
    loss on the test set, and in the case of 
    early stopping variants, after how many epochs 
    convergence are reached.

    Results are saved in bar charts.
    """
    ### Arrays to save trial runs ###
    #                               #
    #################################
    loss_random = np.zeros(NUM_TRIALS)
    loss_gd = np.zeros(NUM_TRIALS)
    loss_adam = np.zeros(NUM_TRIALS)
    loss_early_gd = np.zeros(NUM_TRIALS)
    loss_early_adam = np.zeros(NUM_TRIALS)

    epochs_early_gd = np.zeros(NUM_TRIALS)
    epochs_early_adam = np.zeros(NUM_TRIALS)


    ### Get data, process, and split it ###
    #                                     #
    #######################################
    X, y = load_and_clean_data(FILE)
    X = normalise_data(X)
    X_train, X_other, y_train, y_other = create_data_split(X, y, TRAIN_TO_OTHER_SET_RATIO)
    X_val, X_test, y_val, y_test = create_data_split(X_other, y_other, VAL_TO_TEST_SET_RATIO)

    
    ###      Run trials      ###
    #                          #
    ############################
    for i in range(NUM_TRIALS):
        
        # Initialise weights and train on data
        w = np.random.randn(X.shape[1], 1)      # Important to initialise with right shape
        
        # Random weights
        _, z = forward_pass(X_test, w)
        loss_random[i] = mean_logistic_cross_entropy(z, y_test, w)
        
        # GD
        w_trained = simple_gradient_descent(w, X_train, y_train)
        _, z = forward_pass(X_test, w_trained)
        loss_gd[i] = mean_logistic_cross_entropy(z, y_test, w_trained)
        
        # Early stopping GD
        w_trained, epochs_trained = early_stopping_gradient_descent(
            w, 
            X_train, 
            y_train,
            X_val,
            y_val
            )
        epochs_early_gd[i] = epochs_trained
        _, z = forward_pass(X_test, w_trained)
        loss_early_gd[i] = mean_logistic_cross_entropy(z, y_test, w_trained)
        
        # Adam
        w_trained = adam(
            w, 
            X_train, 
            y_train,
            )
        _, z = forward_pass(X_test, w_trained)
        loss_adam[i] = mean_logistic_cross_entropy(z, y_test, w_trained)

        # Adam with early stopping
        w_trained, epochs_trained = early_stopping_adam(
            w, 
            X_train, 
            y_train,
            X_val,
            y_val
            )
        epochs_early_adam[i] = epochs_trained
        _, z = forward_pass(X_test, w_trained)
        loss_early_adam[i] = mean_logistic_cross_entropy(z, y_test, w_trained)

    ###     Print results to disk     ###
    #                                   #
    #####################################
    print(loss_random.mean())
    print(loss_gd.mean())
    print(loss_adam.mean())
    print(loss_early_gd.mean())    
    print(loss_early_adam.mean())    
    print(epochs_early_gd.mean())     
    print(epochs_early_adam.mean()) 
   
if __name__ == "__main__":
    main()

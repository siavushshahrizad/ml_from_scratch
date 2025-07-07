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
import tracemalloc
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from tqdm import tqdm


TRAIN_TEST_RATIO = 0.8
MAX_EPOCHS = 100
BASE_LEARNING_RATE = 0.01
MAX_PREDICTORS = 25000


def add_bias(X):
    """
    Adds a bias term (column) to each row of the input input matrix X.
    """
    bias_terms = np.ones(X.shape[0])
    return  np.column_stack([bias_terms, X])

def normalise_data(X):
    col_means = np.mean(X, axis=0)
    col_stds = np.std(X, axis=0)

    normalised = (X - col_means) / col_stds
    return normalised

def load_and_clean_data():
    """
    Loads data which is pre-cleaned.
    The first col is the median income variable.
    More can be found out with data.DESCR.
    The data is returned as two numpy arrays.
    """
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names).to_numpy()
    X = normalise_data(X)

    # Add synthetic features
    additions = np.random.randn(X.shape[0], MAX_PREDICTORS - X.shape[1])
    X = np.concatenate((X, additions), axis=1)

    X = add_bias(X)
    y = data.target
    return X, y

def forward_pass(X, w):
    return X @ w

def mean_squared_error(y, y_hat):
    """
    Overflow easily occurs so my quick fix
    is to check for large numbers and scale.
    """
    m = y.shape[0]
    residual = y - y_hat
    scale = np.max(np.abs(residual))
    
    if scale > 1e4:  
        scaled_residual = residual / scale
        squared_residual = np.square(scaled_residual)
        summed_error = np.sum(squared_residual)
        return 0.5 * summed_error / m * (scale ** 2)
    else:
        residual = y - y_hat
        squared_residual = np.square(residual)
        summed_error = np.sum(squared_residual)
        return 0.5 * summed_error / m

def closed_form_solution(X, y):
    """
    This fun calculates the optimal weights in one 
    go by solving a system of linear equations.
    See the accompanying PDF for details    
    """
    w_optimal = np.linalg.inv(X.T @ X) @ X.T @ y            # optimal_w results from solving gradient for 0
    return w_optimal

def measure_time_and_memory(func, X, y):
    """
    Returns time in seconds. Wrapper function
    to see how long each method of calculating
    weights takes. Also returns peak memory usage
    in GB.
    """
    start = time.time()
    tracemalloc.start()

    weights = func(X, y)
    _, peak_mem_usage = tracemalloc.get_traced_memory()     # Returns in bytes

    tracemalloc.stop()
    end = time.time()
    time_needed = end - start               # Returns in seconds
    peak_mem_usage = peak_mem_usage / (1024 ** 3)   # Conversion to GB

    return time_needed, peak_mem_usage, weights

def simulate_closed_form_solution(X_train, X_test, y_train, y_test):
    """
    Returns the time needed to compute weights
    and the loss achieved on the test set.
    """
    time_needed, peak_mem_usage, w = measure_time_and_memory(
        closed_form_solution,
        X_train,
        y_train
    )
    y_hat_test = forward_pass(X_test, w)
    test_loss = mean_squared_error(y_test, y_hat_test)
    return time_needed, peak_mem_usage, test_loss

def gradient_descent(X, y):
    """
    Calculates updates to weights
    via gradient descent. Implementation
    does a certain number of max to keep it simple.

    NOTE: If the below implementation
    was slightly different so that a 
    matrix was multiplied with a matrix,
    then gradient descent would be a much
    worse algorithm.
    """
    w = np.zeros(X.shape[1])     
    n = X.shape[0]

    for i in range(MAX_EPOCHS):     # Left for debugging
        y_hat_curr = forward_pass(X, w)         # Prevents matrix multiplication        
        gradient =  (X.T @ (y_hat_curr  - y)) / n

        w -= (BASE_LEARNING_RATE * gradient)
    
    return w 

def simulate_gradient_descent(X_train, X_test, y_train, y_test):
    time_needed, peak_mem_usage, w = measure_time_and_memory(
        gradient_descent, 
        X_train,
        y_train
    )
    y_hat_test = forward_pass(X_test, w)
    test_loss = mean_squared_error(y_test, y_hat_test)
    return time_needed, peak_mem_usage, test_loss
        
def run_comparison(X, y, predictors_to_use=[1]):
    """
    Func takes numpy arrays of features and the targets.
    It then simulates two optimisation methods, returning
    two arrays with the results for each method.
    The comparison data includes the validation loss and 
    the time needed to compute optimal/improved weights
    for number of features specified to be tested.
    """
    # Create simple train test split
    n = X.shape[0]
    cutoff = int(n * TRAIN_TEST_RATIO)
    results_closed_form = []
    results_gradient_descent = []
    
    for num in tqdm(predictors_to_use, desc="Simulating feature sizes"):
        X_train =  X[:cutoff, 0:num+1]      # +1 accounts for added bias
        X_test = X[cutoff:, 0:num+1]
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

def save_plotted_data(
        data, 
        index,
        y_label,
        title,
        file_name,
        predictors_used=[1]
    ):
    """
    Saves bar charts to disk. The data coming in
    is an array of (time_needed, memory_peak_use,
    loss). The index argument therefore specifies
    which three of these variables should be printed.
    The function is somewhat inefficient, but better
    is the enemy of good.
    """
    def add_labels(x_data, y_data):
        """
        Adds labels; assumes only natural numbers.
        """
        offset = max(y_data) / 100

        for i in range(len(x_data)):
            rounded_num = round(y_data[i], 2)
            plt.text(i, y_data[i] + offset, rounded_num, ha="center")

    y_data = [y[index] for y in data]
    x_data = range(len(predictors_used))

    plt.bar(x_data, y_data)
    plt.xticks(x_data, predictors_used)
    plt.title(title)
    plt.xlabel("Number of features")
    plt.ylabel(y_label)

    add_labels(x_data, y_data)

    plt.savefig(file_name)
    plt.close()

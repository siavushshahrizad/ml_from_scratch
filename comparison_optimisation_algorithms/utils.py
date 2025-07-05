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


NUM_PREDICTORS_TO_USE = [1, 8]
MAX_PREDICTORS = 10000
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
    _, peak_mem_usage = tracemalloc.get_traced_memory()

    tracemalloc.stop()
    end = time.time()
    time_needed = end - start
    peak_mem_usage = peak_mem_usage / (1024 ** 3)        # Convert from bytes to GB

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
    
    for num in tqdm(predictors_to_use, desc="Simulatng feature sizes"):
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

def save_plotted_data(
        results_cf, 
        results_gd, 
        index,
        y_label,
        title,
        file_name,
        predictors_used=[1]
        ):
    """
    Saves plots to disk; not very user friendly func.
    Basically creates grouped bar charts with labels
    on top of the bars.
    """
    y_cf = [y_cf[index] for y_cf in results_cf]
    y_gd = [y_gd[index] for y_gd in results_gd]

    w, x = 0.4, np.arange(len(predictors_used))

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - w/2, y_cf, width=w, label='Closed-form')
    rects2 = ax.bar(x + w/2, y_gd, width=w, label='Gradient descent')

    ax.set_xticks(x, predictors_used)
    ax.set_xlabel("Number of features", fontweight="bold")
    ax.ticklabel_format(style='plain', axis='y')
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend()

    def autolabel(rects):
        """
        Creates the labels on top of the bars.
        """
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),                          
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(file_name)

"""
File: utils.py
Date: 7 July 25
--------------------
SUMMARY
--------------------
The file contains the functions the 
main programme uses.
"""


import numpy as np
import pandas as pd


FILE = "./data/breast-cancer-wisconsin.data"
MISSING_VALUE = "?"


def load_and_clean_data(file):
    """
    Loads and cleans data, i.e. documentation says 
    there are 16 missing values somewhere.
    See documentation what exactly these vars are, 
    """
    data = pd.read_csv(file, header=None)           # Contains no headers
    data = data.replace(MISSING_VALUE, np.nan)
    data = data.dropna()
    data = data.to_numpy()
    data = data.astype(int)     # Doc tells us that all numbers are 1-10, but some get converted into str
    X = data[:, 1:-1]        # Last col is the target; first col is ID
    y = data[:, -1:]
    return X, y

def create_data_split(X, y, ratio):
    """
    Returns a split of two datasets when one is inputted.     
    The split is random and the point of the func is to split
    datasets into train, test, validation.
    """
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    cutoff = int(n * ratio)
    indices_set1 = indices[:cutoff]
    indices_set2 = indices[cutoff:]

    X_1 = X[indices_set1]
    X_2 = X[indices_set2]
    y_1 = y[indices_set1]
    y_2 = y[indices_set2]

    return X_1, X_2, y_1, y_2

def forward_pass(X, w):
    """
    This version returns logits and predictions. 
    Logits are returned for cross-entropy func.    
    """
    z = X @ w
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat, z

def logistic_cross_entropy(logits, y, w, l=0.25):
    """
    L1 regularisation is applied to this loss func.
    The way this is implemented, e.g. logits are consumed
    rather than y_hat and then logaddexp, is supposed to 
    be more stable according to Grosse's notes.
    """
    print("logits: ", logits)
    part1 = np.logaddexp(0, -logits)
    part2 = y.T @ np.logaddexp(0, -logits)
    part3 = np.logaddexp(0, logits)
    part4 = (1 - y).T @ np.logaddexp(0, logits)
    loss = y.T @ np.logaddexp(0, -logits) + (1 - y).T @ np.logaddexp(0, logits)
    print("part1: ", part1)
    print("part2: ", part2)
    print("part3: ", part3)
    print("part4: ", part4)

    loss = y.T @ np.logaddexp(0, -logits) + (1 - y).T @ np.logaddexp(0, logits)
    print("L: ", loss)
    loss_with_l1 = loss +  l * np.sum(np.abs(w))
    print("w: ", w)
    print("L: ", loss_with_l1)
    return loss_with_l1


    



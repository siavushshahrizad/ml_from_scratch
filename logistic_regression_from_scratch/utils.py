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


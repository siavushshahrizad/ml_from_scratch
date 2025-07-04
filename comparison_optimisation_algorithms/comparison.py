"""
File: comparison.py
Date: 30.6.25
--------------------
SUMMARY
--------------------
The file compares the two optimisation methods.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    load_and_clean_data,
    run_comparison
)


NUM_PREDICTORS_TO_USE = [1, 8, 50, 100, 200, 500]

def main():
    X, y = load_and_clean_data()
    results = run_comparison(X, y, NUM_PREDICTORS_TO_USE)


if __name__ == "__main__":
    main()

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


def main():
    X, y = load_and_clean_data()
    results = run_comparison(X, y)


if __name__ == "__main__":
    main()

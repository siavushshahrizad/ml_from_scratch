"""
File: func_test.py
Date: 7 July 25
--------------------
SUMMARY
--------------------
This file contains unit tests
for functions.
"""


import pytest
import numpy as np
from utils import (
    FILE,
    load_and_clean_data
)


NUM_SAMPLES = 683       # Doc says originally 699 samples - 16 that have a missing value
NUM_FEATURES = 9


class TestClass:
    
    def test_load_and_clean_data(self):
        print("Running")
        X, y = load_and_clean_data(FILE)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == NUM_SAMPLES
        assert X.shape[1] == NUM_FEATURES
        assert len(y) == NUM_SAMPLES 
        assert set(np.unique(X)) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}        # Should only be 1-10
        assert set(np.unique(y)) == {2, 4}                                 # Should only be

        

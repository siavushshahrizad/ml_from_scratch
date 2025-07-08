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
    load_and_clean_data,
    create_data_split,
    forward_pass
)


NUM_SAMPLES = 683       # Doc says originally 699 samples - 16 that have a missing value
NUM_FEATURES = 9


class TestClass:
    @pytest.fixture
    def create_static_data(self):
        X = np.array([
            [1, 2,], 
            [3, 4], 
            [4, 5],
            [6, 7],
            [1, 1],
            [1, 4],
            [2, 3],
            [4, 4],
            [3, 2],
            [1, 4]
        ])
        
        w = np.array([1, 2])
        y = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 0])


        X_tiny = np.array([[1, 2], [3, 4]])
        y_tiny = np.array([0, 1])

        return X, y, X_tiny, y_tiny, w

    def test_load_and_clean_data(self):
        X, y = load_and_clean_data(FILE)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == NUM_SAMPLES
        assert X.shape[1] == NUM_FEATURES
        assert len(y) == NUM_SAMPLES 
        assert set(np.unique(X)) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}        # Should only be 1-10
        assert set(np.unique(y)) == {2, 4}                                 # Should only be

    def test_create_data_split(self, create_static_data):
        X, y, _, _, _ = create_static_data

        # Simulate first split into two halves        
        ratio = 0.5
        X_1, X_2, y_1, y_2 = create_data_split(X, y, ratio)
        expected_rows = X.shape[0] * ratio
        expected_cols = X.shape[1]

        assert isinstance(X_1, np.ndarray)
        assert isinstance(X_2, np.ndarray)
        assert isinstance(y_1, np.ndarray)
        assert isinstance(y_2, np.ndarray)
        assert X_1.shape == (expected_rows, expected_cols)
        assert X_2.shape == (expected_rows, expected_cols)
        assert len(y_1) == expected_rows
        assert len(y_2) == expected_rows

        # Simulate a second split that would create val, test set
        ratio = 0.6     
        X_2, X_3, y_2, y_3 = create_data_split(X_2, y_2, ratio)     # Note left X and y are bigger ones
        expected_rows_X_2 = expected_rows * ratio
        expected_rows_X_3 = expected_rows * (1 - ratio)
        assert X_2.shape ==  (expected_rows_X_2, expected_cols)
        assert X_3.shape ==  (expected_rows_X_3, expected_cols)
        assert len(y_2) == expected_rows_X_2
        assert len(y_3) == expected_rows_X_3
        
    def test_forward_pass(self, create_static_data):
        _, _, X, _, w = create_static_data
        y_hat = forward_pass(X, w)
        expected = np.array([0.993, 1])

        assert np.array_equal(np.round(y_hat, 3), expected)



        

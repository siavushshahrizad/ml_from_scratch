"""
File: test_funcs.py
Date: 1.7.25
--------------------
SUMMARY
--------------------
Runs tests to verify correctness.
"""


import pytest
import numpy as np
from utils import (
    add_bias,
    forward_pass
)


class TestClass:
    def test_forward_pass(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        w = np.array([2, 3, 10])
        expected = np.array([38, 83])
        result = forward_pass(X, w)
        assert np.array_equal(expected, result)

    def test_add_bias(self):
        """
        Tests if bias terms are correctly 
        added to the beginning of reach row.
        """
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X_prime = add_bias(X)
        expected = np.array([[1, 1,  2, 3], [1, 4, 5, 6]])
        print(X_prime)
        assert X_prime.shape[1] == X.shape[1] + 1
        assert np.array_equal(X_prime, expected)



        


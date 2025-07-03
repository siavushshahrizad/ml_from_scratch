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
    MAX_PREDICTORS,
    add_bias,
    load_and_clean_data,
    forward_pass,
    mean_squared_error,
    closed_form_solution,
    measure_time_for_weights
)


np.random.seed(42)


class TestClass:
    def test_add_bias(self):
        """
        Tests if bias terms are correctly 
        added to the beginning of reach row.
        """
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X_prime = add_bias(X)
        expected = np.array([[1, 1,  2, 3], [1, 4, 5, 6]])
        assert X_prime.shape[1] == X.shape[1] + 1
        assert np.array_equal(X_prime, expected)

    def test_load_and_clean_data(self):
        X, y = load_and_clean_data()
        assert y.shape[0] >= 1000
        assert len(y.shape) == 1        # Numpy does not yet return true col vector
        assert y.shape[0] == X.shape[0]
        assert X.shape[1] == MAX_PREDICTORS + 1

    def test_forward_pass(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        w = np.array([2, 3, 10])
        expected = np.array([38, 83])
        result = forward_pass(X, w)
        assert np.array_equal(expected, result)

    def test_mean_squared_error(self):
        y = np.array([2, 3, 2, 5])
        y_hat = np.array([1, 10, 2, 5])
        cost = mean_squared_error(y, y_hat)
        expected = 6.25
        assert cost == expected

    def test_closed_form_solution(self):
        X = np.random.randn(400,5)
        y = np.random.randn(400)
        random_w = np.random.randn(5)
        optimised_w = closed_form_solution(X, y)

        random_predictions = forward_pass(X, random_w)
        optimised_predictions = forward_pass(X, optimised_w)

        random_cost = mean_squared_error(y, random_predictions)
        optimised_cost = mean_squared_error(y, optimised_predictions)

        assert optimised_cost < random_cost 

    def test_measure_time_for_weights(self):
        toy_X = np.random.rand(10, 3)
        toy_y  = np.random.rand(10)

        def create_toy_weights(X, y):
            toy_weights = y * 2
            return toy_weights 

        time_needed, toy_weights  = measure_time_for_weights(create_toy_weights, toy_X, toy_y)

        assert isinstance(time_needed, float)
        assert time_needed > 0
        assert isinstance(toy_weights, np.ndarray)
        assert toy_weights.shape[0] == toy_y.shape[0]





        


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
    measure_time_for_weights,
    simulate_closed_form_solution,
    gradient_descent,
    simulate_gradient_descent
)


np.random.seed(42)


class TestClass:
    @pytest.fixture
    def generate_random_data(self):
        n = 100
        m = 10
        X = np.random.randn(n, m)
        y = np.random.randn(n)
        w = np.random.randn(m)
        return X, y, w
    
    @pytest.fixture
    def generate_static_data(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        w = np.array([2, 3, 10])
        return X, w
        
    def test_add_bias(self, generate_static_data):
        """
        Tests if bias terms are correctly 
        added to the beginning of reach row.
        """
        X, _ = generate_static_data
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

    def test_forward_pass(self, generate_static_data):
        X, w = generate_static_data
        expected = np.array([38, 83])
        result = forward_pass(X, w)
        assert np.array_equal(expected, result)

    def test_mean_squared_error(self):
        y = np.array([2, 3, 2, 5])
        y_hat = np.array([1, 10, 2, 5])
        cost = mean_squared_error(y, y_hat)
        expected = 6.25
        assert cost == expected

    def test_closed_form_solution(self, generate_random_data):
        X, y, random_w = generate_random_data
        optimised_w = closed_form_solution(X, y)

        random_predictions = forward_pass(X, random_w)
        optimised_predictions = forward_pass(X, optimised_w)

        random_cost = mean_squared_error(y, random_predictions)
        optimised_cost = mean_squared_error(y, optimised_predictions)

        assert optimised_cost < random_cost 

    def test_measure_time_for_weights(self, generate_random_data):
        X, y, _ = generate_random_data

        def create_toy_weights(X, y):
            toy_weights = y * 2
            return toy_weights 

        time_needed, toy_weights  = measure_time_for_weights(create_toy_weights, X, y)

        assert isinstance(time_needed, float)
        assert time_needed > 0
        assert isinstance(toy_weights, np.ndarray)
        assert toy_weights.shape[0] == y.shape[0]

    def test_simulate_closed_form_solution(self, generate_random_data):
        X_train, y_train, _ = generate_random_data
        X_test, y_test, _ = generate_random_data

        time_needed, loss = simulate_closed_form_solution(
            X_train,
            X_test,
            y_train,
            y_test
        )

        assert isinstance(time_needed, float)
        assert time_needed > 0
        assert isinstance(loss, float)
        assert loss > 0

    def test_gradient_descent(self, generate_random_data):
        X, y, w_random = generate_random_data
        w_trained = gradient_descent(X, y)

        assert isinstance(w_trained, np.ndarray)
        assert w_trained.shape[0] == X.shape[1]

        random_preds = forward_pass(X, w_random)
        trained_preds = forward_pass(X, w_trained)
        random_loss = mean_squared_error(y, random_preds)
        trained_loss = mean_squared_error(y, trained_preds)

        assert trained_loss < random_loss

    def test_simulate_gradient_descent(self, generate_random_data):
        X_train, y_train, _ = generate_random_data
        X_test, y_test, _ = generate_random_data

        time_needed, loss = simulate_gradient_descent(
            X_train,
            X_test,
            y_train,
            y_test
        )

        assert isinstance(time_needed, float)
        assert time_needed > 0
        assert isinstance(loss, float)
        assert loss > 0


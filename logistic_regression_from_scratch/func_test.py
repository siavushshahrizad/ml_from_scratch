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
    normalise_data,
    FILE,
    load_and_clean_data,
    create_data_split,
    forward_pass,
    mean_logistic_cross_entropy,
    simple_gradient_descent,
    early_stopping_gradient_descent,
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
        
        w = np.array([1, 2], dtype=float)
        y = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 0])


        X_tiny = np.array([[1, 2], [3, 4]])
        y_tiny = np.array([0, 1])

        return X, y, X_tiny, y_tiny, w

    @pytest.fixture 
    def create_realistic_data(self):
        pass

    def test_normalise_data(self, create_static_data):
        _, _, X, _, _ = create_static_data
        result = normalise_data(X)
        expected = np.array([[-1, -1], [1, 1]])

        assert np.array_equal(result, expected)


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
        y_hat, logits = forward_pass(X, w)
        expected_logits = np.array([5, 11])
        expected_y_hat = np.array([0.993, 1])

        assert np.array_equal(logits, expected_logits)
        assert np.array_equal(np.round(y_hat, 3), expected_y_hat)


    def test_mean_logistic_cross_entropy(self, create_static_data):
        _, _, X, y, w = create_static_data
        _, logits = forward_pass(X, w)
        loss = mean_logistic_cross_entropy(logits, y, w)
        expected = 2.518366

        assert isinstance(loss, float) 
        assert round(loss, 6) == expected
    
    def test_simple_gradient_descent(self, create_static_data):
        """
        Func tests whether loss goes down over several
        batches of epochs, and whether weights are moving.
        """
        X, y, _, _, w = create_static_data
        _, logits = forward_pass(X, w)
        initial_loss = mean_logistic_cross_entropy(logits, y, w)

        # Comparison after one wave of gradient descent
        trained_w_1 = simple_gradient_descent(
            w,
            X,
            y,
            num_epochs=5
        )
        _, logits = forward_pass(X, trained_w_1)
        trained_loss_1 = mean_logistic_cross_entropy(logits, y, trained_w_1)
        assert trained_loss_1 < initial_loss
        assert not np.allclose(w, trained_w_1)

        # # Comparison after another  wave of gradient descent
        trained_w_2 = simple_gradient_descent(
            trained_w_1,
            X,
            y,
            num_epochs=5
        )
        _, logits = forward_pass(X, trained_w_2)
        trained_loss_2 = mean_logistic_cross_entropy(logits, y, trained_w_2)
        assert trained_loss_2 < trained_loss_1
        assert not np.allclose(trained_w_2, trained_w_1)

    def test_early_stopping_gd(self, create_static_data):
        """
        A bit repetitive compared to previous func but 
        faff to unify. Func tests that validation loss 
        is decreasing and that this optimiser runs
        for a certain min num of epochs and that it
        doesn't run too much. It also smell-tests
        if it runs for fewer epochs if stopping
        criterion is harsher.
        """
        X_train, y_train, X_val, y_val, w = create_static_data
        _, logits = forward_pass(X_val, w)
        initial_loss = mean_logistic_cross_entropy(logits, y_val, w)

        trained_w_1, relaxed_epochs = early_stopping_gradient_descent(
            w,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        _, logits = forward_pass(X_val, trained_w_1)
        trained_loss_1 = mean_logistic_cross_entropy(logits, y_val, trained_w_1)
        assert trained_loss_1 < initial_loss
        assert not np.allclose(w, trained_w_1)
        assert relaxed_epochs > 5       # Min num of epochs
        assert relaxed_epochs < 1000    # Would be red flag if run that long for simple data
        
        # Harsher stropping criterion test
        _, stricter_epochs = early_stopping_gradient_descent(
            w, 
            X_train,
            y_train,
            X_val, 
            y_val,
            threshold=0.5
        )
        assert stricter_epochs < relaxed_epochs


        




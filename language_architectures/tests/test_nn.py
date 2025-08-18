"""
File: test_nn.py
Date: 18.8.25
Summary:  Tests nn class
"""


import pytest
import numpy as np
from architectures.nn import NeuralNetwork
from utils.utils_data_processing import split_words


class TestNN:
    @pytest.fixture
    def simple_nn(self, simple_vocab):
        network = NeuralNetwork()
        network.initialise(simple_vocab)

        return network
    
    @pytest.mark.fast
    def test_build_vocab(self, simple_vocab):
        network = NeuralNetwork()
        assert network.size == 0
        assert network.word_to_int == None
        assert network.int_to_word == None

        network._build_vocabulary(simple_vocab)
        assert network.size == 7
        assert "This" in network.word_to_int
        assert "test" in network.word_to_int

        target = "test"
        pos = network.word_to_int[target]
        assert network.int_to_word[pos] == target

    def test_initialise_H_matrix(self):
        network = NeuralNetwork()
        assert network.H == None
        network._initialise_H_matrix()
        assert isinstance(network.H, np.ndarray)
        context = network.context
        em_size = network.embedding_size
        assert network.H.shape == (network.hidden_size, (context * em_size) + 1)
        assert network.H.dtype == np.float64

    def test_initialise_C_matrix(self, simple_vocab):
        network = NeuralNetwork()

        assert network.C == None
        network._build_vocabulary(simple_vocab)
        network._initialise_C_matrix()
        assert isinstance(network.C, np.ndarray)
        assert network.C.shape == (network.size, network.embedding_size)
        assert network.C.dtype == np.float64

    def test_initilaise_U_matrix(self):
        network = NeuralNetwork()

        assert network.U == None
        network._initialise_U_matrix()
        assert isinstance(network.U, np.ndarray)
        vocab_size = network.size
        hidden_size= network.hidden_size
        assert network.U.shape == (vocab_size, hidden_size + 1)
        assert network.U.dtype == np.float64

    @pytest.mark.fast
    def test_embedding_layer(self, simple_nn, simple_input):
        separated = split_words(simple_input)

        result = simple_nn._forward_embedding_layer(separated)
        context = simple_nn.context
        em_size = simple_nn.embedding_size

        assert len(result) == 2
        assert len(result[0]) == (context * em_size) + 1

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        
        expected_sample_0 = np.zeros((context, em_size))
        for idx, word in enumerate(separated[0]):
            word_idx = simple_nn.word_to_int[word]
            expected_sample_0[idx] = simple_nn.C[word_idx]
        expected_sample_0 = expected_sample_0.flatten()
        expected_sample_0 = np.concatenate([np.ones(1), expected_sample_0], axis=0)
        assert np.array_equal(result[0], expected_sample_0)

    @pytest.mark.fast
    def test_forward_computation_hidden(self, simple_vocab):
        """
        Tests some actual values but that's pointless.
        Left the atavism. Later tests only sense check
        shapes and the nature of values.
        """
        np.random.seed(42)

        hidden_size = embedding_size = context = 2
        network = NeuralNetwork(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            context = context
        )
        network.initialise(simple_vocab)
        X = np.array([[1, 2, 2, 3, 2], [1, 4, 2, 5, 2]])
        batch_size = X.shape[0]

        result = network._forward_hidden_layer(X)
        assert result.shape == (batch_size, hidden_size)

        result_preac= np.arctanh(result)
        expected_preac = X @ network.H.T
        assert np.allclose(result_preac, expected_preac)

        expected = np.tanh(expected_preac)
        assert np.array_equal(result, expected)

    @pytest.mark.fast
    def test_softmax(self):
        test_input = np.array([[1, 2], [2, 3]])
        network = NeuralNetwork()
        expected = np.array([[0.27, 0.73], [0.27, 0.73]])
        result = network._softmax(test_input)
        assert np.array_equal(expected, np.round(result, decimals=2))

    
    @pytest.mark.fast
    def test_forward_computation_output(self, simple_vocab):
        np.random.seed(42)

        hidden_size = embedding_size = context = 2
        network = NeuralNetwork(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            context = context
        )
        network.initialise(simple_vocab)

        activations = np.array([[1, 2], [4, 5]])
        batch = len(activations)

        result = network._forward_output_units(activations)
        assert result.shape == (batch, network.size)
        assert np.all(result >= 0) and np.all(result <= 1)
        assert np.allclose(np.sum(result, axis=1), 1)


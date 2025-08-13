"""
File: test_nn.py
Date: 18.8.25
Summary:  Tests nn class
"""


import pytest
import numpy as np
from architectures.nn import NeuralNetwork
from architectures.utils import split_words


class TestNN:
    @pytest.fixture
    def simple_nn(self, simple_vocab, simple_input):
        vocab = simple_vocab
        network = NeuralNetwork()
        network.build_vocabulary(vocab)

        words = simple_input
        separated = split_words(words)
        return network, separated
    
    @pytest.mark.fast
    def test_build_vocab(self, simple_vocab):
        vocab = simple_vocab
        network = NeuralNetwork()
        assert network.size == 0
        assert network.word_to_int == None
        assert network.int_to_word == None
        assert network.C == None

        network.build_vocabulary(vocab)
        assert network.size == 7
        assert "This" in network.word_to_int
        assert "test" in network.word_to_int

        target = "test"
        pos = network.word_to_int[target]
        assert network.int_to_word[pos] == target
        assert isinstance(network.C, np.ndarray)

    @pytest.mark.fast
    def test_concat(self, simple_nn):
        network, separated = simple_nn
        result = network._concat_embeddings(separated)
        context = network.context
        em_size = network.embedding_size

        assert len(result) == 2
        assert len(result[0]) == context * em_size

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

        expected_sample_0 = np.zeros((context, em_size))
        for idx, word in enumerate(separated[0]):
            word_idx = network.word_to_int[word]
            expected_sample_0[idx] = network.C[word_idx]
        expected_sample_0 = expected_sample_0.flatten()
        assert np.array_equal(result[0], expected_sample_0)





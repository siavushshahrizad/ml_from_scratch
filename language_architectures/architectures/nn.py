"""
File: nn.py
Date: 18.8.25
Summary: A small class implementing
a neural network.
"""

import numpy as np
from collections import Counter
from typing import List
from src.constants import UNK, BOS
from architectures.utils import split_words


class NeuralNetwork:
    """
    A simple neural network with one hidden
    layer, as in the Bengio et al (2003)
    paper. The purpose of the network is to
    predict the next word given a certain
    number of previous words.

    Params:
        embedding_size (int): word vector dimension
        hidden_size(int): hidden layer size
        context (int): number of preceding words.
    """

    def __init__(self, embedding_size:int=60, hidden_size:int=72, context:int=8) -> None:
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context = context
        self.w1 = np.random.rand(context)  # Weights input layer
        self.w2 = np.random.rand(hidden_size)  # Weights hidden layer

        self.size = 0  # Vocab size
        self.C = None  # Vector representation matrix
        self.int_to_word = None  # Mapping
        self.word_to_int = None

    def build_vocabulary(self, vocab: List[str], min_freq:int=3) -> None:
        """
        Uses the brown corpus to build vocab.
        This means it calculates and assigns
        the number of unique words in the brown
        corpus and it creates bidirectional
        mappings from ints to the words. Per the
        Bengio et al paper words that appear less than
        3 times are merged.

        One difference in paper is that syntactic marks
        are here not preserved.
        """
        unique = [UNK, BOS]          
        counts = Counter(vocab)          
        unique.extend(word for word, freq in counts.items() if freq > min_freq)
        self.word_to_int = {word: num for num, word in enumerate(unique)}
        self.int_to_word = {num: word for num, word in enumerate(unique)}
        self.size = len(unique) 
        self.C = np.random.rand(self.size, self.embedding_size)

    def _predict_is_valid(self, context_words: List[str]) -> tuple[bool, str]:
        """
        Func currently only checks
        if vocab initiated.
        But created in case want to
        expand later.
        """
        if not self.word_to_int:
            return False, "Vocab needs initialising"

        if len(context_words) == 0:
            return False, "No input to model"

        if len(context_words) != self.context:
            return False, "Len of input does not align with context size"

        return True, ""

    def _concat_embeddings(self, batch: List[List[str]]) -> np.ndarray:
        """
        Func does the forward computation for the embedding layer.
        Technically it is just retrieving word vectors and 
        concatenating them.

        The bengio paper says that cocnate means 
        x = (x(1), x(2), ..., x(n-1). I took that to mean that 
        each sample gives you flat array for its inputs of shape
        (context x embedding_size).
        """
        batch_size = len(batch)
        context = self.context
        embedding_size = self.embedding_size
        shape = (batch_size, context, embedding_size)
        batched_embeddings = np.zeros(shape)
        unk_id = self.word_to_int[UNK]

        for i, sample in enumerate(batch):
            for j, word in enumerate(sample):
                if word in self.word_to_int:
                    word_id = self.word_to_int[word]
                else: 
                    word_id = unk_id
                word_vec = self.C[word_id]
                batched_embeddings[i, j] = word_vec

        return batched_embeddings.reshape(batch_size, -1)

    def predict(self, context_words: List[str]):
        valid, err = self._predict_is_valid(context_words)
        if not valid:
            raise Exception(err)

        seperated = split_words(context_words)
        word_vecs = self._concat_embeddings(seperated)


        return 1 

    def _loss(self):
        pass

    def train(self):
        pass

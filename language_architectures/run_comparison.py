"""
File: run_comparison.py
Date: 18.8.25
Summary: The main programme
compares the performance of
different neural network
architectures on perplexity,
using the brown corpus.
"""

from architectures.nn import NeuralNetwork
from architectures.utils import split_data


def main():
    train, val, test = split_data()
    nn = NeuralNetwork()
    toy = train[:10000]         # TODO: Remove just for testing
    nn.build_vocabulary(toy)
    nn.predict("I like bananas")


if __name__ == "__main__":
    main()

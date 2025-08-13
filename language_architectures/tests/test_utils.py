"""
File: test_utils.py
Date: 18.8.25
Summary:  Tests utils
"""

import pytest
from architectures.utils import (
    split_data,
    split_words,
)

class TestUtils:
    @pytest.mark.slow
    def test_split_data(self):
        train, val, test = split_data()

        assert isinstance(train, list)
        assert len(train) > 0
        assert isinstance(train[0], str)
        assert len(train) > len(val) > len(test)

    @pytest.mark.fast
    def test_split_words(self, simple_input):
        words = simple_input
        separated = split_words(words)
        assert len(separated) == 2
        assert isinstance(separated, list)
        assert separated[0] == ["This", "is", "a", "test", "."]
        assert separated[1] == ["This", "is", "another", "test", "."]

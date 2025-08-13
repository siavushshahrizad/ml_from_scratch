"""
File: conftest.py
Date: 18.8.25
Summary:  Contains shared
test fixtures
"""

import pytest


@pytest.fixture
def simple_vocab():
    data = [
        "This",
        "This",
        "This",
        "This",
        "is",
        "is",
        "is",
        "is",
        "a",
        "a",
        "a",
        "a",
        "test",
        "test",
        "test",
        "test",
        "revenge",
        "duty",
        "home",
        "redemption",
        "father",
        "father",
        "duty",
        ".",
        ".",
        ".",
        "."
    ]
    return data


@pytest.fixture
def simple_input():
    words = ["This is a test.", "This is  another test."]
    return words


"""
File: utils.py
Date: 18.8.25
Summary:  Contains utils for 
networks
"""


import re
from typing import List
from nltk.corpus import brown


def split_data() -> tuple[list[str], list[str], list[str]]:
    """
    Loads the brown corpus and splits into
    train, val, test. It copies the Bengio
    paper so that first 800k are train,
    the next 200k are val, and rest test.
    """
    all_words = list(brown.words())     # Lazy object
    TRAIN_SIZE = 800_000
    VAL_SIZE = 200_000
    train = all_words[:TRAIN_SIZE]
    val = all_words[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
    test = all_words[TRAIN_SIZE+VAL_SIZE:]
    return train, val, test


def split_words(context_words: List[str]) -> List[List[str]]:
        regex = r"\w+|[^\w\s]"      # Splits words and punctuation
        seperated = [re.findall(regex, words) for words in context_words]
        return seperated



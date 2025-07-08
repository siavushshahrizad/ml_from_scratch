"""
File: logistic_regression.py
Date: 7 July 25
--------------------
SUMMARY
--------------------
This is the main file that runs
the logistic regression on the breast
cancer data.
"""


from utils import (
    FILE,
    load_and_clean_data,
    create_data_split
)


TRAIN_TO_OTHER_SET_RATIO = 0.8
VAL_TO_TEST_SET_RATIO = 0.5


def main():
    X, y = load_and_clean_data(FILE)
    X_train, X_other, y_train, y_other = create_data_split(X, y, TRAIN_TO_OTHER_SET_RATIO)
    X_val, X_test, y_val, y_test = create_data_split(X_other, y_other, VAL_TO_TEST_SET_RATIO)


if __name__ == "__main__":
    main()

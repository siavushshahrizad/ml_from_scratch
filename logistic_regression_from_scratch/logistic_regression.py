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
    load_and_clean_data
)


def main():
    X, y = load_and_clean_data(FILE)

if __name__ == "__main__":
    main()

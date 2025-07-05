"""
File: comparison.py
Date: 30.6.25
--------------------
SUMMARY
--------------------
The file compares the two optimisation methods.
This is the main file to run.
"""


from utils import (
    load_and_clean_data,
    run_comparison,
    save_plotted_data
)


NUM_PREDICTORS_TO_USE = [100, 500, 1000, 5000, 10000]


def main():
    """
    Programme loads a dataset and then simulates
    closed-form and gradient descent on synthetic features.
    Then results are printed to three files.
    """
    X, y = load_and_clean_data()
    results_cf, results_gd = run_comparison(X, y, NUM_PREDICTORS_TO_USE)

    save_plotted_data(
        results_cf, 
        results_gd, 
        0,
        "Time in miliseconds",
        "Fig 1 Time",
        "time.png", 
        NUM_PREDICTORS_TO_USE
    )

    save_plotted_data(
            results_cf, 
            results_gd, 
            1,
            "Gigabytes",
            "Fig 2 Memory",
            "memory.png", 
            NUM_PREDICTORS_TO_USE
    )

    save_plotted_data(
            results_cf, 
            results_gd, 
            2,
            "Loss",
            "Fig 3 Loss",
            "loss.png", 
            NUM_PREDICTORS_TO_USE
    )


if __name__ == "__main__":
    main()

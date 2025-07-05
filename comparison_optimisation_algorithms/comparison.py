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
    save_plotted_time,
    save_plotted_loss,
    save_plotted_memory
)


NUM_PREDICTORS_TO_USE = [100, 500, 1000, 5000, 10000]

def main():
    X, y = load_and_clean_data()
    results_cf, results_gd = run_comparison(X, y, NUM_PREDICTORS_TO_USE)
    save_plotted_time(results_cf, results_gd, NUM_PREDICTORS_TO_USE)
    save_plotted_memory(results_cf, results_gd, NUM_PREDICTORS_TO_USE)
    save_plotted_loss(results_cf, results_gd, NUM_PREDICTORS_TO_USE)


if __name__ == "__main__":
    main()

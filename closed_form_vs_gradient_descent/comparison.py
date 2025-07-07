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


# NUM_PREDICTORS_TO_USE = [100, 500, 1000, 5000, 10000, 50000, 100000]      # Dies at 50_000
NUM_PREDICTORS_TO_USE = [8]
# NUM_PREDICTORS_TO_USE = [1000, 2000, 3000, 4000, 5000]


def main():
    """
    Programme loads a dataset and then simulates
    closed-form and gradient descent on synthetic features.
    Then results are printed to three files.
    """
    ###### Load and simulate data ####
    #                                #
    ##################################
    X, y = load_and_clean_data()
    results_cf, results_gd = run_comparison(X, y, NUM_PREDICTORS_TO_USE)

    ######   Closed-form charts   ####
    #                                #
    ##################################
    save_plotted_data(
        results_cf, 
        0,
        "Seconds",
        "Fig 1 Time for closed-form",
        "./img/time_closed_form.png", 
        NUM_PREDICTORS_TO_USE
    )

    save_plotted_data(
        results_cf, 
        1,
        "Gigabytes",
        "Fig 2 Memory used for closed-form",
        "./img/memory_closed_form.png", 
        NUM_PREDICTORS_TO_USE
    )

    save_plotted_data(
        results_cf, 
        2,
        "Loss",
        "Fig 3 Test loss for closed-form",
        "./img/loss_closed_form.png", 
        NUM_PREDICTORS_TO_USE
    )

    ###  Gradient descent charts   ###
    #                                #
    ##################################

    save_plotted_data(
        results_gd, 
        0,
        "Seconds",
        "Fig 4 Time for gradient descent",
        "./img/time_gradient_descent.png", 
        NUM_PREDICTORS_TO_USE
    )

    save_plotted_data(
        results_gd, 
        1,
        "Gigabytes",
        "Fig 5 Memory for gradient descent",
        "./img/memory_gradient_descent.png", 
        NUM_PREDICTORS_TO_USE
    )

    save_plotted_data(
        results_gd, 
        2,
        "Loss",
        "Fig 6 Loss for gradient descent",
        "./img/loss_gradient_descent.png", 
        NUM_PREDICTORS_TO_USE
    )


if __name__ == "__main__":
    main()

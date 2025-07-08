# Introduction
Here, I built logistic regression from scratch for binary classification. This implementation focused on achieving three things:
- **Vectorisation**
- **L1 regularisation**
- **Comparison of gradient descent, batch gradient descent and mini-batch gradient descent**

I used a train/validation split to train the model to convergence, and then calculated the final loss on the test set.

# TL;DR


# Data
I used the 1992 [Breast Cancer Wisconsin] (https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) dataset. The data was surprisingly well-annotated. See the data folder. The task of the logistic regression, therefore, was to classify biopsy samples for potential breast cancer as benign or malevolent. 

# Lessons

## Vectorisation

## L1 regularisation

## Gradient descent variations

# Structure of this experiment
```
logistic_regression/
├── data/
│   ├── breast-cancer-wisconsin.data
│   └── breast-cancer-wisconsin.names
├── utils.py                            # Functions that make up programme 
├── logistic_regression.py              # Main programme
├── lr_tests.py                         # Unit tests 
├── README.md
└── requirements.txt                    # Libraries needed for running programme
└── logistic_regression.pdf             # My mathematical notes
```

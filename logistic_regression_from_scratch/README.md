# Introduction
Here, I built logistic regression from scratch for binary classification. This implementation focused on achieving three things:
- **Numerically stable vectorisation**
- **L1 regularisation**
- **Adam optimiser**

My from-scratch performance is then compared against the scikit-lean implementation.

# TL;DR


# Data
I used the 1992 [Breast Cancer Wisconsin] (https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) dataset. The data was surprisingly well-annotated. See the data folder. The task of the logistic regression, therefore, was to classify biopsy samples for potential breast cancer as benign or malevolent. 

# Sources
I used Grosse's lecture [notes](https://www.cs.toronto.edu/~mren/teach/csc411_19s/lec/lec08_notes.pdf) to implement the general logistic regression framework. I used the original [Kingma and Ba (2017)](https://arxiv.org/abs/1412.6980) paper to implement adam.

# Lessons
- **unit tests**: Unit-testing ML is notoriously hard. But incremental smell-tests can be built. For example, when building a loss function we can first check that a float is returned. Then we can create print statements to see if each term is as expected against known values before checking that the final result is as expected.



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

# Introduction
Here, I built logistic regression from scratch for binary classification. This implementation focused on achieving three things:
- **Numerically stable vectorisation**
- **L1 regularisation**
- **Adam optimiser**

My from-scratch performance is then compared against the scikit-learn implementation.

# TL;DR


# Data
I used the 1992 [Breast Cancer Wisconsin] (https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) dataset. The data was surprisingly well-annotated. See the data folder. The task of the logistic regression, therefore, was to classify biopsy samples for potential breast cancer as benign or malevolent. 

# Sources
I used Grosse's lecture [notes](https://www.cs.toronto.edu/~mren/teach/csc411_19s/lec/lec08_notes.pdf) to implement the general logistic regression framework. I used the original [Kingma and Ba (2017)](https://arxiv.org/abs/1412.6980) paper to implement adam. I also used the [Prechelt paper](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5) for early stopping.

# Lessons and reflections
- **unit tests**: I heavily rely on unit tests. For me they are quick smell tests or pre-bugging that tell me whether I am going in the right direction. I can't imagine doing this work without theses tests and without scratching my head for hours where I have gone wrong.



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

# Introduction
Here, I built logistic regression from scratch for binary classification. This implementation focused on achieving three things:
- **Numerically stable vectorisation**
- **L1 regularisation**
- **Adam optimiser**

My from-scratch performance is then compared against benchmarks.

# TL;DR


# Data
I used the 1992 [Breast Cancer Wisconsin] (https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) dataset. The data was surprisingly well-annotated. See the data folder. The task of the logistic regression, therefore, was to classify biopsy samples for potential breast cancer as benign or malevolent. 

# Sources
I used Grosse's lecture [notes](https://www.cs.toronto.edu/~mren/teach/csc411_19s/lec/lec08_notes.pdf) to implement the general logistic regression framework. I used the original [Kingma and Ba (2017)](https://arxiv.org/abs/1412.6980) paper to implement adam. I also used the [Prechelt paper](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5) for early stopping initially but later switched to the [Keras source code](https://github.com/keras-team/keras/blob/v3.10.0/keras/src/callbacks/early_stopping.py).

# Findings
TODO: write

# Meta lessons and notable bugs

## 1. Label Encoding Bug Leading to Negative Loss
**Discovery**: Loss values reaching -2000 seemed impossible...
**Root Cause**: Dataset used 2/4 labels instead of 0/1...
**Impact**: This cascaded into numerical instability...
**Fix**: Proper label preprocessing...

## 2. Early Stopping Formula Assumes Positive Losses
**The Bug**: Implemented formula from Prechelt (1998): `GL = 100 * (E_va/E_opt - 1)`
This assumes positive losses (like MSE), but breaks with negative log likelihood.

**Why it Failed Silently**: 
- When optimum_loss = -10 and current_loss = -8 (worse)
- Formula gives: (-8/-10 - 1)*100 = -20% (says improving!)
- Model stopped early when improving, ran forever when degrading

**The Fix**: Check if losses are negative and adjust formula accordingly
[Show corrected code]

**Lesson**: Always verify paper assumptions match your implementation

## l was 0.1

## What the loss actually is

## What the predictions really are

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

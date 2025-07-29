"""
File: charting.py
Date: 29 July 25
--------------------
SUMMARY
--------------------
The file contains functions 
for creating charts.
"""


import matplotlib.pyplot as plt

LOCATION = "./charts/"
FORMAT = ".png"


def create_bar_chart(title, x, y):
    min_val = min(y) * 0.95      
    max_val = max(y) * 1.05      
    plt.ylim(min_val, max_val)
    plt.bar(x, y)
    plt.title(title)
    filename = LOCATION + title + FORMAT
    filename = filename.lower().replace(" ", "")
    plt.savefig(filename)

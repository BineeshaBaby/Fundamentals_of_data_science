# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 22:28:56 2023

@author: BINEESHA BABY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

def read_newborn_data(file_path):
    """
    Reads data from a CSV file without headers and returns a Pandas DataFrame object.
    Args:
    file_path (str): The path to the CSV file.
    Returns:
    Pandas DataFrame object: The data read from the CSV file.
    """
    return pd.read_csv(file_path, header=None)

def calculate_histogram(data_column):
    """
    Calculates the histogram of the given data column using the Freedman-Diaconis rule.
    Args:
    data_column (Pandas Series object): The data column to calculate the histogram for.
    Returns:
    Tuple: A tuple containing the histogram counts and bin edges.
    """
    q1, q3 = np.percentile(data_column, [25, 75])
    iqr = q3 - q1
    bin_width = 2 * iqr / len(data_column) ** (1 / 3)
    bin_range = np.arange(np.min(data_column), np.max(data_column) + bin_width, bin_width)
    hist, bin_edges = np.histogram(data_column, bins=bin_range)
    return hist, bin_edges

def plot_histogram(hist, bin_edges, weights_mean, x_val):
    """
    Plots the histogram of the data, with vertical lines for the mean weight and the X value.
    Args:
    hist (numpy array): The histogram counts.
    bin_edges (numpy array): The bin edges for the histogram.
    weights_mean (float): The mean weight of the data.
    x_val (float): The value of X such that 75% of newborns from the distribution are born with a weight below X.
    Returns:
    None
    """
    legend_label = 'W_tilde = {:.2f}, X = {:.2f}'.format(weights_mean, x_val)
    plt.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], align='edge', edgecolor='black', alpha=0.8,label=legend_label)
    plt.xlabel("Newborn weight", fontsize=12,fontweight="bold")
    plt.ylabel("Count", fontsize=12,fontweight="bold")
    plt.title("Distribution of newborn weights", fontsize=14, fontweight="bold") # Make title bold
    plt.axvline(weights_mean, color='r', linestyle='--', label=r'$\tilde{W}$  (mean weight)')
    plt.axvline(x_val, color='g', linestyle='--', label='X')
    plt.text(0.7, 1.2, '(25.00% of newborns are born with a weight below 3.80)', ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0.7, 0.9))
    plt.show()


# Read in the data from the CSV file
data = read_newborn_data("D:/fundamentals/data7.csv")
print(data)

# Extract the column of data containing the newborn weights
weights = data.iloc[:, 0]

# Calculate the histogram of the weights column
hist, bin_edges = calculate_histogram(weights)

# Calculate the mean weight
weights_mean = np.mean(weights)
print("Mean weight:", weights_mean)

# Calculate the value of X such that 75% of newborns from the distribution are born with a weight below X
x_val = np.percentile(weights, 75)

# Print the value of X
print("The value of X is:", x_val)

# Calculate the percentage of newborns with a weight below X
percent_below_X = 100 - percentileofscore(weights, x_val)

# Print the result
print("{:.2f}% of newborns are born with a weight below {:.2f}.".format(percent_below_X, x_val))


# Plot the histogram with labels and legend
plot_histogram(hist, bin_edges, weights_mean, x_val)
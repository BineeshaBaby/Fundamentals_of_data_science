# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:44:48 2023

@author: BINEESHA BABY
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the data from the CSV file using Pandas, specifying that there is no header
data = pd.read_csv("D:/data7.csv", header=None)

# Extract the column of data containing the newborn weights
weights = data.iloc[:, 0]

# Use the Freedman-Diaconis rule to calculate the optimal number of bins for the histogram
q1, q3 = np.percentile(weights, [25, 75])
iqr = q3 - q1
bin_width = 2 * iqr / len(weights) ** (1 / 3)
n_bins = int((np.max(weights) - np.min(weights)) / bin_width)

# Define the range of the histogram bins
bin_range = np.arange(np.min(weights), np.max(weights) + bin_width, bin_width)

# Create a histogram of the weights using the defined bins
hist, bin_edges = np.histogram(weights, bins=bin_range)

# Print the histogram and the bin edges
print("Histogram:", hist)
print("Bin edges:", bin_edges)

# Assume that the weights array contains the weights of newborn babies in the given region
W_tilde = np.mean(weights)

# Calculate the value of X such that 75% of newborns from the distribution are born with a weight below X
X = np.percentile(weights, 75)

# Define the label for the legend
legend_label = 'W_tilde = {:.2f}, X = {:.2f}'.format(W_tilde, X)

# Plot the histogram with labels and legend
plt.hist(weights, bins=bin_edges, label=legend_label)
plt.xlabel("Newborn weight")
plt.ylabel("Count")
plt.title("Distribution of newborn weights")

# Add vertical lines for W_tilde and X
plt.axvline(W_tilde, color='r', linestyle='--', label=r'$\tilde{W}$')
plt.axvline(X, color='g', linestyle='--', label='X')

# Create the legend outside of the plot box
plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.9))

# Display the plot
plt.show()


from matplotlib.colors import ListedColormap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

classifiers = [(f"KNN (k={k})") for k in [1, 2, 5, 10, 15, 20]]

print(classifiers)

# ||a - b||^2 = ||a||^2 - 2 * a * b + ||b||^2
def euclidean_distance(a, b):
    # a is the test data matrix
    # b is the training data matrix
    # each row is a data point (i.e. vector)
    return (
        np.sum(a**2, axis=1, keepdims=True)
        - 2 * np.matmul(a, b.T)
        + np.sum(b**2, axis=1, keepdims=True).T
    )



a = np.array([[1,2,0], [3,1,1]])

b = np.array([[0,0,2], [4,3,1]])

print(np.sum(a**2, axis=1, keepdims=True))

print(np.matmul(a, b.T))

print(np.sum(b**2, axis=1, keepdims=True).T)

print(euclidean_distance(a, b))
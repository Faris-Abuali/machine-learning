import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# a = np.array([[1, 2]])

# print(a)
# print(a.squeeze())

# def compute_centroids(X, y):
#     """
#         X: np.array of shape (n_samples, 2 features)
#         y: np.array of shape (n_samples,)

#         e.g. X = np.array([[1, 2], [3, 4], [5, 6]])
#              y = np.array([-1, 1, -1])

#              centroids = np.array([[3, 4], [5, 6]])
#     """
#     centroids = []
#     for label in np.unique(y):
#         centroids.append(np.mean(X[y == label], axis=0))
#     return np.array(centroids)

def compute_centroids(X, y):
    for label in np.sort(np.unique(y)):
        yield np.mean(X[y==label], axis=0)    

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([-1, 1, -1])

centroids = compute_centroids(X, y)

X = np.array([[1, 2], [4, 5], [-1, 2]])
w = np.array([3, -1])
b = 1

pred = np.matmul(X, w) + b
pred1 = np.dot(X, w) + b
# So what is the difference between np.matmul and np.dot?
# np.matmul is the matrix multiplication operator, while np.dot is the dot product operator.
# For 2D arrays, np.matmul is equivalent to np.dot.
 
print(pred)
print(pred1)

# print(X)
# print(X[y == 1])

# print(centroids)
# print(list(centroids))
# print(y == -1)

# boool = np.array([True, False, True])
# xx = np.arange(3)[boool]

# print(X[boool])
# print(xx)

c_neg = np.array([3, 4])
c_pos = np.array([1, 2])

print(np.stack([c_neg, c_pos], axis=-1))
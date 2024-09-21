from matplotlib.colors import ListedColormap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# This function computes the Euclidean distance between each pair of test and training data points.
# ||a - b||^2 = ||a||^2 - 2 * a * b + ||b||^2
def euclidean_distance(a, b):
    # a is the test data matrix (mxd)
    # b is the training data matrix (nxd)
    # each row is a data point (i.e. vector)

    # 1. first term produces a column vector (mx1) with the sum of the squares of each row of a
    # 2. second term produces a matrix (mxn) with the dot product of each vector of `a` with each vector of `b`
    # 3. third term produces a row vector (1xn) with the sum of the squares of each row of b

    # Now broadcasting will stretch the first term to have shape (m, n)
    # so that it can be subtracted from the second term (mxn).

    # Whey do we take the transpose of the third term?
    # Because the result without the transpose would be a column vector (n, 1)
    # and this can't be broadcasted to become mxn.
    # So we take the transpose to get a row vector (1, n) which can be broadcasted to mxn.
    return (
        np.sum(a**2, axis=1, keepdims=True)
        - 2 * np.matmul(a, b.T)
        + np.sum(b**2, axis=1, keepdims=True).T
    )


# Broadcasting works by "stretching" arrays so that they have compatible shapes:
# 1. Arrays with a smaller dimension (like a and c in this case) are broadcast along the missing dimensions.
# 2. If one of the dimensions is 1, NumPy will stretch it to match the size of the other array.


class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5, dist_function=euclidean_distance):
        super(KNN, self).__init__()

        self.k = k
        self.dist_function = dist_function
        self.train_points = None

    def fit(self, X, y):
        # KNN does not need to train, just store the training data
        self.train_points = X, y
        # train_points is a tuple of X (the data matrix) and y (the labels)
        # X shape = (n, d) where n is the number of data points and d is the number of features.
        # y shape = (n,) where n is the number of data points.

    def predict(self, X):
        from collections import Counter

        if self.train_points is None:
            raise ValueError(
                "Predict can only be called after supplying training data with fit first!"
            )

        # Compute distance to each training point
        dist_mat = self.dist_function(X, self.train_points[0])
        # remember: X here is the test data matrix
        # and self.train_points[0] is the training data matrix
        # dist_mat shape = (m, n) where m is the number of test data points and n is the number of training data points.

        # So each row is basically the distances from a test data point to all training data points.

        # Generate an empty array of shape (X.shape[0],) with the same data type
        # as self.train_points[1] (which is y, the labels vector of the training data).

        # X.shape[0] is the number of test data points (m)
        # self.train_points[1] is the labels vector of the training data
        num_test_data_points = X.shape[0] # number of test data points (n)
        y = self.train_points[1]
        y_res = np.empty(num_test_data_points, dtype=y.dtype)
        # y_res shape = (m,) where m is the number of test data points.

        for i in range(num_test_data_points):
            dist = dist_mat[i]
            # `dist` is a row of all distances from the i-th test point to all training points
            # Thus, dist is a row vector of shape (1, n) where n is the number of training data points.

            # Get indices of k smallest elements
            # np.argpartition returns an ordered array of indices of the k smallest elements
            # We only need the indices of the first k elements since they are the smallest

            # Why it works:
            # Because np.argpartition returns an array of indices where the kth element is in the correct position
            # and all elements to the left of it are smaller than it.
            # We care about the first k indices because they are the indices of the k smallest elements.
            knn = np.argpartition(dist, self.k)[: self.k]

            # knn now looks like this:
            # [index_of_smallest_distance, index_of_2nd_smallest_distance, ..., index_of_kth_smallest_distance

            # Get the labels of the k nearest neighbours
            knn = y[knn]
            # knn now looks like this:
            # [label_of_smallest_distance, label_of_2nd_smallest_distance, ..., label_of_kth_smallest_distance]

            # perform a majority vote to get the most common class
            classes = Counter(knn).most_common()
            # classes is a list of tuples where the first element is the class and the second element is the count
            # e.g. [(-1, 3), (1, 2)] means that class -1 appeared 3 times and class 1 appeared 2 times

            # classes.most_common() looks like this:
            # e.g. [(-1, 3), (1, 2), ...]

            # Get all classes that have the same count as the maximum
            # This is needed when k is even and there is no clear majority
            # In this case, we choose the class with the smallest value
            # e.g. if classes = [(-1, 3), (1, 3), (2, 2)], we choose -1
            res = [
                classes[j][0]
                for j in range(len(classes))
                if j == 0 or classes[j - 1][1] == classes[j][1]
            ]
            # res looks like this:
            # e.g. [-1, 1, ...]

            y_res[i] = min(res)
            # y_res shape = (m,) where m is the number of test data points

        return y_res
        # y_res shape = (m,) where m is the number of test data points


# Generate toy data (two moons dataset)
n = 200
X, Y = datasets.make_moons(n, noise=0.25, random_state=1234)
# remember: X is the input data, Y is the vector of labels

# X is a data matrix with shape (n, 2)
# Y is a vector of labels with shape (n,)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=12345
)
# train_test_split splits the data into random train and test subsets

classifiers = [(KNN(k), f"KNN ($k={k}$)") for k in [1, 2, 5, 10, 15, 20]]
# Plot decision surface
# First generate grid
res = 200  # Resolution of the grid in cells
x_max, y_max = np.max(X, axis=0) + 0.25
# here x_max and y_max are the max coordinates of the data points.
# remember: X is the input data matrix of shape (n, 2)

x_min, y_min = np.min(X, axis=0) - 0.25
# here x_min and y_min are the min coordinates of the data points.
grid_x, grid_y = np.meshgrid(
    np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res)
)
# remember: np.linspace returns evenly spaced numbers over a specified interval.

# grid_x shape = (res, res)
# grid_x looks like this:
# [[x_min, x_min + step, x_min + 2*step, ..., x_max],
#  [x_min, x_min + step, x_min + 2*step, ..., x_max],
#  ...
#  [x_min, x_min + step, x_min + 2*step, ..., x_max]]
# where step = (x_max - x_min) / (res - 1)

# grid_y shape = (res, res)
# grid_y looks like this:
# [[y_min, y_min, y_min, ..., y_min],
#  [y_min + step, y_min + step, y_min + step, ..., y_min + step],
#  ...
#  [y_max, y_max, y_max, ..., y_max]]
# where step = (y_max - y_min) / (res - 1)

# Get test array from grid
grid_input = np.c_[grid_x.reshape(-1), grid_y.reshape(-1)]
# grid_input.shape = (res^2, 2)
# grid_input is the test data matrix for the grid points
# meaning that each point in the grid is a test data point.

# np.c_ concatenates along the second axis (column-wise)
# grid_input is a matrix where each row is a pair of coordinates in the grid

# grid_input looks like this:
# [[x_min, y_min],
#  [x_min + step, y_min],
#  [x_min + 2*step, y_min],
#  ...
#  [x_max, y_min],
#  [x_min, y_min + step],
#  [x_min + step, y_min + step],
#  ...
#  [x_max, y_min + step],
#  ...
#  [x_min, y_max],
#  [x_min + step, y_max],
#  ...
#  [x_max, y_max]]

# np._c is a short for "column stack" which stacks 1D arrays as columns into a 2D array.
# Purpose of np.c_:
# np.c_ is a shorthand for concatenating arrays column-wise. It helps in creating
# new arrays by stacking columns together from different input arrays.

# - grid_x.reshape(-1) and grid_y.reshape(-1) both create 1D arrays (or vectors).
# - np.c_ is used to concatenate these 1D arrays column-wise, which results in a 2D array where each row is a pair of coordinates: [x_value, y_value].

# The result is that grid_input becomes a 2D array where:

# The first column contains the flattened grid_x values.
# The second column contains the flattened grid_y values.

rows = (len(classifiers) + 2) // 3

fig, axes = plt.subplots(rows, 3, sharex=True, sharey=True, figsize=(12, 4 * rows))
# figsize(width, height) in inches
# figsize is the size of the entire figure (not the size of each subplot)

for (clf, name), ax in zip(classifiers, axes.ravel()):
    clf.fit(X_train, Y_train)
    # 👇 test the classifier on the test data.
    # returns the mean accuracy on the given test data and labels.
    score = clf.score(X_test, Y_test)
    # score is a float representing the accuracy of the classifier on the test data.

    grid_out = clf.predict(grid_input).reshape(grid_x.shape)

    # grid_out before reshape has shape (res^2,) (e.g. when res=200, shape=(40000,))
    
    # because as you noticed, the for loop in the predict function appends the result of 
    # each iteration to y_res. (which is a 1D array of shape (m,)) 
    # where m is the number of test data points.

    # grid_out after reshape has shape (res, res) (e.g. when res=200, shape=(200, 200))

    # ⚠️ Predictions are made for each point in the grid (grid_input),
    # then reshaped using `reshape` to match the grid dimensions.

    cmap = ListedColormap([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    ax.set_title(f"Classifier: {name}, Acc.: {score:.3f}")

    ax.contourf(grid_x, grid_y, grid_out, alpha=0.5, cmap=plt.cm.RdBu)
    # remember:
    # grid_x shape = (res, res)
    # grid_y shape = (res, res)
    # grid_out shape after reshape = (res, res)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cmap, edgecolor="k")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=cmap, marker="x")

    # X_train shape = (n, 2) where n is the number of training data points
    # X_train[:, 0] is the x-coordinates of the training data points
    # X_train[:, 1] is the y-coordinates of the training data points

    # X_test shape = (m, 2) where m is the number of test data points
    # X_test[:, 0] is the x-coordinates of the test data points
    # X_test[:, 1] is the y-coordinates of the test data points

fig.tight_layout()
plt.show()
plt.close(fig)

# Visualizing the Decision Surface with plt.contourf:
# - plt.contourf takes three arguments:
#   grid_x and grid_y: These define the positions (x and y coordinates) of each point in the grid where the color will be plotted.
#   grid_out: This provides the color values for each point based on the predicted class label (e.g., red for class 1, blue for class -1).
# 
# - The combination of these three arguments allows plt.contourf to create a filled contour plot that depicts the decision surface.
# - Areas with the same predicted class label will have a uniform color, while the boundaries between classes will be shown as color transitions.

# Key Points:
    # - grid_x and grid_y act as a "map" defining where to place colors on the plot.
    # - grid_out provides the color values for each point based on the predicted class label.
    # - Together, they create a visual representation of the decision surface learned by the KNN classifier.
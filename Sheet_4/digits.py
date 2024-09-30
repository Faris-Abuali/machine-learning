from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
X, Y = digits.data, digits.target

# Get all data points whose target is 3
X_3 = X[Y == 3]

# Plot the first five data points in one window
plt.figure()
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_3[i].reshape(8, 8), cmap='gray') 
    # reshape to 8x8 matrix because all data points were flattened to (64,) shape
    plt.axis('off')
plt.show()


import numpy as np
import matplotlib.pyplot as plt


def data_spiral(points=20, dimensionality=2, classes=3):
    # https://cs231n.github.io/neural-networks-case-study/
    N = points  # number of points per class
    D = dimensionality  # dimensionality
    K = classes  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


def visualize(X, y):
    # lets visualize the 2D data:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()


def one_hot(inputs):
    input_onehot = np.max(inputs) + 1
    input_onehot = np.eye(input_onehot)[inputs]
    return input_onehot

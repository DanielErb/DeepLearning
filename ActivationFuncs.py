import numpy as np
from matplotlib import pyplot as plt


def softmax(W, X):
    temp = np.dot(X.T, W)
    # fixed = temp - np.max(temp)
    return np.exp(temp) / np.sum(np.exp(temp), axis=0)


def cross_entropy_loss(y, y_hat):
    return -np.sum(y * np.log(y_hat))


def loss(W, X, y):
    y_hat = softmax(W, X)
    return np.mean(cross_entropy_loss(y, y_hat))


def gradient(X_batch, y_batch, W, B):
    after_softmax = softmax(W, X_batch)
    print("after_softmax.shape", after_softmax.shape)
    print("y_batch.shape", y_batch.shape)
    print()
    dw = np.dot(X_batch, (after_softmax - y_batch)) / X_batch.shape[1]
    dx = np.dot(W, (after_softmax - y_batch).T) / X_batch.shape[1]
    db = np.sum(after_softmax - y_batch) / X_batch.shape[1]
    grad = np.concatenate((dw, db), axis=0)
    return grad, dx


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def gradient_check(W, X, y):
    d = normalize(np.random.randn(W.shape[0]))
    epsilon = 1
    for i in range (1, 10):
        epsilon = epsilon / (2** i)
        print("epsilon", epsilon)
        loss1 = loss(W + epsilon * d, X, y)
        loss2 = loss(W - epsilon * d, X, y)



# Example usage
np.random.seed(42)
W = np.random.randn(5, 5)  # Example weights
X = np.random.randn(5, 3)  # Example input
# y is supposed to be 3 5
y = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])  # Example one-hot encoded labels

gradient_check(W, X, y)

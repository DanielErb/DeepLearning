import numpy as np

# z = w^T x + b
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def cross_entropy_loss(y, y_hat):
    return -np.sum(y * np.log(y_hat))

def Loss_softmax_single(X, y, w, b):
    z = np.dot(w.T, X) + b
    y_hat = softmax(z)
    return cross_entropy_loss(y, y_hat)

def Loss_softmax(single_losses):
    return np.mean(single_losses)


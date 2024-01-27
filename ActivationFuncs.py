import numpy as np
from matplotlib import pyplot as plt


# def softmax(A):
#     expA = np.exp(A)
#     return expA / expA.sum(axis=1, keepdims=True)
#
#
def softmax(W, X, b):
    temp = np.dot(X.T, W) + b

    sum = np.sum(np.exp(temp), axis=1)
    # print("temp", np.exp(temp))
    # print("sum", np.sum(np.exp(temp), axis=1))
    # print("total", np.exp(temp) / np.sum(np.exp(temp), axis=1, keepdims=True))
    return np.exp(temp) / np.sum(np.exp(temp), axis=1, keepdims=True)


def cross_entropy_loss(y, y_hat):
    # print("check", -(y * np.log(y_hat)))
    # print("check_y_hat", y_hat)
    # print("check_y", y)
    # print("check_y_hat", y_hat.shape)
    # print("check_y", y.shape)
    # print("np,log", np.log(y_hat))
    return -(y * np.log(y_hat))


# def loss(A, y):
#      y_hat = softmax(A)
#      return np.mean(cross_entropy_loss(y, y_hat))

# def loss(W, X, y):
#     y_hat = softmax(W, X)
#     cross_entropy = cross_entropy_loss(y, y_hat)
#     # sum = np.sum(np.sum(cross_entropy, axis=1), axis=0)
#     # print("sum", sum)
#     print("loss", np.mean(np.sum(cross_entropy_loss(y, y_hat), axis=1)))
#     return np.mean(np.sum(cross_entropy_loss(y, y_hat), axis=1))


def loss(W, X, b ,y):
    m = X.shape[1]
    # print("y*log", y * np.log(softmax(W, X)))
    # print("-1/m", -1/m*(y * np.log(softmax(W, X))))
    loss_check = -1 / m * (y * np.log(softmax(W, X, b))).sum()
    print("loss_check", loss_check)
    return loss_check


def gradient(X_batch, y_batch, W, b):
    # temp = np.dot(X.T, W)
    # after_softmax =  np.exp(temp). / sum(np.exp(temp), axis=0)
    after_softmax = softmax(W, X_batch, b)
    # print("after_softmax.shape", after_softmax.shape)
    # print("y_batch.shape", y_batch.shape)
    # print()
    dw = np.dot(X_batch, (after_softmax - y_batch)) / X_batch.shape[1]
    dx = np.dot(W, (after_softmax - y_batch).T) / X_batch.shape[1]
    # print("dx.shape", dx.shape)
    db = np.sum((after_softmax - y_batch), axis=0, keepdims=True) / X_batch.shape[1]
    print("db", db)
    # print("db.shape", db.shape)
    # grad = np.concatenate((dw, db), axis=0)
    return dw, dx, db



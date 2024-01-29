import numpy as np


class Softmax:
    def __init__(self, W, b):
        # need to save only W and b because these are the only parameters that we are going to change
        # and X and Y are just sampled per each approch so we dont need to save them
        self.W = W
        self.b = b

    def activation(self, X, W=None, b=None):
        if (W is None):
            W = self.W
        if (b is None):
            b = self.b
        temp = np.dot(X.T, W) + b
        # sum = np.sum(np.exp(temp), axis=1)
        return np.exp(temp) / np.sum(np.exp(temp), axis=1, keepdims=True)

    def loss(self, X, Y, W=None, b=None):
        if (W is None):
            W = self.W
        if (b is None):
            b = self.b
        m = X.shape[1]
        # print("y*log", y * np.log(softmax(W, X)))
        # print("-1/m", -1/m*(y * np.log(softmax(W, X))))
        loss_check = -1 / m * (Y * np.log(self.activation(X, W, b))).sum()
        # print("loss_check", loss_check)
        return loss_check

    def gradient(self, X, Y, W=None, b=None):
        if (W is None):
            W = self.W
        if (b is None):
            b = self.b
        # temp = np.dot(X.T, W)
        # after_softmax =  np.exp(temp). / sum(np.exp(temp), axis=0)
        after_softmax = self.activation(X, W, b)
        # print("after_softmax.shape", after_softmax.shape)
        # print("y_batch.shape", y_batch.shape)
        # print()
        dw = np.dot(X, (after_softmax - Y)) / X.shape[1]
        dx = np.dot(W, (after_softmax - Y).T) / X.shape[1]
        # print("dx.shape", dx.shape)
        db = np.sum((after_softmax - Y), axis=0, keepdims=True) / X.shape[1]
        # print("db", db)
        # print("db.shape", db.shape)
        # grad = np.concatenate((dw, db), axis=0)
        return dw, dx, db


class LayerFunc:
    def __init__(self, W, b, Activation):
        # need to save only W and b because these are the only parameters that we are going to change
        # and X and Y are just sampled per each approch so we dont need to save them
        if Activation.lower() != "tanh" and Activation.lower() != "relu":
            raise ValueError(f"Invalid input: '{Activation}'. Please provide Relu or Tanh")
        self.W = W
        self.b = b
        self.Activate = Activation

    def activation(self, X, W=None, b=None):
        if (W is None):
            W = self.W
        if (b is None):
            b = self.b

        if self.Activate.lower() == "tanh":
            return np.tanh(np.dot(W, X) + b)

        return self.reluActivation(X)

    def reluActivation(self, X, W=None, b=None):
        if (W is None):
            W = self.W
        if (b is None):
            b = self.b
        return np.maximum(0, np.dot(W, X) + b)

    def reluDerivative(self, t):
        return np.where(t > 0, 1, 0)

    def tanhDerivative(self, t):

        return 1 - np.tanh(t) ** 2

    def gradient(self, X, V, W=None, b=None):
        if (W is None):
            W = self.W
        if (b is None):
            b = self.b

        derivative = None
        if self.Activate == "relu":
            derivative = self.reluDerivative
        else:
            derivative = self.tanhDerivative

        output_before_activation = np.dot(W, X) + b
        dbV = np.sum(derivative(output_before_activation) * V, axis=1, keepdims=True)
        print("dbV", dbV)
        dwV = np.dot((derivative(output_before_activation) * V), X.T)
        dxV = np.dot(W.T, (derivative(output_before_activation) * V))
        return dwV, dxV, dbV

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
        # print("X.T.shape", X.T.shape)
        # print("W.shape", W.shape)
        # print("b.shape", b.shape)

        temp = np.dot(X.T, W) + b - np.max(np.dot(X.T, W) + b, axis=1, keepdims=True)
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
        # print("after_softmax.shape", after_softmax.shape)
        # print("Y.shape", Y.shape)
        # print("X.shape", X.shape)
        dw = np.dot(X, (after_softmax - Y)) / X.shape[1]
        dx = np.dot(W, (after_softmax - Y).T) / X.shape[1]
        # print("dx.shape", dx.shape)
        db = np.sum((after_softmax - Y), axis=0, keepdims=True) / X.shape[1]
        # print("db", db)
        # print("db.shape", db.shape)
        # grad = np.concatenate((dw, db), axis=0)
        return dw, dx, db


class LayerFunc:
    def __init__(self, W, b, Activation, networkType, W2=None):
        # need to save only W and b because these are the only parameters that we are going to change
        # and X and Y are just sampled per each approch so we dont need to save them
        if Activation.lower() != "tanh" and Activation.lower() != "relu":
            raise ValueError(f"Invalid input: '{Activation}'. Please provide Relu or Tanh")
        self.W = W
        self.b = b
        self.Activate = Activation
        self.networkType = networkType.lower()
        self.W2 = W2

    def activation(self, X, W=None, b=None, W2=None):
        self.X = X
        if (W is None):
            W = self.W
        if (b is None):
            b = self.b
        if (W2 is None):
            W2 = self.W2

        if self.Activate.lower() == "relu":
            return self.reluActivation(X)
        else:
            return self.tanhActivation(X, W, b, W2)  # for jacobian test

    def tanhActivation(self, X, W=None, b=None, W2=None):
        if (W is None):
            W = self.W
        if (b is None):
            b = self.b
        if (W2 is None):
            W2 = self.W2
        if self.networkType == 'standard':
            return np.tanh(np.dot(W, X) + b)
        else:
            return X + np.dot(W2, np.tanh(np.dot(W, X) + b))

    def reluFunc(self, t):
        return np.maximum(0, t)

    def reluActivation(self, X, W=None, b=None, W2=None):
        if (W is None):
            W = self.W
        if (b is None):
            b = self.b
        if (W2 is None):
            W2 = self.W2
        if self.networkType == 'standard':
            return self.reluFunc(np.dot(W, X) + b)
        else:
            return X + np.dot(W2, self.reluFunc(np.dot(W, X) + b))

    def reluDerivative(self, t):
        return np.where(t > 0, 1, 0)

    def tanhDerivative(self, t):

        return 1 - np.tanh(t) ** 2

    def gradient(self, X, V, W=None, b=None, W2=None):
        if (W is None):
            W = self.W
        if (b is None):
            b = self.b
        if (W2 is None):
            W2 = self.W2


        derivative = None
        if self.Activate == "relu":
            derivative = self.reluDerivative
        else:
            derivative = self.tanhDerivative

        activationFunc = None
        if self.Activate == "relu":
            activationFunc = self.reluFunc
        else:
            activationFunc = np.tanh

        output_before_activation = np.dot(W, X) + b
        if self.networkType == 'standard':
            dbV = np.sum(derivative(output_before_activation) * V, axis=1, keepdims=True)
            dwV = np.dot((derivative(output_before_activation) * V), X.T)
            dxV = np.dot(W.T, (derivative(output_before_activation) * V))
            dw2V = None  # in standard network there is no W2 for a single layer
            return dwV, dw2V, dxV, dbV
        else:
            dbV = np.sum(derivative(output_before_activation) * np.dot(W2.T, V), axis=1, keepdims=True)
            dwV = np.dot(derivative(output_before_activation) * np.dot(W2.T, V), X.T)
            if self.Activate == "relu":
                dw2V = np.dot(V, self.reluFunc(output_before_activation).T)
            else:
                dw2V = np.dot(V, np.tanh(output_before_activation).T)
            dxV = V + np.dot(W.T, derivative(output_before_activation) * np.dot(W2.T, V))
            return dwV, dw2V, dxV, dbV

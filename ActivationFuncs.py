import numpy as np


class Softmax:
    def __init__(self, W, b):
        # need to save only W and b because these are the only parameters that we are going to change
        # and X and Y are just sampled per each approch so we dont need to save them
        self.W = W
        self.b = b

    def activation(self, X, W = None, b=None):
        if(W is None):
            W = self.W
        if(b is None):
            b = self.b
        temp = np.dot(X.T, W) + b
        #sum = np.sum(np.exp(temp), axis=1)
        return np.exp(temp) / np.sum(np.exp(temp), axis=1, keepdims=True)

    def loss(self,X, Y, W=None, b=None):
        if(W is None):
            W = self.W
        if(b is None):
            b = self.b
        m = X.shape[1]
        # print("y*log", y * np.log(softmax(W, X)))
        # print("-1/m", -1/m*(y * np.log(softmax(W, X))))
        loss_check = -1 / m * (Y * np.log(self.activation(X, W, b))).sum()
        #print("loss_check", loss_check)
        return loss_check

    def gradient(self, X, Y, W = None, b = None):
        if(W is None):
            W = self.W
        if(b is None):
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
        #print("db", db)
        # print("db.shape", db.shape)
        # grad = np.concatenate((dw, db), axis=0)
        return dw, dx, db

# def cross_entropy_loss(y, y_hat):
#     # print("check", -(y * np.log(y_hat)))
#     # print("check_y_hat", y_hat)
#     # print("check_y", y)
#     # print("check_y_hat", y_hat.shape)
#     # print("check_y", y.shape)
#     # print("np,log", np.log(y_hat))
#     return -(y * np.log(y_hat))


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



class tanh:
    def __init__(self, W, b):
        # need to save only W and b because these are the only parameters that we are going to change
        # and X and Y are just sampled per each approch so we dont need to save them
        self.W = W
        self.b = b

    def activation(self, X, W = None, b=None):
        if(W is None):
            W = self.W
        if(b is None):
            b = self.b
        return np.tanh(np.dot(W, X) + b)
    
    def gradient(self, X, V, W = None, b = None):
        if(W is None):
            W = self.W
        if(b is None):
            b = self.b
        dw = 

class ActiovationFunc:
    def __init__(self, W, b, Activation , Derrative):
        # need to save only W and b because these are the only parameters that we are going to change
        # and X and Y are just sampled per each approch so we dont need to save them
        self.W = W
        self.b = b
        self.Activate = Activation
        self.Derrative = Derrative
    
    def activation(self, X, W = None, b=None):
        if(W is None):
            W = self.W
        if(b is None):
            b = self.b
        return self.Activate(np.dot(W, X) + b)
    
    def gradient(self, X, V, W = None, b = None):
        if(W is None):
            W = self.W
        if(b is None):
            b = self.b
        dwV dbV dxV = self.Derrative(X, V, W, b)
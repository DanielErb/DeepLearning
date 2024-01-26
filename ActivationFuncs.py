import numpy as np
from matplotlib import pyplot as plt


# def softmax(A):
#     expA = np.exp(A)
#     return expA / expA.sum(axis=1, keepdims=True)
#
#
def softmax(W, X):
    temp = np.dot(X.T, W)
    sum = np.sum(np.exp(temp), axis=1)
    print("temp", np.exp(temp))
    print("sum", np.sum(np.exp(temp), axis=1))
    print("total", np.exp(temp) / np.sum(np.exp(temp), axis=1, keepdims=True))
    return np.exp(temp) / np.sum(np.exp(temp), axis=1, keepdims=True)


def cross_entropy_loss(y, y_hat):
    print("check", -(y * np.log(y_hat)))
    # print("check_y_hat", y_hat)
    # print("check_y", y)
    # print("check_y_hat", y_hat.shape)
    # print("check_y", y.shape)
    # print("np,log", np.log(y_hat))
    return -(y * np.log(y_hat))


# def loss(A, y):
#      y_hat = softmax(A)
#      return np.mean(cross_entropy_loss(y, y_hat))

def loss(W, X, y):
    y_hat = softmax(W, X)
    cross_entropy = cross_entropy_loss(y, y_hat)
    #sum = np.sum(np.sum(cross_entropy, axis=1), axis=0)
    print("sum", sum)
    print("loss", np.mean(np.sum(cross_entropy_loss(y, y_hat), axis=1)))
    return np.mean(np.sum(cross_entropy_loss(y, y_hat), axis=1))


def gradient(X_batch, y_batch, W, B):
    # temp = np.dot(X.T, W)
    # after_softmax =  np.exp(temp). / sum(np.exp(temp), axis=0)
    after_softmax = softmax(W, X_batch)
    print("after_softmax.shape", after_softmax.shape)
    print("y_batch.shape", y_batch.shape)
    print()
    dw = np.dot(X_batch, (after_softmax - y_batch)) / X_batch.shape[1]
    dx = np.dot(W, (after_softmax - y_batch).T) / X_batch.shape[1]
    print("dx.shape", dx.shape)
    db = np.sum((after_softmax - y_batch), axis=0, keepdims=True) / X_batch.shape[1]
    print("db.shape", db.shape)
    # grad = np.concatenate((dw, db), axis=0)
    return dw, dx, db


def normalize(v):
    norm = np.linalg.norm(v, axis =0)
    print("v", v)
    print("norm", norm)
    print("v/norm", v / norm)
    return v / norm


def gradient_checkX(W, X, y, b):
    d = normalize(np.random.randn(X.shape[0], X.shape[1]))
    print("d", d)
    print(W.shape)
    epsilon = 1
    epsilons = []
    differences = []
    differences_with_grad = []
    for i in range(1, 10):
        epsilon = epsilon / (2 ** i)
        dw, dx, db = gradient(X, y, W, b)
        # print("epsilon", epsilon)
        loss_epsilon = loss(W, X + epsilon * d, y)
        loss_regular = loss(W, X, y)
        difference = np.abs(loss_epsilon - loss_regular)
        print("loss_epsilon", loss_epsilon - loss_regular)
        print("loss_shape", (loss_epsilon - loss_regular).shape)
        print("d", d)
        #print("dx", dx)
        #print("check vdot", np.vdot(epsilon * d.T, dx))
        print("dx", dx)

        difference_with_grad = np.abs(loss_epsilon - loss_regular - np.vdot(epsilon * d, dx))
        epsilons.append(epsilon)
        differences.append(difference)
        differences_with_grad.append(difference_with_grad)
        print("epsilon", epsilon)
        print("difference", difference)
        print("difference_with_grad", difference_with_grad)
        #print("difference_with_grad", np.mean(difference_with_grad))
        print()
    # print("difference", difference)
    # print("difference_with_grad", difference_with_grad)
    # print(type(differences_with_grad))
    plt.plot(epsilons, differences, label='Without Gradient')
    plt.plot(epsilons, differences_with_grad, label='With Gradient')

    # Set x and y labels
    plt.xlabel('Epsilon')
    plt.ylabel('Difference')

    # Add a legend to differentiate between the two lines
    plt.legend()

    # Set the title
    plt.title('Difference by Epsilon')

    # Show the plot
    plt.show()


def gradient_checkW(W, X, y, b):
    d = normalize(np.random.randn(W.shape[0], W.shape[1]))
    print(W.shape)
    epsilon = 1
    epsilons = []
    differences = []
    grad_diff = []
    for i in range(1, 10):
        epsilon = epsilon / (2 ** i)
        print("epsilon", epsilon)
        loss_epsilon = loss(W + epsilon * d, X, y)
        dw, dx, db = gradient(X, y, W, b)
        loss_regular = loss(W, X, y)
        difference = np.abs(loss_epsilon - loss_regular)
        difference_with_grad = np.abs(loss_epsilon - loss_regular - np.vdot(epsilon * d, dw))
        epsilons.append(epsilon)
        differences.append(difference)
        grad_diff.append(difference_with_grad)
    plt.plot(epsilons, differences, label='Without Gradient')
    plt.plot(epsilons, grad_diff, label='With Gradient')

    plt.xlabel('Epsilon')
    plt.ylabel('Difference')
    plt.title('Difference by Epsilon')
    plt.show()


# Example usage
#np.random.seed(42)
W = np.random.randn(5, 5)  # Example weights
X = np.random.randn(5, 3)  # Example input
b = np.random.randn(1, 5)  # Example biases
# y is supposed to be 3 5
y = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])  # Example one-hot encoded labels

gradient_checkW(W, X, y, b)

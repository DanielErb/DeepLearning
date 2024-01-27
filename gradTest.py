import numpy as np
import matplotlib.pyplot as plt
from ActivationFuncs import gradient
from ActivationFuncs import loss


def normalize(v):
    norm = np.linalg.norm(v, axis=0)
    # print("v", v)
    # print("norm", norm)
    # print("v/norm", v / norm)
    return v / norm


def gradient_checkX(W, X, y, b):
    d = normalize(np.random.randn(X.shape[0], X.shape[1]))
    # print("d", d)
    # print(W.shape)
    epsilon = 1
    epsilons = []
    differences = []
    differences_with_grad = []
    for i in range(1, 20):
        epsilon = (0.5 ** i)
        dw, dx, db = gradient(X, y, W, b)
        # print("epsilon", epsilon)
        loss_epsilon = loss(W, X + epsilon * d, b, y)
        # loss_epsilon_check = loss2(W, X + epsilon * d, y)
        loss_regular = loss(W, X, b, y)
        # loss_regular_check = loss2(W, X, y)
        difference = np.abs(loss_epsilon - loss_regular)
        # print("loss_epsilon", loss_epsilon - loss_regular)
        # print("loss_shape", (loss_epsilon - loss_regular).shape)
        # print("d", d)
        # print("dx", dx)
        # print("check vdot", np.vdot(epsilon * d.T, dx))
        # print("dx", dx)

        difference_with_grad = np.abs(loss_epsilon - loss_regular - np.vdot(epsilon * d, dx))
        epsilons.append(epsilon)
        differences.append(difference)
        differences_with_grad.append(difference_with_grad)
        print("epsilon", epsilon)
        print("difference", difference)
        print("difference_with_grad", difference_with_grad)
        # print("difference_with_grad", np.mean(difference_with_grad))
    # print()
    # print("difference", difference)
    # print("difference_with_grad", difference_with_grad)
    # print(type(differences_with_grad))
    # Plotting
    plt.loglog(epsilons, differences, label='Without Gradient')
    plt.loglog(epsilons, differences_with_grad, label='With Gradient')
    # x_ticks = np.logspace(-20, 0, num=21, base=10)
    # plt.xticks(x_ticks, [f"{tick:.0e}" for tick in x_ticks])

    plt.xlabel('Epsilon')
    plt.ylabel('Difference')

    plt.legend()
    plt.title('Difference by Epsilon')
    plt.gca().invert_xaxis()
    plt.show()


def gradient_checkW(W, X, y, b):
    d = normalize(np.random.randn(W.shape[0], W.shape[1]))
    epsilon = 1
    epsilons = []
    differences = []
    grad_diff = []
    for i in range(1, 20):
        epsilon = (0.5 ** i)
        print("epsilon", epsilon)
        loss_epsilon = loss(W + epsilon * d, X, b, y)
        dw, dx, db = gradient(X, y, W, b)
        loss_regular = loss(W, X, b, y)
        difference = np.abs(loss_epsilon - loss_regular)
        difference_with_grad = np.abs(loss_epsilon - loss_regular - np.vdot(epsilon * d, dw))
        epsilons.append(epsilon)
        differences.append(difference)
        grad_diff.append(difference_with_grad)
    plt.loglog(epsilons, differences, label='Without Gradient')
    plt.loglog(epsilons, grad_diff, label='With Gradient')

    plt.xlabel('Epsilon')
    plt.ylabel('Difference')
    plt.title('Difference by Epsilon')
    plt.gca().invert_xaxis()
    plt.show()


def gradient_checkb(W, X, y, b):
    d = normalize(np.random.randn(b.shape[0], b.shape[1]))
    epsilon = 1
    epsilons = []
    differences = []
    grad_diff = []
    for i in range(1, 20):
        epsilon = (0.5 ** i)
        print("epsilon", epsilon)
        loss_epsilon = loss(W, X, b + epsilon * d, y)
        dw, dx, db = gradient(X, y, W, b)
        loss_regular = loss(W, X, b, y)
        difference = np.abs(loss_epsilon - loss_regular)
        difference_with_grad = np.abs(loss_epsilon - loss_regular - np.vdot(epsilon * d, db))
        epsilons.append(epsilon)
        differences.append(difference)
        grad_diff.append(difference_with_grad)
    plt.loglog(epsilons, differences, label='Without Gradient')
    plt.loglog(epsilons, grad_diff, label='With Gradient')

    plt.xlabel('Epsilon')
    plt.ylabel('Difference')
    plt.title('Difference by Epsilon')
    plt.gca().invert_xaxis()
    plt.show()


# Example usage
# np.random.seed(42)
W = np.random.randn(5, 5)  # Example weights
X = np.random.randn(5, 3)  # Example input
b = np.random.randn(1, 5)  # Example biases
# y is supposed to be 3 5
y = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])  # Example one-hot encoded labels

gradient_checkb(W, X, y, b)

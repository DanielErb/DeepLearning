import numpy as np
import matplotlib.pyplot as plt
from ActivationFuncs import LayerFunc


def normalize(v):
    norm = np.linalg.norm(v, axis=0)
    # print("v", v)
    # print("norm", norm)
    # print("v/norm", v / norm)
    check = v / norm
    return v / norm


def jacobian_CheckX(W, X, b, check):
    d = normalize(np.random.randn(X.shape[0], X.shape[1]))
    # print("d", d)
    # print(W.shape)
    epsilon = 1
    epsilons = []
    differences = []
    differences_with_grad = []
    u = np.random.randn(W.shape[0], X.shape[1])
    for i in range(1, 20):
        epsilon = (0.5 ** i)
        dwu, dxu, dbu = check.gradient(X, u)
        activation_epsilon = np.vdot(u, check.activation(X + epsilon * d, W, b))
        activation_regular = np.vdot(u, check.activation(X))
        difference = np.abs(activation_epsilon - activation_regular)
        difference_with_grad = np.abs(activation_epsilon - activation_regular - np.vdot(dxu, epsilon * d))
        epsilons.append(epsilon)
        differences.append(difference)
        differences_with_grad.append(difference_with_grad)
        print("epsilon", epsilon)
        print("difference", difference)
        print("difference_with_grad", difference_with_grad)

    print("difference", difference)
    print("difference_with_grad", difference_with_grad)
    plt.loglog(epsilons, differences, label='Without Gradient')
    plt.loglog(epsilons, differences_with_grad, label='With Gradient')

    plt.xlabel('Epsilon')
    plt.ylabel('Difference')

    plt.legend()
    plt.title('Difference by Epsilon')
    plt.gca().invert_xaxis()
    plt.show()


def jacobian_CheckW(W, X, b, check):
    d = normalize(np.random.randn(W.shape[0], W.shape[1]))
    # print("d", d)
    # print(W.shape)
    epsilon = 1
    epsilons = []
    differences = []
    differences_with_grad = []
    u = np.random.randn(W.shape[0], X.shape[1])
    for i in range(1, 20):
        epsilon = (0.5 ** i)
        dwu, dxu, dbu = check.gradient(X, u)
        activation_epsilon = np.vdot(u, check.activation(X, W + epsilon * d, b))
        activation_regular = np.vdot(u, check.activation(X))
        difference = np.abs(activation_epsilon - activation_regular)
        difference_with_grad = np.abs(activation_epsilon - activation_regular - np.vdot(dwu, epsilon * d))
        epsilons.append(epsilon)
        differences.append(difference)
        differences_with_grad.append(difference_with_grad)
        print("epsilon", epsilon)
        print("difference", difference)
        print("difference_with_grad", difference_with_grad)

    print("difference", difference)
    print("difference_with_grad", difference_with_grad)
    plt.loglog(epsilons, differences, label='Without Gradient')
    plt.loglog(epsilons, differences_with_grad, label='With Gradient')

    plt.xlabel('Epsilon')
    plt.ylabel('Difference')

    plt.legend()
    plt.title('Difference by Epsilon')
    plt.gca().invert_xaxis()
    plt.show()


def jacobian_Checkb(W, X, b, check):
    d = normalize(np.random.randn(b.shape[0], b.shape[1]))
    # print("d", d)
    # print(W.shape)
    epsilon = 1
    epsilons = []
    differences = []
    differences_with_grad = []
    u = np.random.randn(W.shape[0], X.shape[1])
    for i in range(1, 20):
        epsilon = (0.5 ** i)
        dwu, dxu, dbu = check.gradient(X, u)
        activation_epsilon = np.vdot(u, check.activation(X, W, b + epsilon * d))
        activation_regular = np.vdot(u, check.activation(X))
        difference = np.abs(activation_epsilon - activation_regular)
        print("dbu.shape", dbu.shape)
        difference_with_grad = np.abs(activation_epsilon - activation_regular - np.vdot(dbu, epsilon * d))
        epsilons.append(epsilon)
        differences.append(difference)
        differences_with_grad.append(difference_with_grad)
        print("epsilon", epsilon)
        print("difference", difference)
        print("difference_with_grad", difference_with_grad)

    print("difference", difference)
    print("difference_with_grad", difference_with_grad)
    plt.loglog(epsilons, differences, label='Without Gradient')
    plt.loglog(epsilons, differences_with_grad, label='With Gradient')

    plt.xlabel('Epsilon')
    plt.ylabel('Difference')

    plt.legend()
    plt.title('Difference by Epsilon')
    plt.gca().invert_xaxis()
    plt.show()


def main():
    W = np.random.randn(5, 5)  # Example weights
    X = np.random.randn(5, 3)  # Example input
    b = np.random.randn(5, 1)  # Example biases
    # y is supposed to be 3 5
    y = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])  # Example one-hot encoded labels

    check = LayerFunc(W, b, "tanh")

    jacobian_Checkb(W, X, b, check)


if __name__ == '__main__':
    main()
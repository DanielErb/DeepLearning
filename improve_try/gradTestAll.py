import numpy as np
import matplotlib.pyplot as plt
from ActivationFuncs import Softmax
from improve_try.Sgd import push_forward
from improve_try.NeuralNetwork import NeuralNetwork


def normalize(v):
    norm = np.linalg.norm(v, axis=0)
    # print("v", v)
    # print("norm", norm)
    # print("v/norm", v / norm)
    check = v / norm
    return v / norm


def gradient_checkX(W, X, Y, b, check):
    d = normalize(np.random.randn(X.shape[0], X.shape[1]))
    # print("d", d)
    # print(W.shape)
    epsilon = 1
    epsilons = []
    differences = []
    differences_with_grad = []
    for i in range(1, 20):
        epsilon = (0.5 ** i)
        dw, dx, db = check.gradient(X, Y)
        # print("epsilon", epsilon)
        loss_epsilon = check.loss(X + epsilon * d, Y, W, b)
        # loss_epsilon_check = loss2(W, X + epsilon * d, y)
        loss_regular = check.loss(X, Y)
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


def gradient_checkW(W, X, Y, b, check):
    d = normalize(np.random.randn(W.shape[0], W.shape[1]))
    epsilon = 1
    epsilons = []
    differences = []
    grad_diff = []
    for i in range(1, 20):
        epsilon = (0.5 ** i)
        print("epsilon", epsilon)
        loss_epsilon = check.loss(X, Y, W + epsilon * d, b)
        dw, dx, db = check.gradient(X, Y)
        loss_regular = check.loss(X, Y)
        difference = np.abs(loss_epsilon - loss_regular)
        difference_with_grad = np.abs(loss_epsilon - loss_regular - np.vdot(epsilon * d, dw))
        print("difference", difference)
        print("difference_with_grad", difference_with_grad)
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


def gradient_checkb(W, X, Y, b, check):
    d = normalize(np.random.randn(b.shape[0], b.shape[1]))
    print("d", d)
    epsilon = 1
    epsilons = []
    differences = []
    grad_diff = []
    for i in range(1, 20):
        epsilon = (0.5 ** i)
        print("epsilon", epsilon)
        loss_epsilon = check.loss(X, Y, W, b + epsilon * d)
        dw, dx, db = check.gradient(X, Y)
        loss_regular = check.loss(X, Y)
        difference = np.abs(loss_epsilon - loss_regular)
        difference_with_grad = np.abs(loss_epsilon - loss_regular - np.vdot(epsilon * d, db))
        epsilons.append(epsilon)
        differences.append(difference)
        grad_diff.append(difference_with_grad)
        print("difference", difference)
        print("difference_with_grad", difference_with_grad)
    plt.loglog(epsilons, differences, label='Without Gradient')
    plt.loglog(epsilons, grad_diff, label='With Gradient')

    plt.xlabel('Epsilon')
    plt.ylabel('Difference')
    plt.title('Difference by Epsilon')
    plt.gca().invert_xaxis()
    plt.show()


# Example usage
# np.random.seed(42)

def main():
    neuralNetwork = NeuralNetwork(networkType="standard", epochs=1, numLayers=3)
    X = push_forward(neuralNetwork.X_train, neuralNetwork.layers[:-1])
    gradient_checkX(neuralNetwork.weights[neuralNetwork.numLayers - 1], X, neuralNetwork.Y_train,
                    neuralNetwork.biases[neuralNetwork.numLayers - 1],
                    neuralNetwork.layers[neuralNetwork.numLayers - 1])


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from ActivationFuncs import Softmax
from Sgd import sgd
import scipy.io


def main():
    mat = scipy.io.loadmat('GMMData.mat')
    Y_train = mat['Ct'].T
    #print(X_test.shape)
    X_train = mat['Yt']

    #print(X_train.shape)
    Y_test = mat['Cv'].T
    X_test = mat['Yv']

    #print("X_test.shape", X_test.shape)
    #print("Y_test.shape", Y_test.shape)

    input_size = X_train.shape[1]
    print(input_size)
    output_size = Y_train.shape[1]

    W = np.random.randn(X_train.shape[0], output_size)

    biases = np.random.randn(1, output_size)

    learning_rate = 0.1
    num_epochs = 400
    batch_size = 32

    last_layer = Softmax(W, biases)

    train_accuracy, test_accuracy, train_loss, test_loss = sgd(X_train, Y_train, X_test, Y_test, last_layer, learning_rate, num_epochs,
                           batch_size)
    # print("bi", biases)
    # print("weights.shape", weights)
   # check_loss = last_layer.loss(Y_train, Y_test)
   # print("check_loss", check_loss)

    plt.plot(range(1, num_epochs + 1), train_accuracy, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracy, label='Test Accuracy')
    plt.plot(range(1, num_epochs + 1), train_loss, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    plt.title('Success Percent in the train set per epoch')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

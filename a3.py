import numpy as np
import matplotlib.pyplot as plt
from ActivationFuncs import Softmax
from Sgd import sgd
import scipy.io

def main():
    mat = scipy.io.loadmat('GMMData.mat')
    X_test = mat['Ct'].T
    print(X_test.shape)
    X_train = mat['Yt']
    print(X_train.shape)
    Y_test = mat['Cv'].T
    Y_train = mat['Yv']

    input_size = X_train.shape[1]
    print(input_size)
    output_size = Y_train.shape[0]

    W = np.random.randn(X_train.shape[0], output_size)

    biases = np.random.randn(1, output_size)

    learning_rate = 0.1
    num_epochs = 4000
    batch_size = 4

    last_layer = Softmax(W, biases)

    losses, percents = sgd(X_train, X_test, last_layer, learning_rate, num_epochs,
                           batch_size)
    # print("bi", biases)
    # print("weights.shape", weights)
    check_loss = last_layer.loss(Y_train, Y_test)
    print("check_loss", check_loss)

    plt.plot(range(1, num_epochs + 1), percents)
    plt.plot()
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    plt.title('Success Percent in the train set per epoch')
    plt.show()

if __name__ == '__main__':
    main()
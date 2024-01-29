import scipy.io
from NeuralNetwork import NeuralNetwork


def main():
    mat = scipy.io.loadmat('PeaksData.mat')
    Y_train = mat['Ct'].T
    # print(X_test.shape)
    X_train = mat['Yt']

    # print(X_train.shape)
    Y_test = mat['Cv'].T
    X_test = mat['Yv']

    learning_rate = 0.0003
    num_epochs = 1000
    batch_size = 64
    accuracy_sample_size_train = 6500
    accuracy_sample_size_test = 2500
    numLayers = 2
    activationFuncName = 'relu'

    neuralNetwork = NeuralNetwork(numLayers, activationFuncName, learning_rate, num_epochs, batch_size, Y_train,
                                  X_train, Y_test, X_test,
                                  accuracy_sample_size_train, accuracy_sample_size_test)
    neuralNetwork.runNeuralNetwork()


if __name__ == '__main__':
    main()

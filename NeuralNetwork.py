import random

from matplotlib import pyplot as plt

from Sgd import General_sgd
from ActivationFuncs import LayerFunc
from ActivationFuncs import Softmax

import numpy as np


class NeuralNetwork:
    def __init__(self, numLayers, activationFuncName, learning_rate, epochs, batch_size, Y_train, X_train, Y_test, X_test,
               accuracy_sample_size_train, accuracy_sample_size_test):
        self.numLayers = numLayers
        self.activationFuncName = activationFuncName
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.Y_train = Y_train
        self.X_train = X_train
        self.Y_test = Y_test
        self.X_test = X_test
        self.accuracy_sample_size_train = accuracy_sample_size_train
        self.accuracy_sample_size_test = accuracy_sample_size_test
        self.weights = []
        self.biases = []
        self.layers = []
        self.buildNeuralNetwork()

    def buildNeuralNetwork(self):

        for i in range(1, self.numLayers + 1):  # runs from 1 to layers
            if i == self.numLayers:
                if self.numLayers == 1:
                    w = np.random.randn(self.X_train.shape[0], self.Y_train.shape[1])
                else:
                    w = np.random.randn(self.weights[i - 2].shape[0], self.Y_train.shape[1])
                b = np.random.randn(1, self.Y_train.shape[1])
                self.weights.append(w)
                self.biases.append(b)
                self.layers.append(Softmax(w, b))

            elif i == 1:  # at least 2 layers
                num_weights = random.randint(50, 100)
                w = np.random.randn(num_weights, self.X_train.shape[0])
                b = np.random.randn(num_weights, 1)
                self.weights.append(w)
                self.biases.append(b)
                self.layers.append(LayerFunc(w, b, self.activationFuncName))
            else:
                num_weights = random.randint(50, 100)
                w = np.random.randn(num_weights, self.weights[i - 2].shape[0]) #-2 because i starts from 1
                b = np.random.randn(num_weights, 1)
                self.weights.append(w)
                self.biases.append(b)
                self.layers.append(LayerFunc(w, b, self.activationFuncName))

        # for i in range(1, self.numLayers + 1):
        #     print("layer", i)
        #     print("weights", self.weights[i - 1].shape)
        #     print("biases", self.biases[i - 1].shape)

    def runNeuralNetwork(self):
        train_accuracy, test_accuracy, train_loss, test_loss = General_sgd(self.X_train, self.Y_train, self.X_test,
                                                                           self.Y_test,
                                                                           self.layers, self.learning_rate, self.epochs,
                                                                           self.batch_size,
                                                                           self.accuracy_sample_size_train,
                                                                           self.accuracy_sample_size_test)

        plt.plot(range(1, self.epochs + 1), train_accuracy, label='Train Accuracy')
        plt.plot(range(1, self.epochs + 1), test_accuracy, label='Test Accuracy')
        #plt.plot(range(1, self.epochs + 1), train_loss, label='Train Loss')
        #plt.plot(range(1, self.epochs + 1), test_loss, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Percent')
        plt.title('Success Percent in the train set per epoch')
        plt.legend()
        plt.show()

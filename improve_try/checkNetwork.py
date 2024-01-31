import MatData  # if we want to use another data set, we call MatData.fetch('data_set_name')
from NeuralNetwork import NeuralNetwork


def main():

    neuralNetwork = NeuralNetwork(networkType="ResNet" , epochs=50)
    neuralNetwork.runNeuralNetwork()


if __name__ == '__main__':
    main()

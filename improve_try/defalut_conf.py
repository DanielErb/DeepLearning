import MatData
import numpy as np

Y_train, X_train, Y_test, X_test = MatData.fetch('PeaksData.mat')

# the code for selecting random 200 data points
# num_points = X_train.shape[1]
# indexes = np.arange(num_points)
# np.random.shuffle(indexes)
# X_train = X_train[:, indexes[0:0 + 200]]
# Y_train = Y_train[indexes[0:0 + 200]]

learning_rate = 0.0003
num_epochs = 400
batch_size = 64
accuracy_sample_size_train = 6500
accuracy_sample_size_test = 2500
numLayers = 3
activationFuncName = 'relu'
NetworkType = 'standard'

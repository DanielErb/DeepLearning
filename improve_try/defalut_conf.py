import MatData

Y_train, X_train, Y_test, X_test = MatData.fetch('PeaksData.mat')

learning_rate = 0.0003
num_epochs = 1000
batch_size = 64
accuracy_sample_size_train = 6500
accuracy_sample_size_test = 2500
numLayers = 2
activationFuncName = 'relu'
NetworkType = 'standard'

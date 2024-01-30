import scipy.io

mat = scipy.io.loadmat('PeaksData.mat')

Y_train = mat['Ct'].T
X_train = mat['Yt']

Y_test = mat['Cv'].T
X_test = mat['Yv']

learning_rate = 0.0003
num_epochs = 1000
batch_size = 64
accuracy_sample_size_train = 6500
accuracy_sample_size_test = 2500
numLayers = 4
activationFuncName = 'relu'
NetworkType = 'standard'
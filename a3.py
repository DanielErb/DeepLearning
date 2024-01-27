import numpy as np
import matplotlib.pyplot as plt
import scipy
from ActivationFuncs import gradient
from ActivationFuncs import loss
from ActivationFuncs import softmax

rng = np.random.default_rng()


def sample_minibatch(X, y, batch_size):
    random_indexes = rng.choice(X.shape[1], batch_size, False)
    print("X.T[random_indexes]", (X.T[random_indexes]))
    print("X.T[random_indexes].T", (X.T[random_indexes]).T)
    print(y.shape)
    print("y[random_indexes]", (y[random_indexes]))

    return (X.T[random_indexes]).T, (y[random_indexes])  # this is because X is of shape 100,2 and y is of shape 100,2
    # so in order to select some random rows from X and y we need to transpose X and then select the rows and then transpose it back


def sgd(X, y, weights, biases, learning_rate, epochs, batch_size):
    num_samples = len(y)
    losses = []
    graidentWeights = []
    percents = []
    for epoch in range(epochs):
        # Shuffle the data at the beginning of each epoch
        print("epoch", epoch)
        X_batch, y_batch = sample_minibatch(X, y, batch_size)
        print("X_batch.shape", X_batch.shape)
        print("x")
        gradient_weights, gradient_X, gradient_biases = gradient(X_batch, y_batch, weights, biases)
        weights -= learning_rate * gradient_weights
        biases -= learning_rate * gradient_biases

        percents.append(calcpercents(y, softmax(weights, X,biases)))

        # Calculate and print the mean loss after each epoch
        Loss = loss(weights, X, biases,y)
        losses.append(Loss)
        graidentWeights.append(gradient_weights)
        print(f"Epoch {epoch + 1}/{epochs}, Mean Loss: {Loss}")

    return weights, biases, losses, graidentWeights, percents


def calcpercents(y, y_hat):
    denominator = y_hat.shape[0]
    # Find the indices of the maximum values in each row of y_hat
    max_indices_y_hat = np.argmax(y_hat, axis=1)

    # Find the indices where the value is 1 in each row of y
    indices_y = np.argmax(y, axis=1)

    # Count the number of matching rows
    Numerator = np.sum(max_indices_y_hat == indices_y)

    return Numerator / denominator



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
num_epochs = 100
batch_size = 4

weights, biases, losses, graidentWeights, percents = sgd(X_train, X_test, W, biases, learning_rate, num_epochs,
                                                         batch_size)
#print("bi", biases)
#print("weights.shape", weights)
check_loss = loss(weights, Y_train, biases, Y_test)
print("check_loss", check_loss)

plt.plot(range(1, num_epochs + 1), percents)
plt.plot()
plt.xlabel('Epoch')
plt.ylabel('Percent')
plt.title('Success Percent in the train set per epoch')
plt.show()

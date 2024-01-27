import numpy as np
import matplotlib.pyplot as plt
import scipy

rng = np.random.default_rng()

def sample_minibatch(X, y, batch_size):
    random_indexes = rng.choice(X.shape[1], batch_size, False)
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

        gradient_weights, gradient_X, gradient_biases = gradient(X_batch, y_batch, weights, biases)
        weights -= learning_rate * gradient_weights
        biases -= learning_rate * gradient_biases

        percents.append(calcpercents(y , softmax(weights , X)))

        # Calculate and print the mean loss after each epoch
        Loss = loss(weights, X, y)
        losses.append(Loss)
        graidentWeights.append(gradient_weights)
        print(f"Epoch {epoch + 1}/{epochs}, Mean Loss: {Loss}")

    return weights, biases, losses, graidentWeights , percents


def calcpercents(y ,y_hat):
    denominator = y_hat.shape[0]
    # Find the indices of the maximum values in each row of y_hat
    max_indices_y_hat = np.argmax(y_hat, axis=1)

    # Find the indices where the value is 1 in each row of y
    indices_y = np.argmax(y, axis=1)

    # Count the number of matching rows
    Numerator = np.sum(max_indices_y_hat == indices_y)

    return Numerator / denominator


def softmax(W, X):
    temp = np.dot(X.T, W)
    sum = np.sum(np.exp(temp), axis=1)
    print("temp", np.exp(temp))
    print("sum", np.sum(np.exp(temp), axis=1))
    print("total", np.exp(temp) / np.sum(np.exp(temp), axis=1, keepdims=True))
    return np.exp(temp) / np.sum(np.exp(temp), axis=1, keepdims=True)


def cross_entropy_loss(y, y_hat):
    print("check", -(y * np.log(y_hat)))
    # print("check_y_hat", y_hat)
    # print("check_y", y)
    # print("check_y_hat", y_hat.shape)
    # print("check_y", y.shape)
    # print("np,log", np.log(y_hat))
    return -(y * np.log(y_hat))


# def loss(A, y):
#      y_hat = softmax(A)
#      return np.mean(cross_entropy_loss(y, y_hat))

def loss(W, X, y):
    y_hat = softmax(W, X)
    cross_entropy = cross_entropy_loss(y, y_hat)
    #sum = np.sum(np.sum(cross_entropy, axis=1), axis=0)
    print("sum", sum)
    print("loss", np.mean(np.sum(cross_entropy_loss(y, y_hat), axis=1)))
    return np.mean(np.sum(cross_entropy_loss(y, y_hat), axis=1))


def gradient(X_batch, y_batch, W, B):
    # temp = np.dot(X.T, W)
    # after_softmax =  np.exp(temp). / sum(np.exp(temp), axis=0)
    after_softmax = softmax(W, X_batch)
    print("after_softmax.shape", after_softmax.shape)
    print("y_batch.shape", y_batch.shape)
    print()
    dw = np.dot(X_batch, (after_softmax - y_batch)) / X_batch.shape[1]
    dx = np.dot(W, (after_softmax - y_batch).T) / X_batch.shape[1]
    print("dx.shape", dx.shape)
    db = np.sum((after_softmax - y_batch), axis=0, keepdims=True) / X_batch.shape[1]
    print("db.shape", db.shape)
    # grad = np.concatenate((dw, db), axis=0)
    return dw, dx, db

mat = scipy.io.loadmat('GMMData.mat')
X_test = mat['Cv'].T
X_train = mat['Ct'].T
Y_test = mat['Yv']
Y_train = mat['Yt']

input_size = X_train.shape[0]
print(input_size)
output_size = Y_train.shape[1]

W = np.random.randn(input_size , output_size)

biases = np.random.randn(1,output_size)

learning_rate = 0.1
num_epochs = 100
batch_size = 4

weights, biases, losses, graidentWeights, percents = sgd(X_train , Y_train , W , biases , learning_rate , num_epochs , batch_size)



plt.plot(range(1, num_epochs + 1), percents)
plt.plot()
plt.xlabel('Epoch')
plt.ylabel('Percent')
plt.title('Success Percent in the train set per epoch')
plt.show()

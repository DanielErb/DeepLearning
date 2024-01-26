import numpy as np
import matplotlib.pyplot as plt
import scipy

rng = np.random.default_rng()


def least_squares_loss(X_batch, y_batch, weights, biases):
    predictions = np.dot(X_batch.T, weights) + biases
    losses = (predictions - y_batch) ** 2 / 2
    return np.mean(losses)


def least_squares_gradient(X_batch, y_batch, weights, biases):
    predictions = np.dot(X_batch.T, weights) + biases

    # Calculating the gradient
    gradient_weights = (X_batch @ (predictions - y_batch)) / len(y_batch)
    gradient_biases = np.sum(predictions - y_batch) / len(y_batch)
    return gradient_weights, gradient_biases


def sample_minibatch(X, y, batch_size):
    random_indexes = rng.choice(X_data.shape[1], batch_size, False)
    return (X.T[random_indexes]).T, (y[random_indexes])  # this is because X is of shape 100,2 and y is of shape 100,2
    # so in order to select some random rows from X and y we need to transpose X and then select the rows and then transpose it back


def sgd(X, y, weights, biases, learning_rate, epochs, batch_size):
    num_samples = len(y)
    losses = []
    graidentWeights = []

    for epoch in range(epochs):
        # Shuffle the data at the beginning of each epoch
        print("epoch", epoch)
        X_batch, y_batch = sample_minibatch(X, y, batch_size)

        gradient_weights, gradient_biases = least_squares_gradient(X_batch, y_batch, weights, biases)
        weights -= learning_rate * gradient_weights
        biases -= learning_rate * gradient_biases

        # Calculate and print the mean loss after each epoch
        loss = least_squares_loss(X, y, weights, biases)
        losses.append(loss)
        graidentWeights.append(gradient_weights)
        print(f"Epoch {epoch + 1}/{epochs}, Mean Loss: {loss}")

    return weights, biases, losses, graidentWeights


# Example usage:
# Assuming X_data, y_data, initial_weights, and initial_biases are your actual data and initial parameters
# mat_data = scipy.io.loadmat('PeaksData.mat')

# Extract input (X_data) and target (y_data) from the loaded data
X_data = np.random.randn(100, 2)  # 100 vectors of 2 dimensions
X_data = X_data.T  #
print(X_data)
y_data = X_data.T  # need to see whats the problem with this and why we dont transform it

print(X_data.shape)
# Rest of your code...

# For testing
# X_test = mat_data['Yv']
# X_test = X_test.T
# y_test = X_test
input_size = X_data.shape[0]
print(input_size)
output_size = X_data.shape[0]  # Assuming a regression problem with a single output
initial_weights = np.random.randn(input_size, output_size)  # Random initialization for weights
print(initial_weights.shape)
initial_biases = np.random.randn(1, output_size)  # Initial biases

learning_rate = 0.01
num_epochs = 4000
batch_size = 32

X_test = np.random.randn(100, 2)
y_test = X_test
X_test = X_test.T

optimized_weights, optimized_biases, losses, gradientWeights = sgd(X_data, y_data, initial_weights,
                                                                   initial_biases, learning_rate, num_epochs,
                                                                   batch_size)
test_loss = least_squares_loss(X_test, y_test, optimized_weights, optimized_biases)
print(f"Test Loss: {test_loss}")

# Plotting the optimization process
plt.plot(range(1, num_epochs + 1), losses)
plt.plot()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SGD Optimization for Least Squares')
plt.show()

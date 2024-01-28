import numpy as np

rng = np.random.default_rng()


def sample_minibatch(X, y, batch_size):
    random_indexes = rng.choice(X.shape[1], batch_size, False)
    print("X.T[random_indexes]", (X.T[random_indexes]))
    print("X.T[random_indexes].T", (X.T[random_indexes]).T)
    print(y.shape)
    print("y[random_indexes]", (y[random_indexes]))

    return (X.T[random_indexes]).T, (y[random_indexes])  # this is because X is of shape 100,2 and y is of shape 100,2
    # so in order to select some random rows from X and y we need to transpose X and then select the rows and then transpose it back


def sgd(X, y, layer, learning_rate, epochs, batch_size):
    num_samples = len(y)
    losses = []
    percents = []
    for epoch in range(epochs):
        # Shuffle the data at the beginning of each epoch
        print("epoch", epoch)
        X_batch, y_batch = sample_minibatch(X, y, batch_size)
        dw, dx, db = layer.gradient(X_batch, y_batch)
        layer.W -= learning_rate * dw
        layer.b -= learning_rate * db

        percents.append(calcpercents(y, layer.activation(X)))

        # Calculate and print the mean loss after each epoch
        Loss = layer.loss(X_batch, y_batch)
        losses.append(Loss)
        print(f"Epoch {epoch + 1}/{epochs}, Mean Loss: {Loss}")

    # plt.plot(range(1, epochs + 1), losses)
    # plt.show()

    return losses, percents


def calcpercents(y, y_hat):
    denominator = y_hat.shape[0]
    # Find the indices of the maximum values in each row of y_hat
    max_indices_y_hat = np.argmax(y_hat, axis=1)

    # Find the indices where the value is 1 in each row of y
    indices_y = np.argmax(y, axis=1)

    # Count the number of matching rows
    Numerator = np.sum(max_indices_y_hat == indices_y)

    return Numerator / denominator

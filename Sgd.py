import numpy as np

rng = np.random.default_rng()


def sample_minibatch(X, y, batch_size):
    random_indexes = rng.choice(X.shape[1], batch_size, False)
   # print("X.T[random_indexes]", (X.T[random_indexes]))
    #print("X.T[random_indexes].T", (X.T[random_indexes]).T)
    #print(y.shape)
    #print("y[random_indexes]", (y[random_indexes]))

    return (X.T[random_indexes]).T, (y[random_indexes])  # this is because X is of shape 100,2 and y is of shape 100,2
    # so in order to select some random rows from X and y we need to transpose X and then select the rows and then transpose it back


def sgd(X, y, X_test, y_test, layer, learning_rate, epochs, batch_size, accuracy_sample_size_train, accuracy_sample_size_test):
    #num_samples = len(y)
    train_loss = []
    test_loss = []
    accuracy_train = []
    accuracy_test = []
    num_points = X.shape[1]
    for epoch in range(epochs):
        # Shuffle the data at the beginning of each epoch
        #epoch_train_loss = []
        #epoch_test_loss = []
        #epoch_accuracy_train = []
        #epoch_accuracy_test = []
        indexes = np.arange(num_points)
        np.random.shuffle(indexes)
        print("epoch", epoch)
        for i in range(0, num_points, batch_size):
            X_batch = X[:, indexes[i:i + batch_size]]
            y_batch = y[indexes[i:i + batch_size]]
            dw, dx, db = layer.gradient(X_batch, y_batch)
            layer.W -= learning_rate * dw
            layer.b -= learning_rate * db

            #epoch_train_loss.append(layer.loss(X_batch, y_batch))
            #epoch_accuracy_train.append(calcpercents(y_batch, layer.activation(X_batch)))
        X_train_sample, y_train_sample = sample_minibatch(X, y, accuracy_sample_size_train)
        X_test_sample, y_test_sample = sample_minibatch(X_test, y_test, accuracy_sample_size_test)
        train_loss.append(layer.loss(X_train_sample, y_train_sample))
        test_loss.append(layer.loss(X_test_sample, y_test_sample))
        accuracy_train.append(calcpercents(y_train_sample, layer.activation(X_train_sample)))
        accuracy_test.append(calcpercents(y_test_sample, layer.activation(X_test_sample)))
        #train_loss.append(np.mean(epoch_train_loss))
        #test_loss.append(layer.loss(X_test, y_test))
        #accuracy_train.append(np.mean(epoch_accuracy_train))
        #accuracy_test.append(calcpercents(y_test, layer.activation(X_test)))
        #
        # print("epoch", epoch)
        # X_batch, y_batch = sample_minibatch(X, y, batch_size)
        # X_batch_test, y_batch_test = sample_minibatch(X_test, y_test, batch_size)
        # dw, dx, db = layer.gradient(X_batch, y_batch)
        # layer.W -= learning_rate * dw
        # layer.b -= learning_rate * db
        #
        # accuracy_train.append(calcpercents(y, layer.activation(X)))
        # accuracy_test.append(calcpercents(y_test, layer.activation(X_test)))
        #
        # # Calculate and print the mean loss after each epoch
        # #Loss = layer.loss(X_batch, y_batch)
        # test_loss.append(layer.loss(X_batch_test, y_batch_test))
        # train_loss.append(layer.loss(X_batch, y_batch))

    # plt.plot(range(1, epochs + 1), losses)
    # plt.show()

    return accuracy_train, accuracy_test, train_loss, test_loss

# def sgd(X, y, X_test, y_test, layers, learning_rate, epochs, batch_size, accuracy_sample_size_train, accuracy_sample_size_test):
#     #num_samples = len(y)
#     train_loss = []
#     test_loss = []
#     accuracy_train = []
#     accuracy_test = []
#     num_points = X.shape[1]
#     for epoch in range(epochs):
#         # Shuffle the data at the beginning of each epoch
#         #epoch_train_loss = []
#         #epoch_test_loss = []
#         #epoch_accuracy_train = []
#         #epoch_accuracy_test = []
#         indexes = np.arange(num_points)
#         np.random.shuffle(indexes)
#         print("epoch", epoch)
#         for i in range(0, num_points, batch_size):
#             X_batch = X[:, indexes[i:i + batch_size]]
#             y_batch = y[indexes[i:i + batch_size]]
#             for layer in layers:
#                 dw, dx, db = layer.gradient(X_batch, y_batch)
#                 layer.W -= learning_rate * dw
#                 layer.b -= learning_rate * db
#
#             #epoch_train_loss.append(layer.loss(X_batch, y_batch))
#             #epoch_accuracy_train.append(calcpercents(y_batch, layer.activation(X_batch)))
#         X_train_sample, y_train_sample = sample_minibatch(X, y, accuracy_sample_size_train)
#         X_test_sample, y_test_sample = sample_minibatch(X_test, y_test, accuracy_sample_size_test)
#         train_loss.append(layer.loss(X_train_sample, y_train_sample))
#         test_loss.append(layer.loss(X_test_sample, y_test_sample))
#         accuracy_train.append(calcpercents(y_train_sample, layer.activation(X_train_sample)))
#         accuracy_test.append(calcpercents(y_test_sample, layer.activation(X_test_sample)))
#         #train_loss.append(np.mean(epoch_train_loss))
#         #test_loss.append(layer.loss(X_test, y_test))
#         #accuracy_train.append(np.mean(epoch_accuracy_train))
#         #accuracy_test.append(calcpercents(y_test, layer.activation(X_test)))
#         #
#         # print("epoch", epoch)
#         # X_batch, y_batch = sample_minibatch(X, y, batch_size)
#         # X_batch_test, y_batch_test = sample_minibatch(X_test, y_test, batch_size)
#         # dw, dx, db = layer.gradient(X_batch, y_batch)
#         # layer.W -= learning_rate * dw
#         # layer.b -= learning_rate * db
#         #
#         # accuracy_train.append(calcpercents(y, layer.activation(X)))
#         # accuracy_test.append(calcpercents(y_test, layer.activation(X_test)))
#         #
#         # # Calculate and print the mean loss after each epoch
#         # #Loss = layer.loss(X_batch, y_batch)
#         # test_loss.append(layer.loss(X_batch_test, y_batch_test))
#         # train_loss.append(layer.loss(X_batch, y_batch))
#
#     # plt.plot(range(1, epochs + 1), losses)
#     # plt.show()
#
#     return accuracy_train, accuracy_test, train_loss, test_loss
def calcpercents(y, y_hat):
    denominator = y_hat.shape[0]
    # Find the indices of the maximum values in each row of y_hat
    max_indices_y_hat = np.argmax(y_hat, axis=1)

    # Find the indices where the value is 1 in each row of y
    indices_y = np.argmax(y, axis=1)

    # Count the number of matching rows
    Numerator = np.sum(max_indices_y_hat == indices_y)

    return Numerator / denominator



def General_sgd(X, y, X_test, y_test, layers, learning_rate, epochs, batch_size, accuracy_sample_size_train, accuracy_sample_size_test):
    #num_samples = len(y)
    train_loss = []
    test_loss = []
    accuracy_train = []
    accuracy_test = []
    num_points = X.shape[1]
    last_layer = layers[-1]
    first_Layers = layers[:-1]
    for epoch in range(epochs):
        # Shuffle the data at the beginning of each epoch
        #epoch_train_loss = []
        #epoch_test_loss = []
        #epoch_accuracy_train = []
        #epoch_accuracy_test = []
        indexes = np.arange(num_points)
        np.random.shuffle(indexes)
        print("epoch", epoch)
        for i in range(0, num_points, batch_size):
            X_batch = X[:, indexes[i:i + batch_size]]
            y_batch = y[indexes[i:i + batch_size]]
            #dealing with last layer (with softmax):
            dw, dx, db = last_layer.gradient(X_batch, y_batch)
            last_layer.W -= learning_rate * dw
            last_layer.b -= learning_rate * db

            #dealing with the rest of the layers:
            dx_holder = dx
            for layer in first_Layers[::-1]:
                dw, dx, db = layer.gradient(X_batch, dx_holder)
                layer.W -= learning_rate * dw
                layer.b -= learning_rate * db
                dx_holder = dx


            #epoch_train_loss.append(layer.loss(X_batch, y_batch))
            #epoch_accuracy_train.append(calcpercents(y_batch, layer.activation(X_batch)))
        X_train_sample, y_train_sample = sample_minibatch(X, y, accuracy_sample_size_train)
        X_test_sample, y_test_sample = sample_minibatch(X_test, y_test, accuracy_sample_size_test)
        train_loss.append(layer.loss(X_train_sample, y_train_sample))
        test_loss.append(layer.loss(X_test_sample, y_test_sample))
        accuracy_train.append(calcpercents(y_train_sample, layer.activation(X_train_sample)))
        accuracy_test.append(calcpercents(y_test_sample, layer.activation(X_test_sample)))
        #train_loss.append(np.mean(epoch_train_loss))
        #test_loss.append(layer.loss(X_test, y_test))
        #accuracy_train.append(np.mean(epoch_accuracy_train))
        #accuracy_test.append(calcpercents(y_test, layer.activation(X_test)))
        #
        # print("epoch", epoch)
        # X_batch, y_batch = sample_minibatch(X, y, batch_size)
        # X_batch_test, y_batch_test = sample_minibatch(X_test, y_test, batch_size)
        # dw, dx, db = layer.gradient(X_batch, y_batch)
        # layer.W -= learning_rate * dw
        # layer.b -= learning_rate * db
        #
        # accuracy_train.append(calcpercents(y, layer.activation(X)))
        # accuracy_test.append(calcpercents(y_test, layer.activation(X_test)))
        #
        # # Calculate and print the mean loss after each epoch
        # #Loss = layer.loss(X_batch, y_batch)
        # test_loss.append(layer.loss(X_batch_test, y_batch_test))
        # train_loss.append(layer.loss(X_batch, y_batch))

    # plt.plot(range(1, epochs + 1), losses)
    # plt.show()

    return accuracy_train, accuracy_test, train_loss, test_loss

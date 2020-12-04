import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Conv2D, Flatten

def test_1():
    np.random.seed(123)
    X = np.load('/Users/andrew/Desktop/chomp_proj/features.npy')
    Y = np.load('/Users/andrew/Desktop/chomp_proj/labels.npy')
    samples = X.shape[0]
    board_size = 5 * 5
    X = X.reshape(samples, board_size)
    Y = Y.reshape(samples, board_size)
    train_samples = int(.9 * samples)
    X_train, X_test = X[:train_samples], X[train_samples:]
    Y_train, Y_test = Y[:train_samples], Y[train_samples:]

    model = Sequential()
    model.add(Dense(1000, activation = 'sigmoid', input_shape = (board_size,)))
    model.add(Dense(500, activation = 'sigmoid'))
    model.add(Dense(board_size, activation = 'sigmoid'))
    model.summary()

    model.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = ['accuracy'])

    model.fit(X_train, Y_train, batch_size = 64, epochs = 15, verbose = 1, validation_data = (X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose = 0)

    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

def test_2():
    np.random.seed(123)
    X = np.load('/Users/andrew/Desktop/chomp_proj/features.npy')
    Y = np.load('/Users/andrew/Desktop/chomp_proj/labels.npy')
    samples = X.shape[0]
    size = 5
    input_shape = (size, size, 1)
    X = X.reshape(samples, size, size, 1)
    train_samples = int(.9 * samples)
    X_train, X_test = X[:train_samples], X[train_samples:]
    Y_train, Y_test = Y[:train_samples], Y[train_samples:]

    model = Sequential()
    model.add(Conv2D(filters = 48, kernel_size = (3, 3), activation = 'sigmoid', padding = 'same',
                     input_shape = input_shape))
    model.add(Conv2D(48, (3, 3), activation = 'sigmoid', padding = 'same'))

if __name__ == '__main__':
    #test_1()
    test_2()
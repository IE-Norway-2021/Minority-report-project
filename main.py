# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D
from keras.utils import np_utils


def cnn():
    # Flattening the images from the 28x28 pixels to 1D 787 pixels
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    print("Shape after one-hot encoding: ", Y_train.shape)

    # building a linear stack of layers with the sequential model
    model = Sequential()
    # hidden layer
    model.add(Dense(100, input_shape=(784,), activation='relu'))
    # output layer
    model.add(Dense(10, activation='softmax'))

    # looking at the model summary
    model.summary()
    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # training the model for 10 epochs
    model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cnn()

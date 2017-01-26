import os

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Reshape, BatchNormalization
from keras.regularizers import l2

from lib import remove_history


def get_convnet_model():
    input_shape = (34, 34, 1)
    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))

    assert model.output_shape == (None, 34, 34, 1)

    nb_conv = 3

    model.add(Convolution2D(32, nb_conv, nb_conv, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    assert model.output_shape == (None, 17, 17, 32)

    model.add(Convolution2D(32, nb_conv, nb_conv, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    assert model.output_shape == (None, 9, 9, 32)

    model.add(Convolution2D(32, nb_conv, nb_conv, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    assert model.output_shape == (None, 5, 5, 32)

    model.add(Convolution2D(10, nb_conv, nb_conv, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    assert model.output_shape == (None, 3, 3, 10)

    model.add(Flatten())
    assert model.output_shape == (None, 90)

    model.add(Dense(10))
    model.add(Activation('softmax'))
    assert model.output_shape == (None, 10)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    return model


def main():
    model = get_convnet_model()
    model.save('my_model.h5')
    remove_history()

if __name__ == '__main__':
    main()

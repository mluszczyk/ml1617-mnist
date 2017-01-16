from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Reshape, BatchNormalization


def get_convnet_model():
    input_shape = (40, 40, 1)
    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))

    assert model.output_shape == (None, 40, 40, 1)

    nb_conv = 3

    model.add(Convolution2D(16, nb_conv, nb_conv, border_mode='same', subsample=(2, 2)))
    assert model.output_shape == (None, 20, 20, 16)

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    assert model.output_shape == (None, 20, 20, 16)

    model.add(Convolution2D(16, nb_conv, nb_conv, border_mode='same', subsample=(2, 2)))
    assert model.output_shape == (None, 10, 10, 16)

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    assert model.output_shape == (None, 10, 10, 16)

    model.add(Convolution2D(10, nb_conv, nb_conv, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    assert model.output_shape == (None, 5, 5, 10)

    model.add(AveragePooling2D(pool_size=(5, 5)))
    assert model.output_shape == (None, 1, 1, 10)

    model.add(Flatten())
    model.add(Activation('softmax'))
    assert model.output_shape == (None, 10)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    return model


def main():
    model = get_convnet_model()
    model.save('my_model.h5')


if __name__ == '__main__':
    main()

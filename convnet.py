from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Reshape, BatchNormalization


def get_convnet_model():
    input_shape = (40, 40, 1)
    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))
    print(model.output_shape)

    nb_conv = 3

    # model.add(Dropout(0.2))

    model.add(Convolution2D(16, nb_conv, nb_conv, border_mode='same', subsample=(2, 2)))
    print(model.output_shape)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    print(model.output_shape)

    # model.add(Dropout(0.5))

    model.add(Convolution2D(16, nb_conv, nb_conv, border_mode='same', subsample=(2, 2)))
    print(model.output_shape)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    print(model.output_shape)

    model.add(AveragePooling2D(pool_size=(5, 5)))

    # model.add(Dropout(0.5))

    model.add(Convolution2D(10, nb_conv, nb_conv, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    print(model.output_shape)

    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Activation('softmax'))
    print(model.output_shape)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

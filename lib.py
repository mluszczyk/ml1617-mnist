import json
import os
from collections import namedtuple

import numpy
import tensorflow
from keras import backend
from keras.metrics import categorical_accuracy
from keras.models import load_model
from sklearn.model_selection import train_test_split


from loadmat import my_loadmat


Batch = namedtuple("Batch", ["image", "label_one_of_n"])


def remove_history():
    try:
        os.remove("my_hist.json")
    except OSError:
        pass


def load_batch(num):
    path = "data/training_and_validation_batches/{}.mat".format(num)
    raw = my_loadmat(path)['affNISTdata']
    image = raw['image'].transpose().reshape(-1, 40, 40)
    label = raw['label_one_of_n'].transpose()
    return Batch(image, label)


def train_convnet(model, d, nb_epoch, prev_epochs):
    train_data_X, test_data_X, train_data_y, test_data_y = d

    print("train convnet")
    print(len(train_data_X), train_data_y[0], train_data_y[-1])
    return model.fit(
        train_data_X, train_data_y, nb_epoch=nb_epoch, validation_data=(test_data_X, test_data_y),
        initial_epoch=prev_epochs)


def combine_history(old_hist, history):
    return {
        key: old_hist[key] + history.get(key, []) for key in old_hist.keys()
    }


def load_data(batch_start, batch_end):
    batches = []
    for batch_num in range(batch_start, batch_end):
        b = load_batch(batch_num)
        tup = (
            b.image.reshape(-1, 40, 40, 1),
            b.label_one_of_n
        )
        batches.append(tup)
    return batches


def train_test_batch_split(batches, test_size, random_state):
    split_by_batch = ([], [], [], [])
    for batch in batches:
        b_split = train_test_split(
            batch[0], batch[1], test_size=test_size, random_state=random_state)
        for l, b in zip(split_by_batch, b_split):
            l.append(b)
    final = tuple(numpy.concatenate(b) for b in split_by_batch)
    return final


def perform_training(batch_start, batch_end, nb_epoch):
    batches = load_data(batch_start, batch_end)
    d = train_test_batch_split(batches, 0.1, 42)

    model = load_model("my_model.h5")
    try:
        with open("my_hist.json", "r") as f:
            old_hist = json.load(f)
    except OSError:
        old_hist = {"val_categorical_accuracy": [], "categorical_accuracy": [], "loss": [], "val_loss": []}
    prev_epochs = len(old_hist['loss'])
    hist = train_convnet(model, d, nb_epoch, prev_epochs=prev_epochs)
    history = combine_history(old_hist, hist.history)
    with open("my_hist.json", "w") as f:
        json.dump(history, f)
    model.save('my_model.h5')

    train_x, test_x, train_y, test_y = d

    pred_y = model.predict(train_x)
    acc_computation = categorical_accuracy(
        tensorflow.convert_to_tensor(train_y), tensorflow.convert_to_tensor(pred_y))
    acc = acc_computation.eval(session=backend.get_session())
    print("train acc", acc)

    pred_y = model.predict(test_x)
    acc_computation = categorical_accuracy(
        tensorflow.convert_to_tensor(test_y), tensorflow.convert_to_tensor(pred_y))
    acc = acc_computation.eval(session=backend.get_session())
    print("test acc", acc)

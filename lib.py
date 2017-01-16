from collections import namedtuple

import tensorflow
from keras.metrics import categorical_accuracy
from sklearn.model_selection import train_test_split


from convnet import get_convnet_model
from loadmat import my_loadmat


Batch = namedtuple("Batch", ["image", "label_one_of_n"])


def load_batch(num):
    path = "data/training_and_validation_batches/{}.mat".format(num)
    raw = my_loadmat(path)['affNISTdata']
    image = raw['image'].transpose().reshape(-1, 40, 40)
    label = raw['label_one_of_n'].transpose()
    return Batch(image, label)


def train_convnet(model, d, nb_epoch):
    train_data_X, test_data_X, train_data_y, test_data_y = d

    print("train convnet")
    print(len(train_data_X), train_data_y[0], train_data_y[-1])
    model.fit(train_data_X, train_data_y, nb_epoch=nb_epoch)


def main():
    b = load_batch(1)
    image = b.image.reshape(-1, 40, 40, 1)
    d = train_test_split(image, b.label_one_of_n, test_size=0.1, random_state=42)
    model = get_convnet_model()
    train_convnet(model, d, 3)

    test_x, _, test_y, _ = d
    test_x, test_y = test_x, test_y
    pred_y = model.predict(test_x)

    from keras import backend
    acc_computation = categorical_accuracy(
        tensorflow.convert_to_tensor(test_y), tensorflow.convert_to_tensor(pred_y))
    acc = acc_computation.eval(session=backend.get_session())
    print(acc)

    print(test_y[3])
    print(pred_y[3])


if __name__ == '__main__':
    main()
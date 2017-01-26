import numpy
from keras import models
from pandas import DataFrame


def main():
    model = models.load_model("my_model.h5")
    labels = numpy.load("test-preprocessed-labels.npy")
    images = numpy.load("test-preprocessed-images.npy")
    output = model.predict_classes(images)

    df = DataFrame({"Id": labels, "Label": output})
    df.to_csv("data/submission.csv", index=False)


if __name__ == '__main__':
    main()
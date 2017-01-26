import numpy
from numpy.testing import assert_array_equal

from lib import center


def main():
    m = numpy.load("data/test.npy")
    labels = m[:, 0]
    images = m[:, 1:].reshape(-1, 40, 40)
    images = center(images)
    assert_array_equal(images.shape[1:], (34, 34))
    images = images.reshape(-1, 34, 34, 1)
    numpy.save("test-preprocessed-labels.npy", labels)
    numpy.save("test-preprocessed-images.npy", images)


if __name__ == '__main__':
    main()

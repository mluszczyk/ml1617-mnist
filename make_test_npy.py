import numpy
import pandas


def main():
    m = pandas.read_csv("data/test.csv").values
    numpy.save("data/test.npy", m)


if __name__ == '__main__':
    main()

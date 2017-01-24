import sys
from lib import perform_training

if __name__ == '__main__':
    perform_training(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

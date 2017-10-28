import argparse
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import numpy as np
from numpy import array, zeros

kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):

        import pickle
        import gzip

        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()

        labelsIdx = np.where((self.train_y == 3) | (self.train_y == 8))[0]
        self.train_y = self.train_y[labelsIdx]
        self.train_x = self.train_x[labelsIdx]
        labelsIdx = np.where((self.test_y == 3) | (self.test_y == 8))[0]
        self.test_y = self.test_y[labelsIdx]
        self.test_x = self.test_x[labelsIdx]


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector.
    """

    w = zeros(len(x[0]))
    temp = np.array(np.multiply(
        np.matrix(np.multiply(alpha, y)).T, np.matrix(x)))
    for i in range(len(x)):
        w += temp[i, :]
    return w


def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a primal support vector, return the indices for all of the support
    vectors
    """

    support = set()
    dot = [np.dot(x[i, :], w) + b for i in range(len(x))]
    prediction = np.multiply(dot, y)
    for i in range(len(prediction)):
        if prediction[i] > 1 - tolerance and prediction[i] < 1 + tolerance:
            support.add(i)
    return support


def find_slack(x, y, w, b):
    """
    Given a primal support vector instance, return the indices for all of the
    slack vectors
    """

    slack = set()
    dot = [np.dot(x[i, :], w) + b for i in range(len(x))]
    prediction = np.multiply(dot, y)
    for i in range(len(prediction)):
        if np.sign(prediction[i]) < 1.0:
            slack.add(i)
    return slack


if __name__ == "__main__":

    data = Numbers("../kNearestNeighbors/data/mnist.pkl.gz")
    print("Data loaded.")
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--limit', type=int, default=0,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    i = 1
    accuracy = list()
    if args.limit > 0:
        for kernel in ("poly", "linear"):
            for itr in range(1, 26):
                classifier = SVC(kernel=kernel, C=itr)
                classifier.fit(data.train_x[:args.limit],
                               data.train_y[:args.limit])

                predicted = classifier.predict(data.test_x)
                acc = accuracy_score(data.test_y, predicted)
                accuracy.append(acc)
                print("Accuracy of the classifier with",
                      kernel, "kernel and regularization paramter", itr, "is:", acc, ".")

        fig, ax = plt.subplots()
        ax.plot(np.arange(1, 26), accuracy[:25], 'ro', label='RBF kernel')
        ax.plot(np.arange(1, 26), accuracy[-25:], 'bo', label='Linear kernel')
        legend = ax.legend(loc='center right')
        plt.xlabel("Regularization Paramter(C)")
        plt.ylabel("Accuracy score")
        plt.title("Accuracy vs Regularization plot")
        plt.show()

        for itr in range(1, 5):
            plt.subplot(int("24" + str(itr)))
            plt.imshow(np.reshape(classifier.support_vectors_[
                itr], (28, 28)), cmap="gray_r")
            plt.subplot(int("24" + str(itr + 4)))
            plt.imshow(np.reshape(classifier.support_vectors_[
                -itr], (28, 28)), cmap="gray_r")

        plt.show()

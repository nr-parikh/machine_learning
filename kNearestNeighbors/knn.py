import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree
from itertools import chain

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):

        #Open the file and load the data as soon as the object of the
        #class is created

        import pickle, gzip

        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = pickle.load(f) 

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()


class Knearest:
    """
    kNN classifier class
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median value (as implemented in numpy).

        :param item_indices: The indices of the k nearest neighbors to the 
        queried data point 
        """
        assert len(item_indices) == self._k, "Did not get k inputs"

        #Use the counter object to count the frequency of the data 
        counter = Counter(item_indices) 

        #Find the indices with the highest frequency
        indices = [(key, val) for key,val in counter.items() if val == max(counter.values())]

        #If there is tie in the frequency take the median value of the 
        #labels of the tied indices
        if len(indices)>1:
            indices = [[key]* val for (key, val) in indices]
            indices = sum(indices, [])

            #Check if the number of indices is odd or even 
            if len(indices)%2==0:
                label = numpy.take(self._y, indices)
                label.sort()
                #return the median value of label 
                return (int(label[int((len(label)-1)/2)]+label[int(len(label)/2)]/2))
            else:
                label = numpy.take(self._y, indices)
                label.sort()
                #return the middle value 
                return label[int((len(indices)-1)/2)]

        return self._y[indices[0][0]]

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """

        _, self._ind = self._kdtree.query([example], k=self._k)

        self._ind = self._ind.tolist()
        self._ind = sum(self._ind, [])

        return self.majority(self._ind)

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        #Create an empty dictionary of dictionary and initialize it to 0
        d = defaultdict(dict)
        for xx in range(10):
            for yy in range(10):
                d[xx][yy]=0

        data_index = 0
        for xx, yy in zip(test_x, test_y):
            #classify the test example 
            predicted = self.classify(xx)
            #populate the dictionary     
            d[yy][predicted] += 1
            data_index += 1
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(test_x)))
        return d

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("data/mnist.pkl.gz")

    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in range(10)))
    print("".join(["-"] * 90))
    for ii in range(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in range(10)))
    print("Accuracy: %f" % knn.accuracy(confusion))

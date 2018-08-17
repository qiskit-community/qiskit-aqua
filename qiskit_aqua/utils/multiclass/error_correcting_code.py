import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.multiclass import _ConstantPredictor


class ErrorCorrectingCode:
    """
      the multiclass extension based on the error-correcting-code algorithm.
    """

    def __init__(self, estimator_cls, code_size=4, params=None):
        self.estimator_cls = estimator_cls

        self.code_size = code_size
        self.rand = np.random.RandomState(0)

        self.params = params

    def train(self, X, y):
        """
        training multiple estimators each for distinguishing a pair of classes.
        Args:
            X (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        self.estimators = []
        self.classes = np.unique(y)
        n_classes = self.classes.shape[0]
        code_size = int(n_classes * self.code_size)
        self.codebook = self.rand.random_sample((n_classes, code_size))
        self.codebook[self.codebook > 0.5] = 1
        self.codebook[self.codebook != 1] = -1
        classes_index = dict((c, i) for i, c in enumerate(self.classes))
        Y = np.array([self.codebook[classes_index[y[i]]]
                      for i in range(X.shape[0])], dtype=np.int)
        for i in range(Y.shape[1]):
            Ybit = Y[:, i]
            unique_y = np.unique(Ybit)
            if len(unique_y) == 1:
                estimator = _ConstantPredictor()
                estimator.fit(X, unique_y)
            else:
                if self.params is None:
                    estimator = self.estimator_cls()
                else:
                    estimator = self.estimator_cls(*self.params)

                estimator.fit(X, Ybit)
            self.estimators.append(estimator)

    def test(self, X, y):
        """
        testing multiple estimators each for distinguishing a pair of classes.
        Args:
            X (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        A = self.predict(X)
        B = y
        l = len(A)
        diff = 0
        for i in range(0, l):
            if A[i] != B[i]:
                diff += 1
        print("%d out of %d are wrong" % (diff, l))
        return 1 - (diff * 1.0 / l)

    def predict(self, X):
        """
        applying multiple estimators for prediction
        Args:
            X (numpy.ndarray): input points
        """
        confidences = []
        for e in self.estimators:
            confidence = np.ravel(e.decision_function(X))
            confidences.append(confidence)
        Y = np.array(confidences).T
        pred = euclidean_distances(Y, self.codebook).argmin(axis=1)
        return self.classes[pred]

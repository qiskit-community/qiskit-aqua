# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import numpy as np
from sklearn.utils.validation import _num_samples
from sklearn.preprocessing import LabelBinarizer


class OneAgainstRest:
    """
      the multiclass extension based on the one-against-rest algorithm.
    """

    def __init__(self, estimator_cls, params=None):
        self.estimator_cls = estimator_cls
        self.params = params

    def train(self, X, y):
        """
        training multiple estimators each for distinguishing a pair of classes.
        Args:
            X (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        self.label_binarizer_ = LabelBinarizer(neg_label=-1)
        Y = self.label_binarizer_.fit_transform(y)
        self.classes = self.label_binarizer_.classes_
        columns = (np.ravel(col) for col in Y.T)
        self.estimators = []
        for i, column in enumerate(columns):
            unique_y = np.unique(column)
            if len(unique_y) == 1:
                raise Exception("given all data points are assigned to the same class, the prediction would be boring.")
            if self.params is None:
                estimator = self.estimator_cls()
            else:
                estimator = self.estimator_cls(*self.params)
            estimator.fit(X, column)
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
        n_samples = _num_samples(X)
        maxima = np.empty(n_samples, dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n_samples, dtype=int)
        for i, e in enumerate(self.estimators):
            pred = np.ravel(e.decision_function(X))
            np.maximum(maxima, pred, out=maxima)
            argmaxima[maxima == pred] = i
        return self.classes[np.array(argmaxima.T)]

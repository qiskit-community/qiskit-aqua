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


from sklearn.preprocessing import LabelBinarizer
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from qiskit_aqua.multiclass.dimension_reduction import reduce_dim_to
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.utils.validation import _num_samples


class OneAgainstRest: # binary: 1 and 0
    def __init__(self, estimator_cls, params=None):
        self.estimator_cls = estimator_cls
        self.params = params

    # def balance(self, X, Y, num_of_classes):
    #     cond = (Y==1)
    #     indcond = np.arange(Y.shape[0])[cond]
    #     X_filtered = X[indcond]
    #     Y_filtered = Y[indcond]
    #
    #     for i in range(num_of_classes-2):
    #         Y = np.concatenate((Y,Y_filtered))
    #         X = np.concatenate((X,X_filtered))
    #
    #     return X, Y

    def train(self, X_train, y_train):
        self.label_binarizer_ = LabelBinarizer(neg_label=-1)
        Y = self.label_binarizer_.fit_transform(y_train)
        # Y = Y.tocsc()
        self.classes = self.label_binarizer_.classes_
        num_of_classes = len(self.classes)

        columns = (np.ravel(col) for col in Y.T)
        self.estimators = []
        for i, column in enumerate(columns):
            # print(i, column) #X, column
            unique_y = np.unique(column)
            if len(unique_y) == 1:
                raise Exception("given all data points are assigned to the same class, the prediction would be boring.")
            if self.params == None:
                estimator = self.estimator_cls()
            else:
                estimator = self.estimator_cls(*self.params)

            # X_train_balanced, column_balanced = self.balance(X_train, column, num_of_classes)
            # estimator.fit(X_train_balanced, column_balanced)
            estimator.fit(X_train, column)
            self.estimators.append(estimator)


    def test(self, X, y):
        A = self.predict(X)
        B = y
        l = len(A)
        diff = 0
        for i in range(0, l):
            if A[i] != B[i]:
                diff = diff + 1
        print("%d out of %d are wrong" %(diff, l))
        return 1-(diff*1.0/l)


    def predict(self, X):
        n_samples = _num_samples(X)
        maxima = np.empty(n_samples, dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n_samples, dtype=int)
        for i, e in enumerate(self.estimators):
            pred = np.ravel(e.decision_function(X))
            np.maximum(maxima, pred, out=maxima)
            argmaxima[maxima == pred] = i
        return self.classes[np.array(argmaxima.T)]






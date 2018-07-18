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
from qiskit_acqua.multiclass.dimension_reduction import reduce_dim_to
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.utils.validation import _num_samples
from sklearn.utils.multiclass import _ovr_decision_function


class AllPairs: # binary: 1 and 0
    def __init__(self, estimator_cls, params=None):
        self.estimator_cls = estimator_cls
        self.params = params




    def train(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError(" can not be fit when only one"
                             " class is present.")
        n_classes = self.classes_.shape[0]
        self.estimators = {} # 2d dictionary dict[i][j]
        for i in range(n_classes):
            estimators_from_i = {}
            for j in range(i + 1, n_classes):
                if self.params is None:
                    estimator = self.estimator_cls()
                else:
                    estimator = self.estimator_cls(*self.params)

                cond = np.logical_or(y == i, y == j)
                indcond = np.arange(X.shape[0])[cond]
                X_filtered = X[indcond]
                y_filtered = y[indcond]
                y_filtered[y_filtered == i] = -1
                y_filtered[y_filtered == j] = 1
                estimator.fit(X_filtered, y_filtered)

                estimators_from_i[j] = estimator

            self.estimators[i] = estimators_from_i





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
        predictions = []
        confidences = []
        for i in self.estimators:
            estimators_from_i = self.estimators[i]
            for j in estimators_from_i:
                estimator = estimators_from_i[j]
                confidence = np.ravel(estimator.decision_function(X))

                indices = (confidence > 0).astype(np.int)
                prediction = self.classes_[indices]

                predictions.append(prediction.reshape(-1,1))
                confidences.append(confidence.reshape(-1,1))

        predictions = np.hstack(predictions)
        confidences = np.hstack(confidences)
        Y = _ovr_decision_function(predictions,
                                   confidences, len(self.classes_))
        return self.classes_[Y.argmax(axis=1)]






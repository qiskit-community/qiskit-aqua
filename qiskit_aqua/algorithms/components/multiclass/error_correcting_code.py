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

import logging

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.multiclass import _ConstantPredictor

from qiskit_aqua.algorithms.components.multiclass.multiclass_extension import MulticlassExtension

logger = logging.getLogger(__name__)


class ErrorCorrectingCode(MulticlassExtension):
    """
      the multiclass extension based on the error-correcting-code algorithm.
    """
    ErrorCorrectingCode_CONFIGURATION = {
        'name': 'ErrorCorrectingCode',
        'description': 'ErrorCorrectingCode extension',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'error_correcting_code_schema',
            'type': 'object',
            'properties': {
                'estimator': {
                    'type': 'string',
                    'default': 'RBF_SVC_Estimator',
                    'oneOf': [
                        {'enum': ['RBF_SVC_Estimator', 'QKernalSVM_Estimator']}
                    ]
                },
                'code_size': {
                    'type': 'integer',
                    'default': 4,
                    'minimum': 1
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.ErrorCorrectingCode_CONFIGURATION.copy())
        self.estimator_cls = None
        self.params = None
        self.rand = np.random.RandomState(0)

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
        logger.debug("%d out of %d are wrong" % (diff, l))
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

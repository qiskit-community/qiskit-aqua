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

from sklearn.svm import SVC

from qiskit_aqua.algorithms.components.multiclass_extensions import Estimator


class RBF_SVC_Estimator(Estimator):
    """The estimator that uses the RBF Kernel."""

    def __init__(self):
        self._estimator = SVC(kernel='rbf')

    def fit(self, x, y):
        """
        fit values for the points and the labels
        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        self._estimator.fit(x, y)

    def decision_function(self, x):
        """
        predicted values for the points which account for both the labels and the confidence
        Args:
            x (numpy.ndarray): input points
        """
        return self._estimator.decision_function(x)

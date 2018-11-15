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

from qiskit_aqua.algorithms.components.multiclass_extensions import Estimator
from qiskit_aqua.algorithms.many_sample.qsvm import QSVM_Kernel_Binary


class QSVM_Kernel_Estimator(Estimator):
    """The estimator that uses the quantum kernel."""

    def __init__(self, feature_map, qalgo):
        super(QSVM_Kernel_Estimator, self).__init__()
        self.qsvm_binary = QSVM_Kernel_Binary()
        self.qsvm_binary.init_args({}, {}, [], feature_map, qalgo)
        self._ret = {}

    def fit(self, x, y):
        """
        fit values for the points and the labels
        Args:
            x (numpy.ndarray): input points, NxD array
            y (numpy.ndarray): input labels, Nx1 array
        """
        self.qsvm_binary.train(x, y)
        self._ret = self.qsvm_binary.ret

    def decision_function(self, x):
        """
        predicted values for the points which account for both the labels and the confidence
        Args:
            x (numpy.ndarray): NxD array
        Returns:
            numpy.ndarray: predicted confidence, Nx1 array
        """
        confidence = self.qsvm_binary.get_predicted_confidence(x)
        return confidence

    @property
    def ret(self):
        return self._ret

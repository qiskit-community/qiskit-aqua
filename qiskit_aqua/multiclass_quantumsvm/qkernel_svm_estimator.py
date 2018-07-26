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

from qiskit_aqua.multiclass.estimator import Estimator
from sklearn.svm import LinearSVC
from qiskit_aqua.svm.svm_qkernel import SVM_QKernel
import numpy as np
from qiskit_aqua.svm import (get_points_and_labels, optimize_SVM,
                             kernel_join, entangler_map_creator)

class QKernalSVM_Estimator(Estimator, SVM_QKernel):
    def __init__(self, backend=None, shots=None):
        super(QKernalSVM_Estimator, self).__init__()
        self._backend = backend
        self.shots = shots



    def fit(self, X, y):
        y=y.astype(float) # to make sure cvxopt does not complain about the type!

        self.class_labels = np.unique(y)
        if len(self.class_labels) == 1:
            raise ValueError(" can not be fit when only one"
                             " class is present.")

        self.num_of_qubits = X.shape[1]
        self.entangler_map = entangler_map_creator(self.num_of_qubits)
        self.coupling_map = None
        self.initial_layout = None
        if self._backend is None:
            self._backend = 'local_qasm_simulator'
        if self.shots is None:
            self.shots = 1000

        kernel_matrix = kernel_join(X, X, self.entangler_map,
                                    self.coupling_map, self.initial_layout, self.shots,
                                    self._random_seed, self.num_of_qubits, self._backend)

        [alpha, b, support] = optimize_SVM(kernel_matrix, y)
        alphas = np.array([])
        SVMs = np.array([])
        yin = np.array([])
        for alphindex in range(len(support)):
            if support[alphindex]:
                alphas = np.vstack([alphas, alpha[alphindex]]) if alphas.size else alpha[alphindex]
                SVMs = np.vstack([SVMs, X[alphindex]]) if SVMs.size else X[alphindex]
                yin = np.vstack([yin, y[alphindex]]
                                ) if yin.size else y[alphindex]

        self._ret['svm'] = {}
        self._ret['svm']['alphas'] = alphas
        self._ret['svm']['bias'] = b
        self._ret['svm']['support_vectors'] = SVMs
        self._ret['svm']['yin'] = yin



    def decision_function(self, X):
        alphas = self._ret['svm']['alphas']
        bias = self._ret['svm']['bias']
        SVMs = self._ret['svm']['support_vectors']
        yin = self._ret['svm']['yin']

        kernel_matrix = kernel_join(X, SVMs, self.entangler_map, self.coupling_map,
                                    self.initial_layout, self.shots, self._random_seed,
                                    self.num_of_qubits, self._backend)

        total_num_points = len(X)
        Lsign = np.zeros(total_num_points)
        for tin in range(total_num_points):
            Ltot = 0
            for sin in range(len(SVMs)):
                L = yin[sin]*alphas[sin]*kernel_matrix[tin][sin]
                Ltot += L
            Lsign[tin] = Ltot+bias
        return Lsign

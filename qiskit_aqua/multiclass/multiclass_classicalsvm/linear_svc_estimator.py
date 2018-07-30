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

class LinearSVC_Estimator(Estimator):
    def __init__(self):
        self._estimator = LinearSVC(random_state=0)

    def fit(self, X, y):
        self._estimator.fit(X, y)

    def decision_function(self, X):
        return self._estimator.decision_function(X)


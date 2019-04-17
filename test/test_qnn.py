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
import os
import unittest

import numpy as np

from test.common import QiskitAquaTestCase
from qiskit import BasicAer
from qiskit.aqua.input import SVMInput
from qiskit.aqua import run_algorithm, QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QSVMVariational
from qiskit.aqua.components.optimizers import SPSA, COBYLA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.variational_forms import RYRZ, RY

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

class TestQNN(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        self.feature_dim = 4 # dimension of each data point
        self.training_dataset_size = 20
        self.testing_dataset_size = 10
        self.random_seed = 10598
        np.random.seed(self.random_seed)

        sample_Total, training_input, test_input, class_labels = Wine(training_size=self.training_dataset_size,
                                                                             test_size=self.testing_dataset_size,
                                                                             n=self.feature_dim)
        self.svm_input = SVMInput(training_input, test_input)

    # We test the accuracy upon the Wine dataset from sklearn
    def test_qsvm_variational_via_run_algorithm(self):
        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'QNN'}, #QSVM.Variational
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'qasm_simulator', 'shots': 1024},
            'optimizer': {'name': 'SPSA', 'max_trials': 500, 'save_steps': 1},
            'variational_form': {'name': 'RYRZ', 'depth': 3}
        }
        result = run_algorithm(params, self.svm_input)
        print(result['testing_accuracy'])


def Wine(training_size, test_size, n):
    class_labels = [r'A', r'B', r'C']

    data, target = datasets.load_wine(True)
    sample_train, sample_test, label_train, label_test = train_test_split(data, target, test_size=test_size, random_state=7)

    # Now we standarize for gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Now reduce number of features to number of qubits
    pca = PCA(n_components=n).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)

    # Scale to the range (-1,+1)
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)
    # Pick training size number of samples from each distro
    training_input = {key: (sample_train[label_train == k, :])[:training_size] for k, key in enumerate(class_labels)}
    test_input = {key: (sample_train[label_train == k, :])[training_size:(
        training_size+test_size)] for k, key in enumerate(class_labels)}
    return sample_train, training_input, test_input, class_labels


if __name__ == '__main__':
    unittest.main()

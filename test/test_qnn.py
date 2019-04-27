# -*- coding: utf-8 -*-

# Copyright 2019 IBM RESEARCH. All Rights Reserved.
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

import unittest
import scipy
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from qiskit.aqua.input import SVMInput
from qiskit.aqua import run_algorithm
from test.common import QiskitAquaTestCase


class TestQNN(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        self.feature_dim = 4  # dimension of each data point
        self.training_dataset_size = 20
        self.testing_dataset_size = 10
        self.random_seed = 10598
        np.random.seed(self.random_seed)

        sample_total, training_input, test_input, class_labels = wine_data(
            training_size=self.training_dataset_size,
            test_size=self.testing_dataset_size,
            n=self.feature_dim
        )
        self.svm_input = SVMInput(training_input, test_input)

    # We test the accuracy upon the Wine dataset from sklearn
    def test_qnn_via_run_algorithm(self):
        params = {
            'problem': {'name': 'classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'VQC'},
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'statevector_simulator'},
            'optimizer': {'name': 'COBYLA', 'maxiter': 200},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
            'feature_map': {'name': 'RawFeatureVector', 'feature_dimension': self.feature_dim}
        }
        result = run_algorithm(params, self.svm_input)
        self.log.debug(result['testing_accuracy'])

        self.assertGreater(result['testing_accuracy'], 0.85)

    def test_qsvm_variational_via_run_algorithm(self):
        params = {
            'problem': {'name': 'classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'VQC'},
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'statevector_simulator'},
            'optimizer': {'name': 'COBYLA', 'maxiter': 200},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
        }
        result = run_algorithm(params, self.svm_input)
        self.log.debug(result['testing_accuracy'])

        self.assertLess(result['testing_accuracy'], 0.6)

    def test_qnn_2d_via_run_algorithm(self):
        n_dim = 2
        params = {
            'problem': {'name': 'classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'VQC'},
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'statevector_simulator'},
            'optimizer': {'name': 'COBYLA'},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
            'feature_map': {'name': 'RawFeatureVector', 'feature_dimension': n_dim}
        }
        sample_Total, training_input, test_input, class_labels = ad_hoc_data(
            training_size=20, test_size=10, n=n_dim, gap=0.3
        )
        self.svm_input = SVMInput(training_input, test_input)

        result = run_algorithm(params, self.svm_input)
        self.log.debug(result['testing_accuracy'])
        self.assertGreater(result['testing_accuracy'], 0.55)
        self.assertLess(result['testing_accuracy'], 0.7)

    def test_qsvm_variational_2d_via_run_algorithm(self):
        params = {
            'problem': {'name': 'classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'VQC'}, #
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'statevector_simulator'},
            'optimizer': {'name': 'COBYLA'},
            'variational_form': {'name': 'RYRZ', 'depth': 3}
        }
        n_dim = 2
        sample_Total, training_input, test_input, class_labels = ad_hoc_data(training_size=20,
                                                                             test_size=10,
                                                                             n=n_dim, gap=0.3)
        self.svm_input = SVMInput(training_input, test_input)

        result = run_algorithm(params, self.svm_input)
        self.log.debug(result['testing_accuracy'])

        self.assertGreater(result['testing_accuracy'], 0.9)


def wine_data(training_size, test_size, n):
    class_labels = [r'A', r'B', r'C']

    data, target = datasets.load_wine(True)
    sample_train, sample_test, label_train, label_test = train_test_split(
        data, target, test_size=test_size, random_state=7
    )

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
    training_input = {
        key: (sample_train[label_train == k, :])[:training_size]
        for k, key in enumerate(class_labels)
    }
    test_input = {
        key: (sample_train[label_train == k, :])[training_size:(training_size + test_size)]
        for k, key in enumerate(class_labels)
    }
    return sample_train, training_input, test_input, class_labels


def ad_hoc_data(training_size, test_size, n, gap):
    class_labels = [r'A', r'B']
    if n == 2:
        N = 100
    elif n == 3:
        N = 20   # courseness of data seperation

    label_train = np.zeros(2*(training_size+test_size))
    sample_train = []
    sampleA = [[0 for x in range(n)] for y in range(training_size+test_size)]
    sampleB = [[0 for x in range(n)] for y in range(training_size+test_size)]

    sample_Total = [[[0 for x in range(N)] for y in range(N)] for z in range(N)]

    interactions = np.transpose(np.array([[1, 0], [0, 1], [1, 1]]))

    steps = 2 * np.pi / N

    sx = np.array([[0, 1], [1, 0]])
    X = np.asmatrix(sx)
    sy = np.array([[0, -1j], [1j, 0]])
    Y = np.asmatrix(sy)
    sz = np.array([[1, 0], [0, -1]])
    Z = np.asmatrix(sz)
    J = np.array([[1, 0], [0, 1]])
    J = np.asmatrix(J)
    H = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    H2 = np.kron(H, H)
    H3 = np.kron(H, H2)
    H = np.asmatrix(H)
    H2 = np.asmatrix(H2)
    H3 = np.asmatrix(H3)

    f = np.arange(2 ** n)

    my_array = [[0 for x in range(n)] for y in range(2 ** n)]

    for arindex in range(len(my_array)):
        temp_f = bin(f[arindex])[2:].zfill(n)
        for findex in range(n):
            my_array[arindex][findex] = int(temp_f[findex])

    my_array = np.asarray(my_array)
    my_array = np.transpose(my_array)

    # Define decision functions
    maj = (-1) ** (2 * my_array.sum(axis=0) > n)
    parity = (-1) ** (my_array.sum(axis=0))
    dict1 = (-1) ** (my_array[0])
    if n == 2:
        D = np.diag(parity)
    elif n == 3:
        D = np.diag(maj)

    Basis = np.random.random((2 ** n, 2 ** n)) + 1j * np.random.random((2 ** n, 2 ** n))
    Basis = np.asmatrix(Basis).getH() * np.asmatrix(Basis)

    [S, U] = np.linalg.eig(Basis)

    idx = S.argsort()[::-1]
    S = S[idx]
    U = U[:, idx]

    M = (np.asmatrix(U)).getH() * np.asmatrix(D) * np.asmatrix(U)

    psi_plus = np.transpose(np.ones(2)) / np.sqrt(2)
    psi_0 = 1
    for k in range(n):
        psi_0 = np.kron(np.asmatrix(psi_0), np.asmatrix(psi_plus))

    sample_total_A = []
    sample_total_B = []
    sample_total_void = []
    if n == 2:
        for n1 in range(N):
            for n2 in range(N):
                x1 = steps * n1
                x2 = steps * n2
                phi = x1 * np.kron(Z, J) + x2 * np.kron(J, Z) + (np.pi-x1) * (np.pi-x2) * np.kron(Z, Z)
                Uu = scipy.linalg.expm(1j * phi)
                psi = np.asmatrix(Uu) * H2 * np.asmatrix(Uu) * np.transpose(psi_0)
                temp = np.asscalar(np.real(psi.getH() * M * psi))
                if temp > gap:
                    sample_Total[n1][n2] = +1
                elif temp < -gap:
                    sample_Total[n1][n2] = -1
                else:
                    sample_Total[n1][n2] = 0

        # Now sample randomly from sample_Total a number of times training_size+testing_size
        tr = 0
        while tr < (training_size + test_size):
            draw1 = np.random.choice(N)
            draw2 = np.random.choice(N)
            if sample_Total[draw1][draw2] == +1:
                sampleA[tr] = [2 * np.pi * draw1 / N, 2 * np.pi * draw2 / N]
                tr += 1

        tr = 0
        while tr < (training_size+test_size):
            draw1 = np.random.choice(N)
            draw2 = np.random.choice(N)
            if sample_Total[draw1][draw2] == -1:
                sampleB[tr] = [2 * np.pi * draw1 / N, 2 * np.pi * draw2 / N]
                tr += 1

        sample_train = [sampleA, sampleB]

        for lindex in range(training_size + test_size):
            label_train[lindex] = 0
        for lindex in range(training_size + test_size):
            label_train[training_size + test_size + lindex] = 1
        label_train = label_train.astype(int)
        sample_train = np.reshape(sample_train, (2 * (training_size + test_size), n))
        training_input = {
            key: (sample_train[label_train == k, :])[:training_size]
            for k, key in enumerate(class_labels)
        }
        test_input = {
            key: (sample_train[label_train == k, :])[training_size:(training_size + test_size)]
            for k, key in enumerate(class_labels)
        }

    elif n == 3:
        for n1 in range(N):
            for n2 in range(N):
                for n3 in range(N):
                    x1 = steps * n1
                    x2 = steps * n2
                    x3 = steps * n3
                    phi = x1 * np.kron(np.kron(Z, J), J) + \
                          x2 * np.kron(np.kron(J, Z), J) + \
                          x3 * np.kron(np.kron(J, J), Z) + \
                          (np.pi - x1) * (np.pi - x2) * np.kron(np.kron(Z, Z), J) + \
                          (np.pi - x2) * (np.pi - x3) * np.kron(np.kron(J, Z), Z) + \
                          (np.pi - x1) * (np.pi - x3) * np.kron(np.kron(Z, J), Z)
                    Uu = scipy.linalg.expm(1j * phi)
                    psi = np.asmatrix(Uu) * H3 * np.asmatrix(Uu) * np.transpose(psi_0)
                    temp = np.asscalar(np.real(psi.getH() * M * psi))
                    if temp > gap:
                        sample_Total[n1][n2][n3] = +1
                        sample_total_A.append([n1, n2, n3])
                    elif temp < -gap:
                        sample_Total[n1][n2][n3] = -1
                        sample_total_B.append([n1, n2, n3])
                    else:
                        sample_Total[n1][n2][n3] = 0
                        sample_total_void.append([n1, n2, n3])

        # Now sample randomly from sample_Total a number of times training_size+testing_size
        tr = 0
        while tr < (training_size + test_size):
            draw1 = np.random.choice(N)
            draw2 = np.random.choice(N)
            draw3 = np.random.choice(N)
            if sample_Total[draw1][draw2][draw3] == +1:
                sampleA[tr] = [2 * np.pi * draw1 / N, 2 * np.pi * draw2 / N, 2 * np.pi * draw3 / N]
                tr += 1

        tr = 0
        while tr < (training_size + test_size):
            draw1 = np.random.choice(N)
            draw2 = np.random.choice(N)
            draw3 = np.random.choice(N)
            if sample_Total[draw1][draw2][draw3] == -1:
                sampleB[tr] = [2 * np.pi * draw1 / N, 2 * np.pi * draw2 / N, 2 * np.pi * draw3 / N]
                tr += 1

        sample_train = [sampleA, sampleB]

        for lindex in range(training_size + test_size):
            label_train[lindex] = 0
        for lindex in range(training_size + test_size):
            label_train[training_size + test_size + lindex] = 1
        label_train = label_train.astype(int)
        sample_train = np.reshape(sample_train, (2 * (training_size + test_size), n))
        training_input = {
            key: (sample_train[label_train == k, :])[:training_size]
            for k, key in enumerate(class_labels)
        }
        test_input = {
            key: (sample_train[label_train == k, :])[training_size:(training_size + test_size)]
            for k, key in enumerate(class_labels)
        }

    return sample_Total, training_input, test_input, class_labels


if __name__ == '__main__':
    unittest.main()

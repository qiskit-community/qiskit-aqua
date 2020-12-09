# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QuantumKernel """

import unittest

from test.aqua import QiskitAquaTestCase

import numpy as np

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua import QuantumInstance
from qiskit.ml.algorithms import QuantumKernel

class TestQuantumKernel(QiskitAquaTestCase):
    """ Test QuantumKernel """

    def setUp(self):
        super().setUp()

        self.random_seed = 10598
        self.shots = 12000

        self.qasm_simulator = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                              shots=self.shots,
                                              seed_simulator=self.random_seed,
                                              seed_transpiler=self.random_seed)
        self.qasm_sample = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                           shots=10,
                                           seed_simulator=self.random_seed,
                                           seed_transpiler=self.random_seed)
        self.statevector_simulator = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                     shots=1,
                                                     seed_simulator=self.random_seed,
                                                     seed_transpiler=self.random_seed)

        self.feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

        self.sample_train = np.asarray([[2.95309709, 2.51327412],
                                        [3.14159265, 4.08407045],
                                        [4.08407045, 2.26194671],
                                        [4.46106157, 2.38761042]])
        self.label_train = np.asarray([0,0,1,1])

        self.sample_test  = np.asarray([[3.83274304, 2.45044227],
                                        [3.89557489, 0.31415927]])
        self.label_test = np.asarray([0,1])

        self.ref_kernel_train = {
            'qasm': np.array([[1.        , 0.84375   , 0.12109375, 0.35351562],
                              [0.84375   , 1.        , 0.1171875 , 0.44921875],
                              [0.12109375, 0.1171875 , 1.        , 0.65136719],
                              [0.35351562, 0.44921875, 0.65136719, 1.        ]]),
            'statevector': np.array([[1.        , 0.8534228 , 0.12267747, 0.36334379],
                                     [0.8534228 , 1.        , 0.11529017, 0.45246347],
                                     [0.12267747, 0.11529017, 1.        , 0.67137258],
                                     [0.36334379, 0.45246347, 0.67137258, 1.        ]]),
            'qasm_sample': np.array([[1. , 0.9, 0.2, 0.4],
                                    [0.9, 1. , 0.2, 0.7],
                                    [0.2, 0.2, 1. , 0.9],
                                    [0.4, 0.7, 0.9, 1. ]]),
            'qasm_sample_psd': np.array([[1.01098001, 0.88069042, 0.18458327, 0.42079597],
                                         [0.88069042, 1.03395806, 0.22711205, 0.66342794],
                                         [0.18458327, 0.22711205, 1.02164621, 0.87080094],
                                         [0.42079597, 0.66342794, 0.87080094, 1.03938728]])
        }

        self.ref_kernel_test = {
            'qasm': np.array([[0.14453125, 0.328125  ],
                              [0.17675781, 0.36425781],
                              [0.47363281, 0.02246094],
                              [0.14648438, 0.16308594]]),
            'statevector': np.array([[0.1443953 , 0.33041779],
                                     [0.18170069, 0.37663733],
                                     [0.47479649, 0.02115561],
                                     [0.14691763, 0.16106199]])
        }

    def test_qasm_symmetric(self):
        """ Test symmetric matrix evaluation using qasm simulator """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.qasm_simulator)

        kernel = qkclass.evaluate(x_vec=self.sample_train)

        np.testing.assert_array_almost_equal(kernel,
                                             self.ref_kernel_train['qasm'],
                                             decimal=1)

    def test_qasm_unsymmetric(self):
        """ Test unsymmetric matrix evaluation using qasm simulator """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.qasm_simulator)

        kernel = qkclass.evaluate(x_vec=self.sample_train,y_vec=self.sample_test)

        np.testing.assert_array_almost_equal(kernel,
                                             self.ref_kernel_test['qasm'],
                                             decimal=1)

    def test_sv_symmetric(self):
        """ Test symmetric matrix evaluation using state vector simulator """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.statevector_simulator)

        kernel = qkclass.evaluate(x_vec=self.sample_train)

        np.testing.assert_array_almost_equal(kernel,
                                             self.ref_kernel_train['statevector'],
                                             decimal=1)

    def test_sv_unsymmetric(self):
        """ Test unsymmetric matrix evaluation using state vector simulator """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.statevector_simulator)

        kernel = qkclass.evaluate(x_vec=self.sample_train,y_vec=self.sample_test)

        np.testing.assert_array_almost_equal(kernel,
                                             self.ref_kernel_test['statevector'],
                                             decimal=1)

    def test_qasm_sample(self):
        """ Test symmetric matrix qasm sample """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.qasm_sample)

        kernel = qkclass.evaluate(x_vec=self.sample_train)

        np.testing.assert_array_almost_equal(kernel,
                                             self.ref_kernel_train['qasm_sample'],
                                             decimal=1)

    def test_qasm_psd(self):
        """ Test symmetric matrix positive semi-definite enforcement qasm sample """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.qasm_sample,
                                enforce_psd=True)

        kernel = qkclass.evaluate(x_vec=self.sample_train)

        np.testing.assert_array_almost_equal(kernel,
                                             self.ref_kernel_train['qasm_sample_psd'],
                                             decimal=1)

if __name__ == '__main__':
    unittest.main()

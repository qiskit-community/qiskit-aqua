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
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.ml.algorithms import QuantumKernel


class TestQuantumKernelEvaluate(QiskitAquaTestCase):
    """ Test QuantumKernel Evaluate Method"""

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
        self.label_train = np.asarray([0, 0, 1, 1])

        self.sample_test = np.asarray([[3.83274304, 2.45044227],
                                       [3.89557489, 0.31415927]])
        self.label_test = np.asarray([0, 1])

        self.ref_kernel_train = {
            'qasm': np.array([[1.000000, 0.856583, 0.120417, 0.358833],
                              [0.856583, 1.000000, 0.113167, 0.449250],
                              [0.120417, 0.113167, 1.000000, 0.671500],
                              [0.358833, 0.449250, 0.671500, 1.000000]]),
            'statevector': np.array([[1.00000000, 0.85342280, 0.12267747, 0.36334379],
                                     [0.85342280, 1.00000000, 0.11529017, 0.45246347],
                                     [0.12267747, 0.11529017, 1.00000000, 0.67137258],
                                     [0.36334379, 0.45246347, 0.67137258, 1.00000000]]),
            'qasm_sample': np.array([[1.0, 0.9, 0.1, 0.4],
                                     [0.9, 1.0, 0.1, 0.6],
                                     [0.1, 0.1, 1.0, 0.9],
                                     [0.4, 0.6, 0.9, 1.0]]),
            'qasm_sample_psd': np.array([[1.004036, 0.891664, 0.091883, 0.410062],
                                         [0.891664, 1.017215, 0.116764, 0.579220],
                                         [0.091883, 0.116764, 1.016324, 0.879765],
                                         [0.410062, 0.579220, 0.879765, 1.025083]])
        }

        self.ref_kernel_test = {
            'qasm': np.array([[0.140667, 0.327833],
                              [0.177750, 0.371750],
                              [0.467833, 0.018417],
                              [0.143333, 0.156750]]),
            'statevector': np.array([[0.14439530, 0.33041779],
                                     [0.18170069, 0.37663733],
                                     [0.47479649, 0.02115561],
                                     [0.14691763, 0.16106199]])
        }

    def test_qasm_symmetric(self):
        """ Test symmetric matrix evaluation using qasm simulator """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.qasm_simulator)

        kernel = qkclass.evaluate(x_vec=self.sample_train)

        np.testing.assert_allclose(kernel, self.ref_kernel_train['qasm'], rtol=1e-4)

    def test_qasm_unsymmetric(self):
        """ Test unsymmetric matrix evaluation using qasm simulator """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.qasm_simulator)

        kernel = qkclass.evaluate(x_vec=self.sample_train, y_vec=self.sample_test)

        np.testing.assert_allclose(kernel, self.ref_kernel_test['qasm'], rtol=1e-4)

    def test_sv_symmetric(self):
        """ Test symmetric matrix evaluation using state vector simulator """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.statevector_simulator)

        kernel = qkclass.evaluate(x_vec=self.sample_train)

        np.testing.assert_allclose(kernel, self.ref_kernel_train['statevector'], rtol=1e-4)

    def test_sv_unsymmetric(self):
        """ Test unsymmetric matrix evaluation using state vector simulator """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.statevector_simulator)

        kernel = qkclass.evaluate(x_vec=self.sample_train, y_vec=self.sample_test)

        np.testing.assert_allclose(kernel, self.ref_kernel_test['statevector'], rtol=1e-4)

    def test_qasm_sample(self):
        """ Test symmetric matrix qasm sample """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.qasm_sample)

        kernel = qkclass.evaluate(x_vec=self.sample_train)

        np.testing.assert_allclose(kernel, self.ref_kernel_train['qasm_sample'], rtol=1e-4)

    def test_qasm_psd(self):
        """ Test symmetric matrix positive semi-definite enforcement qasm sample """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.qasm_sample,
                                enforce_psd=True)

        kernel = qkclass.evaluate(x_vec=self.sample_train)

        np.testing.assert_allclose(kernel, self.ref_kernel_train['qasm_sample_psd'], rtol=1e-4)

    def test_no_backend(self):
        """ Test no backend provided """
        qkclass = QuantumKernel(feature_map=self.feature_map)

        with self.assertRaises(AquaError):
            _ = qkclass.evaluate(x_vec=self.sample_train)

    def test_xdim(self):
        """ Test incorrect x_vec dimension """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.qasm_simulator)

        with self.assertRaises(ValueError):
            _ = qkclass.evaluate(x_vec=self.label_train)

    def test_ydim(self):
        """ Test incorrect y_vec dimension """
        qkclass = QuantumKernel(feature_map=self.feature_map,
                                quantum_instance=self.qasm_simulator)

        with self.assertRaises(ValueError):
            _ = qkclass.evaluate(x_vec=self.sample_train, y_vec=self.label_train)


class TestQuantumKernelConstructCircuit(QiskitAquaTestCase):
    """ Test QuantumKernel ConstructCircuit Method"""

    def setUp(self):
        super().setUp()

        self.x = [1, 1]
        self.y = [2, 2]
        self.z = [3]

        self.feature_map = ZZFeatureMap(feature_dimension=2, reps=1)

    def test_innerproduct(self):
        """ Test inner product"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x, self.y)
        self.assertEqual(qc.decompose().size(), 16)

    def test_selfinnerproduct(self):
        """ Test self inner product"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x)
        self.assertEqual(qc.decompose().size(), 16)

    def test_innerproduct_nomeasurement(self):
        """ Test inner product no measurement"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x, self.y, measurement=False)
        self.assertEqual(qc.decompose().size(), 14)

    def test_selfinnerprodect_nomeasurement(self):
        """ Test self inner product no measurement"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x, measurement=False)
        self.assertEqual(qc.decompose().size(), 14)

    def test_statevector(self):
        """ Test state vector simulator"""
        qkclass = QuantumKernel(feature_map=self.feature_map)
        qc = qkclass.construct_circuit(self.x, is_statevector_sim=True)
        self.assertEqual(qc.decompose().size(), 7)

    def test_xdim(self):
        """ Test incorrect x dimension """
        qkclass = QuantumKernel(feature_map=self.feature_map)

        with self.assertRaises(ValueError):
            _ = qkclass.construct_circuit(self.z)

    def test_ydim(self):
        """ Test incorrect y dimension """
        qkclass = QuantumKernel(feature_map=self.feature_map)

        with self.assertRaises(ValueError):
            _ = qkclass.construct_circuit(self.x, self.z)


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum Kernel tests.

"""

import unittest
import numpy as np

from test.aqua.common import QiskitAquaTestCase
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.quantum_instance import QuantumInstance
from qiskit import BasicAer

from qiskit.aqua.algorithms.qkernel import QKernel


class QkernelTest(QiskitAquaTestCase):

    def test_build_qkernel_varform(self):
        ryrz = RYRZ(3, depth=3)
        params = np.zeros((5, ryrz.num_parameters))
        backend = BasicAer.get_backend('qasm_simulator')
        qkernel = QKernel(construct_circuit_fn=ryrz.construct_circuit,
                          num_qubits=3,
                          quantum_instance=QuantumInstance(backend))
        kernel = qkernel.construct_kernel_matrix(x1_vec=params).tolist()
        self.assertListEqual(kernel, [[1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1.],])

    def test_build_qkernel_featmap(self):
        second_order = SecondOrderExpansion(feature_dimension=5, depth=2)
        params = np.zeros((5, second_order.num_qubits))
        backend = BasicAer.get_backend('qasm_simulator')
        qkernel = QKernel(construct_circuit_fn=second_order.construct_circuit,
                          num_qubits=5,
                          quantum_instance=QuantumInstance(backend))
        kernel = qkernel.construct_kernel_matrix(x1_vec=params).tolist()
        self.assertListEqual(kernel, [[1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1.],])

    def test_qkernel_statevector(self):
        second_order = SecondOrderExpansion(feature_dimension=5, depth=2)
        params = np.zeros((5, second_order.num_qubits))
        backend = BasicAer.get_backend('statevector_simulator')
        qkernel = QKernel(construct_circuit_fn=second_order.construct_circuit,
                          num_qubits=5,
                          quantum_instance=QuantumInstance(backend))
        kernel = qkernel.construct_kernel_matrix(x1_vec=params).tolist()
        self.assertListEqual(kernel, [[1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1.],])

    def test_qkernel_calculate_diags(self):
        ryrz = RYRZ(3, depth=3)
        params = np.zeros((5, ryrz.num_parameters))
        backend = BasicAer.get_backend('qasm_simulator')
        qkernel = QKernel(construct_circuit_fn=ryrz.construct_circuit,
                          num_qubits=3,
                          quantum_instance=QuantumInstance(backend),)
        kernel = qkernel.construct_kernel_matrix(x1_vec=params, x2_vec=params+1, calculate_diags=True).tolist()
        self.assertNotEqual(np.diagonal(kernel).tolist(), [1., 1., 1., 1., 1.])

    def test_qkernel_save_counts(self):
        ryrz = RYRZ(3, depth=3)
        params = np.zeros((5, ryrz.num_parameters))
        backend = BasicAer.get_backend('qasm_simulator')
        qkernel = QKernel(construct_circuit_fn=ryrz.construct_circuit,
                          num_qubits=3,
                          quantum_instance=QuantumInstance(backend),)
        kernel = qkernel.construct_kernel_matrix(x1_vec=params, preserve_counts=True).tolist()
        self.assertIn({'000': 1024}, qkernel.counts[0])

    def test_qkernel_normalize_matrix(self, norm='l1'):
        ryrz = RYRZ(3, depth=3)
        params = np.zeros((5, ryrz.num_parameters))
        backend = BasicAer.get_backend('qasm_simulator')
        qkernel = QKernel(construct_circuit_fn=ryrz.construct_circuit,
                          num_qubits=3,
                          quantum_instance=QuantumInstance(backend),)
        qkernel.construct_kernel_matrix(x1_vec=params, preserve_counts=True).tolist()
        norm_mat = qkernel.normalize_matrix()
        self.assertListEqual(list(np.sum(norm_mat, axis=0)), list(np.ones(norm_mat.shape[0])))

    def test_qkernel_center_matrix(self, metric='linear'):
        ryrz = RYRZ(3, depth=3)
        params = np.zeros((5, ryrz.num_parameters))
        backend = BasicAer.get_backend('qasm_simulator')
        qkernel = QKernel(construct_circuit_fn=ryrz.construct_circuit,
                          num_qubits=3,
                          quantum_instance=QuantumInstance(backend), )
        qkernel.construct_kernel_matrix(x1_vec=params, preserve_counts=True).tolist()
        centered_mat = qkernel.center_matrix()
        self.assertEqual(np.average(centered_mat), 0)

    def test_qkernel_larger_measurement_basis(self):
        ryrz = RYRZ(3, depth=3)
        params = np.linspace(0, np.pi/2, 5*ryrz.num_parameters).reshape((5, ryrz.num_parameters))
        backend = BasicAer.get_backend('qasm_simulator')
        kernels = []
        max_dist = 4
        for dist in range(max_dist):
            qkernel = QKernel(construct_circuit_fn=ryrz.construct_circuit,
                          num_qubits=3,
                          quantum_instance=QuantumInstance(backend, seed_simulator=50, seed_transpiler=50),
                          measurement_edit_distance=dist
                          )
            kernels += [qkernel.construct_kernel_matrix(x1_vec=params)]
        for dist in range(max_dist-1):
            self.assertTrue((kernels[dist] <= kernels[dist+1]).all())
            self.assertTrue(kernels[dist].all())

    def test_qkernel_construct_single_vector(self):
        ryrz = RYRZ(3, depth=3)
        samples = 5
        params = np.linspace(0, np.pi, samples * ryrz.num_parameters).reshape(samples, ryrz.num_parameters)
        backend = BasicAer.get_backend('qasm_simulator')
        qkernel = QKernel(construct_circuit_fn=ryrz.construct_circuit,
                          num_qubits=3,
                          quantum_instance=QuantumInstance(backend, seed_simulator=50, seed_transpiler=50))
        kernel = qkernel.construct_kernel_matrix(x1_vec=params)
        new_vec = qkernel.construct_kernel_matrix(x1_vec=params, x2_vec=params[0,:].reshape(1, -1))
        np.testing.assert_array_almost_equal(kernel[0,:].reshape(-1,1), new_vec)


if __name__ == '__main__':
    unittest.main()

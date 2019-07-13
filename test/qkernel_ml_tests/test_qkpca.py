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
Quantum Kernel PCA algorithm tests.

"""

import unittest
import numpy as np

from test.common import QiskitAquaTestCase
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua import run_algorithm
from qiskit import BasicAer

from qiskit.aqua.algorithms.qkernel import QKernelPCA


class QkPCATest(QiskitAquaTestCase):

    def test_build_qkpca_varform(self):
        ryrz = RYRZ(3, depth=3)
        samples = 5
        params = np.linspace(0, np.pi, samples*ryrz.num_parameters).reshape(samples, ryrz.num_parameters)
        backend = BasicAer.get_backend('qasm_simulator')
        qkpca = QKernelPCA(circuit_maker=ryrz, dataset=params, num_components=2)
        result = qkpca.run(backend)
        self.assertGreaterEqual(result['explained_variance_ratios'][0], .5)

    def test_build_qkpca_featmap(self):
        second_order = SecondOrderExpansion(feature_dimension=5, depth=2)
        samples = 5
        params = np.linspace(0, np.pi, samples*second_order.num_qubits).reshape(samples, second_order.num_qubits)
        backend = BasicAer.get_backend('qasm_simulator')
        qkpca = QKernelPCA(circuit_maker=second_order,
                           dataset=params,
                           num_components=2)
        result = qkpca.run(backend)
        self.assertGreaterEqual(result['explained_variance_ratios'][0], .5)

    def test_build_qkpca_varform_dict(self):
        samples = 5
        qkpca_input = np.linspace(0, np.pi, samples * 24).reshape(samples, 24)
        qkpca_params = {
            'algorithm': {'name': 'QKernel.PCA',
                          'num_qubits': 3},
            'problem': {'name': 'dimensionality_reduction'},
            'backend': {'name': 'qasm_simulator', 'shots': 1024},
            'variational_form': {'name': 'RYRZ', 'depth': 3}
        }
        result = run_algorithm(qkpca_params, qkpca_input)
        self.assertGreaterEqual(result['explained_variance_ratios'][0], .5)

    def test_build_qkpca_featmap_dict(self):
        samples = 6
        features = 10
        qkpca_input = np.linspace(0, np.pi, samples * features).reshape(samples, features)
        qkpca_params = {
            'algorithm': {'name': 'QKernel.PCA',
                          'num_qubits': features},
            'problem': {'name': 'dimensionality_reduction'},
            'backend': {'name': 'qasm_simulator', 'shots': 1024},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        result = run_algorithm(qkpca_params, qkpca_input)
        self.assertGreaterEqual(result['explained_variance_ratios'][0], .5)

if __name__ == '__main__':
    unittest.main()

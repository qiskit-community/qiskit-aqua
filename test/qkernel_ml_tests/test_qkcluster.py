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
from qiskit.aqua import run_algorithm
from qiskit import BasicAer

from qiskit.aqua.algorithms.qkernel import QKernelCluster


class QkClusterTest(QiskitAquaTestCase):

    def test_build_qkcluster_varform(self):
        samples = 6
        ryrz = RYRZ(3, depth=3)
        params = np.ones((samples, ryrz.num_parameters))
        params[0:int(samples/2), :] *= 0
        backend = BasicAer.get_backend('qasm_simulator')
        qkcluster = QKernelCluster(circuit_maker=ryrz,
                                   dataset=params,
                                   num_clusters=2)
        result = qkcluster.run(backend)
        for mode in qkcluster.modes:
            self.assertGreaterEqual(result[mode]['silhouette_score'], -1)
            self.assertGreaterEqual(result[mode]['calinski_harabaz_score'], -1)
            self.assertGreaterEqual(result[mode]['davies_bouldin_score'], -1)

    def test_build_qkcluster_featmap(self):
        samples = 6
        second_order = SecondOrderExpansion(feature_dimension=samples, depth=2)
        params = np.ones((samples, second_order.num_qubits))
        params[0:int(samples / 2), :] *= 0
        backend = BasicAer.get_backend('qasm_simulator')
        qkcluster = QKernelCluster(circuit_maker=second_order,
                                   dataset=params,
                                   num_clusters=2)
        result = qkcluster.run(backend)
        for mode in qkcluster.modes:
            self.assertGreaterEqual(result[mode]['silhouette_score'], -1)
            self.assertGreaterEqual(result[mode]['calinski_harabaz_score'], -1)
            self.assertGreaterEqual(result[mode]['davies_bouldin_score'], -1)

    def test_build_qkcluster_varform_dict(self):
        samples = 6
        qkclustering_input = np.ones((samples, 24))
        qkclustering_input[0:int(samples/2), :] *= 0
        qkclustering_params = {
            'algorithm': {'name': 'QKernel.Cluster',
                          'num_qubits': 3},
            'problem': {'name': 'clustering'},
            'backend': {'name': 'qasm_simulator', 'shots': 1024},
            'variational_form': {'name': 'RYRZ', 'depth': 3}
        }
        result = run_algorithm(qkclustering_params, qkclustering_input)
        for mode in ['spectral', 'dbscan', 'agglomerative']:
            self.assertGreaterEqual(result[mode]['silhouette_score'], -1)
            self.assertGreaterEqual(result[mode]['calinski_harabaz_score'], -1)
            self.assertGreaterEqual(result[mode]['davies_bouldin_score'], -1)

    def test_build_qkcluster_featmap_dict(self):
        samples = 6
        features = 10
        qkclustering_input = np.ones((samples, features))
        qkclustering_input[0:int(samples / 2), :] *= 0
        qkclustering_params = {
            'algorithm': {'name': 'QKernel.Cluster',
                          'num_qubits': features},
            'problem': {'name': 'clustering'},
            'backend': {'name': 'qasm_simulator', 'shots': 1024},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        result = run_algorithm(qkclustering_params, qkclustering_input)
        for mode in ['spectral', 'dbscan', 'agglomerative']:
            self.assertGreaterEqual(result[mode]['silhouette_score'], -1)
            self.assertGreaterEqual(result[mode]['calinski_harabaz_score'], -1)
            self.assertGreaterEqual(result[mode]['davies_bouldin_score'], -1)

if __name__ == '__main__':
    unittest.main()

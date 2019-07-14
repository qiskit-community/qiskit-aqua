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
Quantum Kernel Classifier tests.

"""

import unittest
import numpy as np

from test.aqua.common import QiskitAquaTestCase
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua import run_algorithm
from qiskit import BasicAer

from qiskit.aqua.algorithms.qkernel import QKernelClassification


class QkClassificationTest(QiskitAquaTestCase):

    def test_build_qkclassification_varform(self):
        samples = 6
        ryrz = RYRZ(3, depth=3)
        params = np.ones((samples, ryrz.num_parameters))
        params[0:int(samples/2), :] = 0
        labels = np.mean(params, axis=1)
        backend = BasicAer.get_backend('qasm_simulator')
        qkcluster = QKernelClassification(circuit_maker=ryrz, X=params, y=labels)
        result = qkcluster.run(backend)
        np.testing.assert_array_equal(result['labels'], labels)
        self.assertGreaterEqual(result['score'], 1)

    def test_build_qkclassification_featmap(self):
        samples = 6
        second_order = SecondOrderExpansion(feature_dimension=samples, depth=2)
        params = np.ones((samples, second_order.num_qubits))
        params[0:int(samples / 2), :] = 0
        labels = np.mean(params, axis=1)
        backend = BasicAer.get_backend('qasm_simulator')
        qkcluster = QKernelClassification(circuit_maker=second_order, X=params, y=labels)
        result = qkcluster.run(backend)
        np.testing.assert_array_equal(result['labels'], labels)
        self.assertGreaterEqual(result['score'], 1)

    def test_qkclassification_multiclass_featmap(self):
        samples = 6
        second_order = SecondOrderExpansion(feature_dimension=samples, depth=2)
        params = np.zeros((samples, second_order.num_qubits))
        params[2:4,:] = 1
        params[4:6,:] = 2
        labels = np.mean(params, axis=1)
        backend = BasicAer.get_backend('qasm_simulator')
        qkcluster = QKernelClassification(circuit_maker=second_order, X=params, y=labels)
        result = qkcluster.run(backend)
        np.testing.assert_array_equal(result['labels'], labels)
        self.assertGreaterEqual(result['score'], 1)

    def test_qkclassification_predict_featmap(self):
        samples = 6
        second_order = SecondOrderExpansion(feature_dimension=samples, depth=2)
        params = np.zeros((samples, second_order.num_qubits))
        params[2:4,:] = 1
        params[4:6,:] = 2
        labels = np.mean(params, axis=1)
        backend = BasicAer.get_backend('qasm_simulator')
        qkcluster = QKernelClassification(circuit_maker=second_order, X=params, y=labels)
        qkcluster.run(backend)

        new_x = np.ones((2, second_order.num_qubits))
        new_x[1,:] = 0
        new_y = np.mean(new_x, axis=1)
        predict = qkcluster.predict(new_x=new_x)
        score = qkcluster.score(new_x=new_x, new_y=new_y)

        np.testing.assert_array_equal(predict, new_y)
        self.assertGreaterEqual(score, 1)

    def test_build_qkclassification_varform_from_dict(self):
        samples = 6
        params = np.ones((samples, 24))
        params[0:int(samples/2), :] = 0
        labels = np.mean(params, axis=1)
        qkclassifier_params = {
            'algorithm': {'name': 'QKernel.Classification',
                          'num_qubits': 3},
            'problem': {'name': 'classification'},
            'backend': {'name': 'qasm_simulator', 'shots': 1024},
            'variational_form': {'name': 'RYRZ', 'depth': 3}
        }
        qkcluster_input = (params, labels)
        result = run_algorithm(qkclassifier_params, qkcluster_input)
        np.testing.assert_array_equal(result['labels'], labels)
        self.assertGreaterEqual(result['score'], 1)

    def test_build_qkclassification_featmap_from_dict(self):
        samples = 6
        features = 10
        params = np.ones((samples, features))
        params[0:int(samples / 2), :] = 0
        labels = np.mean(params, axis=1)
        qkclassifier_params = {
            'algorithm': {'name': 'QKernel.Classification',
                          'num_qubits': features},
            'problem': {'name': 'classification'},
            'backend': {'name': 'qasm_simulator', 'shots': 1024},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        qkcluster_input = (params, labels)
        result = run_algorithm(qkclassifier_params, qkcluster_input)
        np.testing.assert_array_equal(result['labels'], labels)
        self.assertGreaterEqual(result['score'], 1)

if __name__ == '__main__':
    unittest.main()

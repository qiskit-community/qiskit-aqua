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
Quantum Kernel Regression algorithm tests.

"""

import unittest
import numpy as np
import copy

from test.common import QiskitAquaTestCase
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua import QuantumInstance, run_algorithm
from qiskit import BasicAer

from qiskit.aqua.algorithms.qkernel import QKernelRegression


class QkRegressionTest(QiskitAquaTestCase):

    def test_build_qkreg_varform(self):
        ryrz = RYRZ(3, depth=3)
        samples = 5
        params = np.linspace(0, np.pi, samples*ryrz.num_parameters).reshape(samples, ryrz.num_parameters)
        backend = QuantumInstance(BasicAer.get_backend('qasm_simulator'), seed_transpiler=50, seed_simulator=50)
        qk_reg = QKernelRegression(circuit_maker=ryrz, X=params, y=np.sum(params, axis=1))
        result = qk_reg.run(backend)
        for mode in qk_reg.modes:
            self.assertGreaterEqual(result[mode]['score'], .05)

    def test_build_qkreg_featmap(self):
        second_order = SecondOrderExpansion(feature_dimension=5, depth=2)
        samples = 5
        params = np.linspace(0, np.pi, samples*second_order.num_qubits).reshape(samples, second_order.num_qubits)
        backend = QuantumInstance(BasicAer.get_backend('qasm_simulator'), seed_transpiler=50, seed_simulator=50)
        qk_reg = QKernelRegression(circuit_maker=second_order, X=params, y=np.sum(params, axis=1))
        result = qk_reg.run(backend)
        for mode in qk_reg.modes:
            self.assertGreaterEqual(result[mode]['score'], .05)

    def test_regression_predict_varform(self):
        ryrz = RYRZ(3, depth=3)
        samples = 5
        params = np.linspace(0, np.pi, samples*ryrz.num_parameters).reshape(samples, ryrz.num_parameters)
        ys = np.sum(params, axis=1)
        backend = QuantumInstance(BasicAer.get_backend('qasm_simulator'), seed_transpiler=50, seed_simulator=50)
        qk_reg = QKernelRegression(circuit_maker=ryrz,
                                   X=params,
                                   y=ys,
                                   mode_kwargs={'ridge': {'alpha':.00001},
                                    'svr': {'C':50, 'epsilon':.00001},
                                    'gpr': {}}
        )
        qk_reg.run(backend)
        predictions, errors = qk_reg.predict(new_x=params[0:2,:].reshape(2, -1),
                                             new_y_to_score=ys[0:2].reshape(2))
        for error in errors.values():
            np.testing.assert_array_less(error, np.ones(error.shape)/5)

    def test_regression_score_round_robin(self):
        ryrz = RYRZ(3, depth=3)
        samples = 5
        params = np.linspace(0, np.pi, samples*ryrz.num_parameters).reshape(samples, ryrz.num_parameters)
        ys = np.sum(params, axis=1)
        backend = QuantumInstance(BasicAer.get_backend('qasm_simulator'), seed_transpiler=50, seed_simulator=50)
        qk_reg = QKernelRegression(circuit_maker=ryrz,
                                   X=params,
                                   y=np.sum(params, axis=1),
                                   modes='all',
                                   mode_kwargs={'ridge': {'alpha':.00001},
                                    'svr': {'C':50, 'epsilon':.00001},
                                    'gpr': {}}
        )
        qk_reg.run(backend)
        orig_kernel = copy.deepcopy(qk_reg.qkernel.kernel_matrix)
        predictions, scores = qk_reg.score_round_robin(quantum_instance=backend)
        np.testing.assert_allclose(orig_kernel, qk_reg.qkernel.kernel_matrix)
        modes = ['svr', 'ridge', 'gpr']
        # Not a great test, but the function being estimated is messy
        for mode, prediction, score in zip(modes, predictions, scores):
            self.assertIn(mode, prediction)
            self.assertIn(mode, score)
        self.assertEqual(len(predictions), samples)
        self.assertEqual(len(scores), samples)

    def test_build_qkregression_varform_from_dict(self):
        samples = 5
        params = np.linspace(0, np.pi, samples * 24).reshape(samples, 24)
        qkregression_input = (params, np.sum(params, axis=1))
        qkregression_params = {
            'algorithm': {'name': 'QKernel.Regression',
                          'num_qubits': 3},
            'problem': {'name': 'regression'},
            'backend': {'name': 'qasm_simulator', 'shots': 1024},
            'variational_form': {'name': 'RYRZ', 'depth': 3}
        }
        result = run_algorithm(qkregression_params, qkregression_input)
        modes = ['svr', 'ridge', 'gpr']
        for mode in modes:
            self.assertGreaterEqual(result[mode]['score'], .05)

    def test_build_qkregression_featmap_from_dict(self):
        samples = 6
        features = 10
        params = np.linspace(0, np.pi, samples * features).reshape(samples, features)
        qkregression_input = (params, np.sum(params, axis=1))
        qkregression_params = {
            'algorithm': {'name': 'QKernel.Regression',
                          'num_qubits': features},
            'problem': {'name': 'regression'},
            'backend': {'name': 'qasm_simulator', 'shots': 1024},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        result = run_algorithm(qkregression_params, qkregression_input)
        modes = ['svr', 'ridge', 'gpr']
        for mode in modes:
            self.assertGreaterEqual(result[mode]['score'], .05)

if __name__ == '__main__':
    unittest.main()

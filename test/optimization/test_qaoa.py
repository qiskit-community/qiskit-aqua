# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QAOA """

import unittest
from test.optimization import QiskitOptimizationTestCase

import numpy as np
from ddt import ddt, idata, unpack
from qiskit import BasicAer

from qiskit.optimization.applications.ising import max_cut
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.operators import X, I

W1 = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])
P1 = 1
M1 = (I ^ I ^ I ^ X) + (I ^ I ^ X ^ I) + (I ^ X ^ I ^ I) + (X ^ I ^ I ^ I)
S1 = {'0101', '1010'}


W2 = np.array([
    [0., 8., -9., 0.],
    [8., 0., 7., 9.],
    [-9., 7., 0., -8.],
    [0., 9., -8., 0.],
])
P2 = 1
M2 = None
S2 = {'1011', '0100'}


@ddt
class TestQAOA(QiskitOptimizationTestCase):
    """Test QAOA with MaxCut."""
    @idata([
        [W1, P1, M1, S1, False],
        [W2, P2, M2, S2, False],
        [W1, P1, M1, S1, True],
        [W2, P2, M2, S2, True],
    ])
    @unpack
    def test_qaoa(self, w, prob, m, solutions, convert_to_matrix_op):
        """ QAOA test """
        seed = 0
        aqua_globals.random_seed = seed
        self.log.debug('Testing %s-step QAOA with MaxCut on graph\n%s', prob, w)

        backend = BasicAer.get_backend('statevector_simulator')
        optimizer = COBYLA()
        qubit_op, offset = max_cut.get_operator(w)
        qubit_op = qubit_op.to_opflow()
        if convert_to_matrix_op:
            qubit_op = qubit_op.to_matrix_op()

        qaoa = QAOA(qubit_op, optimizer, prob, mixer=m)
        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

        result = qaoa.run(quantum_instance)
        x = sample_most_likely(result.eigenstate)
        graph_solution = max_cut.get_graph_solution(x)
        self.log.debug('energy:             %s', result.eigenvalue.real)
        self.log.debug('time:               %s', result.optimizer_time)
        self.log.debug('maxcut objective:   %s', result.eigenvalue.real + offset)
        self.log.debug('solution:           %s', graph_solution)
        self.log.debug('solution objective: %s', max_cut.max_cut_value(x, w))
        self.assertIn(''.join([str(int(i)) for i in graph_solution]), solutions)

    def test_change_operator_size(self):
        """ QAOA change operator size test """

        aqua_globals.random_seed = 0
        qubit_op, _ = max_cut.get_operator(
            np.array([
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0]
            ]))
        qaoa = QAOA(qubit_op.to_opflow(), COBYLA(), 1)
        quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = qaoa.run(quantum_instance)
        x = sample_most_likely(result.eigenstate)
        graph_solution = max_cut.get_graph_solution(x)
        with self.subTest(msg='QAOA 4x4'):
            self.assertIn(''.join([str(int(i)) for i in graph_solution]), {'0101', '1010'})

        try:
            qubit_op, _ = max_cut.get_operator(
                np.array([
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                ]))
            qaoa.operator = qubit_op.to_opflow()
        except Exception as ex:  # pylint: disable=broad-except
            self.fail("Failed to change operator. Error: '{}'".format(str(ex)))
            return

        result = qaoa.run()
        x = sample_most_likely(result.eigenstate)
        graph_solution = max_cut.get_graph_solution(x)
        with self.subTest(msg='QAOA 6x6'):
            self.assertIn(''.join([str(int(i)) for i in graph_solution]), {'010101', '101010'})

    @idata([
        [W2, S2, None],
        [W2, S2, [0.0, 0.0]],
        [W2, S2, [1.0, 0.8]]
    ])
    @unpack
    def test_qaoa_initial_point(self, w, solutions, init_pt):
        """ Check first parameter value used is initial point as expected """
        optimizer = COBYLA()
        qubit_op, _ = max_cut.get_operator(w)

        first_pt = []

        def cb_callback(eval_count, parameters, mean, std):
            nonlocal first_pt
            if eval_count == 1:
                first_pt = list(parameters)

        quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'))
        qaoa = QAOA(qubit_op, optimizer, initial_point=init_pt, callback=cb_callback,
                    quantum_instance=quantum_instance)

        result = qaoa.compute_minimum_eigenvalue()
        x = sample_most_likely(result.eigenstate)
        graph_solution = max_cut.get_graph_solution(x)

        if init_pt is None:       # If None the preferred initial point of QAOA variational form
            init_pt = [0.0, 0.0]  # i.e. 0,0 should come through as the first point

        with self.subTest('Initial Point'):
            self.assertListEqual(init_pt, first_pt)

        with self.subTest('Solution'):
            self.assertIn(''.join([str(int(i)) for i in graph_solution]), solutions)


if __name__ == '__main__':
    unittest.main()

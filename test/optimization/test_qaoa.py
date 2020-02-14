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

from qiskit.optimization.ising import max_cut
from qiskit.optimization.ising.common import sample_most_likely
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.operators import WeightedPauliOperator

W1 = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])
P1 = 1
M1 = WeightedPauliOperator.from_dict({'paulis': [{'label': 'IIIX', 'coeff': {'real': 1}},
                                                 {'label': 'IIXI', 'coeff': {'real': 1}},
                                                 {'label': 'IXII', 'coeff': {'real': 1}},
                                                 {'label': 'XIII', 'coeff': {'real': 1}}]
                                      })
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
        [W1, P1, M1, S1],
        [W2, P2, M2, S2],
    ])
    @unpack
    def test_qaoa(self, w, prob, m, solutions):
        """ QAOA test """
        seed = 0
        aqua_globals.random_seed = seed
        self.log.debug('Testing %s-step QAOA with MaxCut on graph\n%s', prob, w)

        backend = BasicAer.get_backend('statevector_simulator')
        optimizer = COBYLA()
        qubit_op, offset = max_cut.get_operator(w)

        qaoa = QAOA(qubit_op, optimizer, prob, mixer=m)
        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

        result = qaoa.run(quantum_instance)
        x = sample_most_likely(result['eigvecs'][0])
        graph_solution = max_cut.get_graph_solution(x)
        self.log.debug('energy:             %s', result['energy'])
        self.log.debug('time:               %s', result['eval_time'])
        self.log.debug('maxcut objective:   %s', result['energy'] + offset)
        self.log.debug('solution:           %s', graph_solution)
        self.log.debug('solution objective: %s', max_cut.max_cut_value(x, w))
        self.assertIn(''.join([str(int(i)) for i in graph_solution]), solutions)


if __name__ == '__main__':
    unittest.main()

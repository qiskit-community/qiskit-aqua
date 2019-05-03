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

import unittest

import numpy as np
from parameterized import parameterized
from qiskit import BasicAer

from test.common import QiskitAquaTestCase
from qiskit.aqua.translators.ising import max_cut
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua import Operator, QuantumInstance

w1 = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])
p1 = 1
m1 = Operator().load_from_dict({'paulis':[{'label': 'IIIX', 'coeff': {'real': 1}},
                                          {'label': 'IIXI', 'coeff': {'real': 1}},
                                          {'label': 'IXII', 'coeff': {'real': 1}},
                                          {'label': 'XIII', 'coeff': {'real': 1}}]
                                })
s1 = {'0101', '1010'}


w2 = np.array([
    [0., 8., -9., 0.],
    [8., 0., 7., 9.],
    [-9., 7., 0., -8.],
    [0., 9., -8., 0.],
])
p2 = 1
m2 = None
s2 = {'1011', '0100'}


class TestQAOA(QiskitAquaTestCase):
    """Test QAOA with MaxCut."""
    @parameterized.expand([
        [w1, p1, m1, s1],
        [w2, p2, m2, s2],
    ])
    def test_qaoa(self, w, p, m, solutions):
        self.log.debug('Testing {}-step QAOA with MaxCut on graph\n{}'.format(p, w))
        np.random.seed(0)

        backend = BasicAer.get_backend('statevector_simulator')
        optimizer = COBYLA()
        qubitOp, offset = max_cut.get_max_cut_qubitops(w)

        qaoa = QAOA(qubitOp, optimizer, p, operator_mode='matrix', mixer=m)
        quantum_instance = QuantumInstance(backend)

        result = qaoa.run(quantum_instance)
        x = max_cut.sample_most_likely(result['eigvecs'][0])
        graph_solution = max_cut.get_graph_solution(x)
        self.log.debug('energy:             {}'.format(result['energy']))
        self.log.debug('time:               {}'.format(result['eval_time']))
        self.log.debug('maxcut objective:   {}'.format(result['energy'] + offset))
        self.log.debug('solution:           {}'.format(graph_solution))
        self.log.debug('solution objective: {}'.format(max_cut.max_cut_value(x, w)))
        self.assertIn(''.join([str(int(i)) for i in graph_solution]), solutions)
        if quantum_instance.has_circuit_caching:
            self.assertLess(quantum_instance._circuit_cache.misses, 3)


if __name__ == '__main__':
    unittest.main()

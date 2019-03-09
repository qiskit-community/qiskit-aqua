# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import unittest
import itertools

from parameterized import parameterized

from qiskit.aqua import QuantumInstance, get_aer_backend
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import LogicExpressionOracle as LEO, TruthTableOracle as TTO
from test.common import QiskitAquaTestCase


tests = [
    ['p cnf 3 5 \n -1 -2 -3 0 \n 1 -2 3 0 \n 1 2 -3 0 \n 1 -2 -3 0 \n -1 2 3 0', ['101', '000', '011'], LEO],
    ['p cnf 2 2 \n 1  0 \n -2  0', ['01'], LEO],
    ['p cnf 2 4 \n 1  0 \n -1 0 \n 2  0 \n -2 0', [], LEO],
    ['a & b & c', ['111'], LEO],
    ['(a ^ b) & a & b', [], LEO],
    ['a & b | c & d', ['0011', '1011', '0111', '1100', '1101', '1110', '1111'], LEO],
    ['1000000000000001', ['0000', '1111'], TTO],
    ['00000000', [], TTO],
]

mct_modes = ['basic', 'advanced', 'noancilla']
simulators = ['statevector_simulator', 'qasm_simulator']
optimizations = ['on', 'off']


class TestGrover(QiskitAquaTestCase):
    @parameterized.expand(
        [x[0] + list(x[1:]) for x in list(itertools.product(tests, mct_modes, simulators, optimizations))]
    )
    def test_grover(self, input, sol, oracle_cls, mct_mode, simulator, optimization='off'):
        self.groundtruth = sol
        if optimization == 'off':
            oracle = oracle_cls(input, optimization='off')
        else:
            oracle = oracle_cls(input, optimization='qm-dlx' if oracle_cls == TTO else 'espresso')
        grover = Grover(oracle, incremental=True, mct_mode=mct_mode)
        backend = get_aer_backend(simulator)
        quantum_instance = QuantumInstance(backend, shots=1000)

        ret = grover.run(quantum_instance)

        self.log.debug('Ground-truth Solutions: {}.'.format(self.groundtruth))
        self.log.debug('Top measurement:        {}.'.format(ret['top_measurement']))
        if ret['oracle_evaluation']:
            self.assertIn(ret['top_measurement'], self.groundtruth)
            self.log.debug('Search Result:          {}.'.format(ret['result']))
        else:
            self.assertEqual(self.groundtruth, [])
            self.log.debug('Nothing found.')


if __name__ == '__main__':
    unittest.main()

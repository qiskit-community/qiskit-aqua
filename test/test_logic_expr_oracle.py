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

import itertools
import unittest

from parameterized import parameterized
from qiskit import execute as q_execute
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.aqua import get_aer_backend

from qiskit.aqua.components.oracles import LogicExpressionOracle
from test.common import QiskitAquaTestCase

dimacs_tests = [
    [
        'p cnf 2 4 \n 1 2 0 \n 1 -2 0 \n -1 2 0 \n -1 -2 0',
        []
    ],

    [
        '(a | b) & (a | ~b) & (~a | b) & (~a | ~b)',
        [],
    ],

    [
        'p cnf 2 3 \n 1 2 0 \n 1 -2 0 \n -1 2 0',
        [(True, True)]
    ],

    [
        '(v[0] | v[1]) & (v[0] | ~v[1]) & (~v[0] | v[1])',
        [(True, True)]
    ],

    [
        'p cnf 3 4 \n 1 -2 3 0 \n 1 2 -3 0 \n 1 -2 -3 0 \n -1 2 3 0',
        [(True, True, True), (True, False, True), (False, False, False), (True, True, False)]
    ],

    [
        'x ^ y',
        [(True, False), (False, True)]
    ]
]

mct_modes = ['basic', 'advanced', 'noancilla']
optimizations = ['off', 'espresso']


class TestLogicExpressionOracle(QiskitAquaTestCase):
    @parameterized.expand(
        [x[0] + list(x[1:]) for x in list(itertools.product(dimacs_tests, mct_modes, optimizations))]
    )
    def test_logic_expr_oracle(self, dimacs_str, sols, mct_mode, optimization='off'):
        num_shots = 1024
        leo = LogicExpressionOracle(dimacs_str, optimization=optimization, mct_mode=mct_mode)
        leo_circuit = leo.circuit
        m = ClassicalRegister(1, name='m')
        for assignment in itertools.product([True, False], repeat=len(leo.variable_register)):
            qc = QuantumCircuit(m, leo.variable_register)
            for idx, tf in enumerate(assignment):
                if tf:
                    qc.x(leo.variable_register[idx])
            qc += leo_circuit
            qc.barrier(leo.output_register)
            qc.measure(leo.output_register, m)
            # print(qc.draw(line_length=10000))
            counts = q_execute(qc, get_aer_backend(
                'qasm_simulator'), shots=num_shots).result().get_counts(qc)
            if assignment in sols:
                assert(counts['1'] == num_shots)
            else:
                assert(counts['0'] == num_shots)


if __name__ == '__main__':
    unittest.main()

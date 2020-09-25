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

""" Test Logical Expression Oracle """

import itertools
import unittest
from test.aqua import QiskitAquaTestCase
from ddt import ddt, idata, unpack
from qiskit import execute as q_execute
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit import BasicAer
from qiskit.aqua.components.oracles import LogicalExpressionOracle

DIMAC_TESTS = [
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
        '(v0 or v1) & (v0 | ~v1) & (not v0 | v1)',
        [(True, True)]
    ],

    [
        'p cnf 3 4 \n 1 -2 3 0 \n 1 2 -3 0 \n 1 -2 -3 0 \n -1 2 3 0',
        [(True, True, True), (True, False, True), (False, False, False), (True, True, False)]
    ],

    [
        'x ^ y',
        [(True, False), (False, True)]
    ],

    [
        '(x & y) | (~x & ~y)',
        [(True, True), (False, False)]
    ]
]

MCT_MODES = ['basic', 'basic-dirty-ancilla', 'advanced', 'noancilla']
OPTIMIZATIONS = [True, False]
LIST_EXPRESSIONS = list(itertools.product(DIMAC_TESTS, MCT_MODES, OPTIMIZATIONS))


@ddt
class TestLogicalExpressionOracle(QiskitAquaTestCase):
    """ Test Logical Expression Oracle """
    @idata(
        [x[0] + list(x[1:]) for x in LIST_EXPRESSIONS]  # type: ignore
    )
    @unpack
    def test_logic_expr_oracle(self, dimacs_str, sols, mct_mode, optimization):
        """ Logic Expr oracle test """
        num_shots = 1024
        leo = LogicalExpressionOracle(dimacs_str, optimization=optimization, mct_mode=mct_mode)
        leo_circuit = leo.circuit
        m = ClassicalRegister(1, name='m')
        for assignment in itertools.product([True, False], repeat=len(leo.variable_register)):
            qc = QuantumCircuit(m, leo.variable_register)
            for idx, t_f in enumerate(assignment):
                if t_f:
                    qc.x(leo.variable_register[idx])
            qc += leo_circuit
            qc.barrier(leo.output_register)
            qc.measure(leo.output_register, m)
            # self.log.debug(qc.draw(line_length=10000))
            counts = q_execute(qc,
                               BasicAer.get_backend('qasm_simulator'),
                               shots=num_shots).result().get_counts(qc)
            if assignment in sols:
                self.assertEqual(counts['1'], num_shots)
            else:
                self.assertEqual(counts['0'], num_shots)


if __name__ == '__main__':
    unittest.main()

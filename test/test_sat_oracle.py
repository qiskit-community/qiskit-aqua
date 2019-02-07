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

from test.common import QiskitAquaTestCase
from qiskit.aqua.components.oracles import SAT


dimacs_cnf_1 = '''
p cnf 2 4
1 2 0
1 -2 0
-1 2 0
-1 -2 0
'''
sols_1 = []

dimacs_cnf_2 = '''
p cnf 2 3
1 2 0
1 -2 0
-1 2 0
'''
sols_2 = [(True, True)]

dimacs_cnf_3 = '''
p cnf 3 4
1 -2 3 0
1 2 -3 0
1 -2 -3 0
-1 2 3 0
'''
sols_3 = [(True, True, True), (True, False, True), (False, False, False), (True, True, False)]


class TestSATOracle(QiskitAquaTestCase):
    @parameterized.expand([
        [dimacs_cnf_1, sols_1],
        [dimacs_cnf_2, sols_2],
        [dimacs_cnf_3, sols_3],
    ])
    def test_sat_oracle(self, dimacs_cnf, sols):
        num_shots = 1024
        for mct_mode in ['basic', 'advanced', 'noancilla']:
            sat = SAT(dimacs_cnf, mct_mode=mct_mode)
            sat_circuit = sat.construct_circuit()
            m = ClassicalRegister(1, name='m')
            for assignment in itertools.product([True, False], repeat=len(sat.variable_register)):
                qc = QuantumCircuit(m, sat.variable_register)
                for idx, tf in enumerate(assignment):
                    if tf:
                        qc.x(sat.variable_register[idx])
                qc += sat_circuit
                qc.barrier(sat.outcome_register)
                qc.measure(sat.outcome_register, m)
                counts = q_execute(qc, get_aer_backend(
                    'qasm_simulator'), shots=num_shots).result().get_counts(qc)
                if assignment in sols:
                    assert(counts['1'] == num_shots)
                else:
                    assert(counts['0'] == num_shots)


if __name__ == '__main__':
    unittest.main()

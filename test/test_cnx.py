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
from itertools import combinations, chain
from parameterized import parameterized
from test.common import QiskitAquaTestCase
from qiskit.tools.qi.qi import state_fidelity
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.wrapper import execute as q_execute
import numpy as np
from qiskit_aqua.utils.summarize_circuits import summarize_circuits as sc
from qiskit import transpiler, load_qasm_string
from qiskit.wrapper import get_backend
import qiskit_aqua.grover.cnx


class TestCNX(QiskitAquaTestCase):
    @parameterized.expand([
        [1, 1],
        [2, 1],
        [3, 1],
        [4, 2],
        [5, 3],
        [6, 4],
        [7, 5],
    ])
    def test_cnx(self, num_controls, num_ancillae):
        c = QuantumRegister(num_controls, name='c')
        o = QuantumRegister(1, name='o')
        allsubsets = list(chain(*[combinations(range(num_controls), ni) for ni in range(num_controls + 1)]))
        for subset in allsubsets:
            for mode in ['basic', 'advanced']:
                qc = QuantumCircuit(o, c)
                if mode == 'basic':
                    if num_controls <= 2:
                        num_ancillae = 0
                else:
                    if num_controls <= 4:
                        num_ancillae = 0
                    else:
                        num_ancillae = 1
                if num_ancillae > 0:
                    a = QuantumRegister(num_ancillae, name='a')
                    qc.add(a)
                for idx in subset:
                    qc.x(c[idx])
                qc.cnx(
                    [c[i] for i in range(num_controls)],
                    [a[i] for i in range(num_ancillae)],
                    o[0],
                    mode=mode
                )
                for idx in subset:
                    qc.x(c[idx])

                qasm_w_basis_gates = transpiler.compile(
                    qc,
                    get_backend('local_qasm_simulator'),
                    basis_gates='u1,u2,u3,cx,id'
                )['circuits'][0]['compiled_circuit_qasm']

                self.log.debug('mode: {}, controls: {}, ancilla: {}, circuit summary: \n{}'.format(
                    mode,
                    num_controls,
                    num_ancillae,
                    sc(load_qasm_string(qasm_w_basis_gates)),
                ))
                vec = np.asarray(q_execute(qc, 'local_statevector_simulator').result().get_statevector(qc))
                vec_o = [0, 1] if len(subset) == num_controls else [1, 0]
                # print(vec, np.array(vec_o + [0] * (2 ** (num_controls + num_ancillae + 1) - 2)))
                f = state_fidelity(vec, np.array(vec_o + [0] * (2 ** (num_controls + num_ancillae + 1) - 2)))
                self.assertAlmostEqual(f, 1)
            return


if __name__ == '__main__':
    unittest.main()

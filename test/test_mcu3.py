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
import numpy as np
from math import pi

from parameterized import parameterized
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import execute as q_execute
from qiskit.quantum_info import state_fidelity

from qiskit.aqua import get_aer_backend
from test.common import QiskitAquaTestCase


class TestMCU3(QiskitAquaTestCase):
    @parameterized.expand(
        [[i + 1] for i in range(7)]
    )
    def test_mcu3(self, num_controls):
        c = QuantumRegister(num_controls, name='c')
        o = QuantumRegister(1, name='o')
        allsubsets = list(chain(*[combinations(range(num_controls), ni) for ni in range(num_controls + 1)]))
        for subset in allsubsets:
            qc = QuantumCircuit(o, c)
            for idx in subset:
                qc.x(c[idx])
            qc.mcu3(
                pi, 0, 0,
                [c[i] for i in range(num_controls)],
                o[0]
            )
            for idx in subset:
                qc.x(c[idx])

            vec = np.asarray(q_execute(qc, get_aer_backend(
                'statevector_simulator')).result().get_statevector(qc, decimals=16))
            vec_o = [0, 1] if len(subset) == num_controls else [1, 0]
            # print(vec, np.array(vec_o + [0] * (2 ** (num_controls + num_ancillae + 1) - 2)))
            f = state_fidelity(vec, np.array(vec_o + [0] * (2 ** (num_controls + 1) - 2)))
            self.assertAlmostEqual(f, 1)


if __name__ == '__main__':
    unittest.main()

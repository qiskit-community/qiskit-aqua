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
import numpy as np

from parameterized import parameterized
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import execute as q_execute
from qiskit.quantum_info import state_fidelity

from qiskit.aqua import get_aer_backend
from test.common import QiskitAquaTestCase

num_controls = [i + 1 for i in range(7)]
modes = ['basic', 'advanced', 'noancilla']


class TestMCT(QiskitAquaTestCase):
    @parameterized.expand(
        itertools.product(num_controls, modes)
    )
    def test_mct(self, num_controls, mode):
        c = QuantumRegister(num_controls, name='c')
        o = QuantumRegister(1, name='o')
        subsets = [tuple(range(i)) for i in range(num_controls + 1)]
        for subset in subsets:
            qc = QuantumCircuit(o, c)
            if mode == 'basic':
                if num_controls <= 2:
                    num_ancillae = 0
                else:
                    num_ancillae = num_controls - 2
            elif mode == 'noancilla':
                num_ancillae = 0
            else:
                if num_controls <= 4:
                    num_ancillae = 0
                else:
                    num_ancillae = 1
            if num_ancillae > 0:
                a = QuantumRegister(num_ancillae, name='a')
                qc.add_register(a)
            for idx in subset:
                qc.x(c[idx])
            qc.mct(
                [c[i] for i in range(num_controls)],
                o[0],
                [a[i] for i in range(num_ancillae)],
                mode=mode
            )
            for idx in subset:
                qc.x(c[idx])

            vec = np.asarray(q_execute(qc, get_aer_backend(
                'statevector_simulator')).result().get_statevector(qc, decimals=16))
            vec_o = [0, 1] if len(subset) == num_controls else [1, 0]
            f = state_fidelity(vec, np.array(vec_o + [0] * (2 ** (num_controls + num_ancillae + 1) - 2)))
            self.assertAlmostEqual(f, 1)


if __name__ == '__main__':
    unittest.main()

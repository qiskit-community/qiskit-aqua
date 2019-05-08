# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest
from itertools import combinations, chain
import numpy as np
from math import pi

from parameterized import parameterized
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import execute as q_execute
from qiskit.quantum_info import state_fidelity

from qiskit import BasicAer
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

            vec = np.asarray(q_execute(qc, BasicAer.get_backend(
                'statevector_simulator')).result().get_statevector(qc, decimals=16))
            vec_o = [0, 1] if len(subset) == num_controls else [1, 0]
            # print(vec, np.array(vec_o + [0] * (2 ** (num_controls + num_ancillae + 1) - 2)))
            f = state_fidelity(vec, np.array(vec_o + [0] * (2 ** (num_controls + 1) - 2)))
            self.assertAlmostEqual(f, 1)


if __name__ == '__main__':
    unittest.main()

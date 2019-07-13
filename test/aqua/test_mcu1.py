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
from qiskit import execute
from qiskit import BasicAer

from test.aqua.common import QiskitAquaTestCase

nums_controls = [[i + 1] for i in range(6)]


class TestMCU1(QiskitAquaTestCase):
    @parameterized.expand(
        nums_controls
    )
    def test_mcu1(self, num_controls):
        c = QuantumRegister(num_controls, name='c')
        o = QuantumRegister(1, name='o')
        allsubsets = list(chain(*[combinations(range(num_controls), ni) for ni in range(num_controls + 1)]))
        for subset in allsubsets:
            control_int = 0
            lam = np.random.random(1)[0] * pi
            qc = QuantumCircuit(o, c)
            for idx in subset:
                control_int += 2**idx
                qc.x(c[idx])
            qc.h(o[0])
            qc.mcu1(
                lam,
                [c[i] for i in range(num_controls)],
                o[0]
            )
            qc.h(o[0])
            for idx in subset:
                qc.x(c[idx])

            mat_mcu = execute(qc, BasicAer.get_backend('unitary_simulator')).result().get_unitary(qc)

            dim = 2**(num_controls+1)
            pos = dim - 2*(control_int+1)
            mat_groundtruth = np.eye(dim, dtype=complex)
            d = np.exp(1.j*lam)
            mat_groundtruth[pos:pos+2, pos:pos+2] = [[(1+d)/2, (1-d)/2],
                                                     [(1-d)/2, (1+d)/2]]
            self.assertTrue(np.allclose(mat_mcu, mat_groundtruth))


if __name__ == '__main__':
    unittest.main()

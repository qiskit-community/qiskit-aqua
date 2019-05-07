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
import itertools
import numpy as np

from parameterized import parameterized
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import execute
from qiskit.quantum_info import state_fidelity

from qiskit import BasicAer
from test.common import QiskitAquaTestCase

nums_controls = [i + 1 for i in range(6)]
clean_ancilla_modes = ['basic']
dirty_ancilla_modes = ['basic-dirty-ancilla', 'advanced', 'noancilla']


class TestMCT(QiskitAquaTestCase):
    @parameterized.expand(
        itertools.product(nums_controls, clean_ancilla_modes)
    )
    def test_mct_with_clean_ancillae(self, num_controls, mode):
        c = QuantumRegister(num_controls, name='c')
        o = QuantumRegister(1, name='o')
        qc = QuantumCircuit(o, c)
        num_ancillae = 0 if num_controls <= 2 else num_controls - 2
        if num_ancillae > 0:
            a = QuantumRegister(num_ancillae, name='a')
            qc.add_register(a)
        qc.h(c)
        qc.mct(
            [c[i] for i in range(num_controls)],
            o[0],
            [a[i] for i in range(num_ancillae)],
            mode=mode
        )
        vec_mct = execute(qc, BasicAer.get_backend('statevector_simulator')).result().get_statevector(qc)

        mat = np.eye(2 ** (num_controls + 1))
        mat[-2:, -2:] = [[0, 1], [1, 0]]
        if num_ancillae > 0:
            mat = np.kron(np.eye(2 ** num_ancillae), mat)

        vec_groundtruth = mat @ np.kron(np.kron(
            np.array([1] + [0] * (2 ** num_ancillae - 1)),
            [1 / 2 ** (num_controls / 2)] * 2 ** num_controls),
            [1, 0]
        )

        f = state_fidelity(vec_mct, vec_groundtruth)
        self.assertAlmostEqual(f, 1)

    @parameterized.expand(
        itertools.product(nums_controls, dirty_ancilla_modes)
    )
    def test_mct_with_dirty_ancillae(self, num_controls, mode):
        c = QuantumRegister(num_controls, name='c')
        o = QuantumRegister(1, name='o')
        qc = QuantumCircuit(o, c)
        if mode == 'basic-dirty-ancilla':
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

        qc.mct(
            [c[i] for i in range(num_controls)],
            o[0],
            [a[i] for i in range(num_ancillae)],
            mode=mode
        )

        mat_mct = execute(qc, BasicAer.get_backend('unitary_simulator')).result().get_unitary(qc)

        mat_groundtruth = np.eye(2 ** (num_controls + 1))
        mat_groundtruth[-2:, -2:] = [[0, 1], [1, 0]]
        if num_ancillae > 0:
            mat_groundtruth = np.kron(np.eye(2 ** num_ancillae), mat_groundtruth)

        self.assertTrue(np.allclose(mat_mct, mat_groundtruth))


if __name__ == '__main__':
    unittest.main()

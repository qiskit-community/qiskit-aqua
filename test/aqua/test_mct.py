# -*- coding: utf-8 -*-

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

""" Test MCT """

import unittest
import itertools
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, idata, unpack
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import execute
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer


NUM_CONTROLS = [i + 1 for i in range(6)]
CLEAN_ANCILLA_MODES = ['basic']
DIRTY_ANCILLA_MODES = ['basic-dirty-ancilla', 'advanced', 'noancilla']


@ddt
class TestMCT(QiskitAquaTestCase):
    """ Test MCT """
    @idata(itertools.product(NUM_CONTROLS, CLEAN_ANCILLA_MODES))
    @unpack
    def test_mct_with_clean_ancillae(self, num_controls, mode):
        """ MCT with Clean Ancillae test """
        c = QuantumRegister(num_controls, name='c')
        q_o = QuantumRegister(1, name='o')
        qc = QuantumCircuit(q_o, c)
        num_ancillae = 0 if num_controls <= 2 else num_controls - 2
        q_a = None
        if num_ancillae > 0:
            q_a = QuantumRegister(num_ancillae, name='a')
            qc.add_register(q_a)
        qc.h(c)
        qc.mct(
            [c[i] for i in range(num_controls)],
            q_o[0],
            [q_a[i] for i in range(num_ancillae)],
            mode=mode
        )
        vec_mct = execute(qc,
                          BasicAer.get_backend(
                              'statevector_simulator')).result().get_statevector(qc)

        mat = np.eye(2 ** (num_controls + 1))
        mat[-2:, -2:] = [[0, 1], [1, 0]]
        if num_ancillae > 0:
            mat = np.kron(np.eye(2 ** num_ancillae), mat)

        vec_groundtruth = mat @ np.kron(np.kron(
            np.array([1] + [0] * (2 ** num_ancillae - 1)),
            [1 / 2 ** (num_controls / 2)] * 2 ** num_controls), [1, 0])

        s_f = state_fidelity(vec_mct, vec_groundtruth)
        self.assertAlmostEqual(s_f, 1)

    @idata(itertools.product(NUM_CONTROLS, DIRTY_ANCILLA_MODES))
    @unpack
    def test_mct_with_dirty_ancillae(self, num_controls, mode):
        """ MCT qith Dirty Ancillae test """
        c = QuantumRegister(num_controls, name='c')
        q_o = QuantumRegister(1, name='o')
        qc = QuantumCircuit(q_o, c)
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
        q_a = None
        if num_ancillae > 0:
            q_a = QuantumRegister(num_ancillae, name='a')
            qc.add_register(q_a)

        qc.mct(
            [c[i] for i in range(num_controls)],
            q_o[0],
            [q_a[i] for i in range(num_ancillae)],
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

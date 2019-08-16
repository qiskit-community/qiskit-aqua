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

""" Test MCR """

import unittest
from test.aqua.common import QiskitAquaTestCase
from itertools import combinations, chain, product
from math import pi
from parameterized import parameterized
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import execute
from qiskit import BasicAer

NUM_CONTROLS = [[i + 1] for i in range(6)]
NUM_CONTROLS_BASIC = [[i + 1] for i in range(4)]
USE_BASIS_GATES_VALS = [True, False]


class TestMCR(QiskitAquaTestCase):
    """ Test MCR """
    @parameterized.expand(
        product(NUM_CONTROLS, USE_BASIS_GATES_VALS)
    )
    def test_mcrx(self, num_controls, use_basis_gates):
        """ mcrx test """
        num_controls = num_controls[0]
        c = QuantumRegister(num_controls, name='c')
        q_o = QuantumRegister(1, name='o')
        allsubsets = list(chain(*[combinations(range(num_controls), ni) for
                                  ni in range(num_controls + 1)]))
        for subset in allsubsets:
            control_int = 0
            theta = np.random.random(1)[0] * pi
            qc = QuantumCircuit(q_o, c)
            for idx in subset:
                control_int += 2**idx
                qc.x(c[idx])
            qc.mcrx(theta, [c[i] for i in range(num_controls)], q_o[0],
                    use_basis_gates=use_basis_gates)
            for idx in subset:
                qc.x(c[idx])

            mat_mcu = execute(qc, BasicAer.get_backend(
                'unitary_simulator')).result().get_unitary(qc)

            dim = 2**(num_controls+1)
            pos = dim - 2*(control_int+1)
            mat_groundtruth = np.eye(dim, dtype=complex)
            rot_mat = np.array([[np.cos(theta / 2), - 1j * np.sin(theta / 2)],
                                [- 1j * np.sin(theta / 2), np.cos(theta / 2)]],
                               dtype=complex)
            mat_groundtruth[pos:pos + 2, pos:pos + 2] = rot_mat
            self.assertTrue(np.allclose(mat_mcu, mat_groundtruth))

    @parameterized.expand(
        product(NUM_CONTROLS, USE_BASIS_GATES_VALS)
    )
    def test_mcry(self, num_controls, use_basis_gates):
        """ mcry test """
        num_controls = num_controls[0]
        c = QuantumRegister(num_controls, name='c')
        q_o = QuantumRegister(1, name='o')
        allsubsets = list(chain(*[combinations(range(num_controls), ni) for
                                  ni in range(num_controls + 1)]))
        for subset in allsubsets:
            control_int = 0
            theta = np.random.random(1)[0] * pi
            qc = QuantumCircuit(q_o, c)
            for idx in subset:
                control_int += 2**idx
                qc.x(c[idx])
            qc.mcry(theta, [c[i] for i in range(num_controls)], q_o[0], None,
                    mode='noancilla', use_basis_gates=use_basis_gates)
            for idx in subset:
                qc.x(c[idx])

            mat_mcu = execute(qc, BasicAer.get_backend(
                'unitary_simulator')).result().get_unitary(qc)
            dim = 2**(num_controls+1)
            pos = dim - 2*(control_int+1)
            mat_groundtruth = np.eye(dim, dtype=complex)
            rot_mat = np.array([[np.cos(theta / 2), - np.sin(theta / 2)],
                                [np.sin(theta / 2), np.cos(theta / 2)]],
                               dtype=complex)
            mat_groundtruth[pos:pos + 2, pos:pos + 2] = rot_mat
            self.assertTrue(np.allclose(mat_mcu, mat_groundtruth))

    @parameterized.expand(
        product(NUM_CONTROLS_BASIC, USE_BASIS_GATES_VALS)
    )
    def test_mcry_basic(self, num_controls, use_basis_gates):
        """ mcry basic test """
        num_controls = num_controls[0]
        if num_controls <= 2:
            num_ancillae = 0
        else:
            num_ancillae = num_controls - 2
        c = QuantumRegister(num_controls, name='c')
        q_o = QuantumRegister(1, name='o')
        allsubsets = list(chain(*[combinations(range(num_controls), ni) for
                                  ni in range(num_controls + 1)]))
        for subset in allsubsets:
            control_int = 0
            theta = np.random.random(1)[0] * pi
            qc = QuantumCircuit(q_o, c)
            if num_ancillae > 0:
                q_a = QuantumRegister(num_ancillae, name='a')
                qc.add_register(q_a)
            else:
                q_a = None
            for idx in subset:
                control_int += 2**idx
                qc.x(c[idx])
            qc.mcry(theta, [c[i] for i in range(num_controls)], q_o[0],
                    [q_a[i] for i in range(num_ancillae)], mode='basic',
                    use_basis_gates=use_basis_gates)
            for idx in subset:
                qc.x(c[idx])

            mat_mcu = execute(qc, BasicAer.get_backend(
                'unitary_simulator')).result().get_unitary(qc)
            dim = 2**(num_controls+1)
            mat_mcu = mat_mcu[:dim, :dim]
            pos = dim - 2*(control_int+1)
            mat_groundtruth = np.eye(dim, dtype=complex)
            rot_mat = np.array([[np.cos(theta / 2), - np.sin(theta / 2)],
                                [np.sin(theta / 2), np.cos(theta / 2)]],
                               dtype=complex)
            mat_groundtruth[pos:pos + 2, pos:pos + 2] = rot_mat
            self.assertTrue(np.allclose(mat_mcu, mat_groundtruth))

    @parameterized.expand(
        product(NUM_CONTROLS, USE_BASIS_GATES_VALS)
    )
    def test_mcrz(self, num_controls, use_basis_gates):
        """ mcrz test """
        num_controls = num_controls[0]
        c = QuantumRegister(num_controls, name='c')
        q_o = QuantumRegister(1, name='o')
        allsubsets = list(chain(*[combinations(range(num_controls), ni) for
                                  ni in range(num_controls + 1)]))
        for subset in allsubsets:
            control_int = 0
            lam = np.random.random(1)[0] * pi
            qc = QuantumCircuit(q_o, c)
            for idx in subset:
                control_int += 2**idx
                qc.x(c[idx])
            qc.mcrz(lam, [c[i] for i in range(num_controls)], q_o[0],
                    use_basis_gates=use_basis_gates)
            for idx in subset:
                qc.x(c[idx])

            mat_mcu = execute(qc, BasicAer.get_backend(
                'unitary_simulator')).result().get_unitary(qc)

            dim = 2**(num_controls+1)
            pos = dim - 2*(control_int+1)
            mat_groundtruth = np.eye(dim, dtype=complex)
            rot_mat = np.array([[1, 0],
                                [0, np.exp(1j * lam)]],
                               dtype=complex)
            mat_groundtruth[pos:pos + 2, pos:pos + 2] = rot_mat
            self.assertTrue(np.allclose(mat_mcu, mat_groundtruth))


if __name__ == '__main__':
    unittest.main()

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
from itertools import combinations, chain, product
from parameterized import parameterized
import numpy as np
from math import pi

from qiskit import QuantumCircuit, QuantumRegister
from qiskit import execute
from qiskit import BasicAer

from test.aqua.common import QiskitAquaTestCase

nums_controls = [[i + 1] for i in range(6)]
nums_controls_basic = [[i + 1] for i in range(4)]
use_basis_gates_vals = [True, False]


class TestMCR(QiskitAquaTestCase):
    @parameterized.expand(
        product(nums_controls, use_basis_gates_vals)
    )
    def test_mcrx(self, num_controls, use_basis_gates):
        num_controls = num_controls[0]
        c = QuantumRegister(num_controls, name='c')
        o = QuantumRegister(1, name='o')
        allsubsets = list(chain(*[combinations(range(num_controls), ni) for
                                  ni in range(num_controls + 1)]))
        for subset in allsubsets:
            control_int = 0
            theta = np.random.random(1)[0] * pi
            qc = QuantumCircuit(o, c)
            for idx in subset:
                control_int += 2**idx
                qc.x(c[idx])
            qc.mcrx(theta, [c[i] for i in range(num_controls)], o[0],
                    use_basis_gates=use_basis_gates)
            for idx in subset:
                qc.x(c[idx])

            mat_mcu = execute(qc, BasicAer.get_backend(
                'unitary_simulator')).result().get_unitary(qc)

            dim = 2**(num_controls+1)
            pos = dim - 2*(control_int+1)
            mat_groundtruth = np.eye(dim, dtype=complex)
            rot_mat = np.array(
                               [[np.cos(theta / 2), - 1j * np.sin(theta / 2)],
                                [- 1j * np.sin(theta / 2), np.cos(theta / 2)]],
                               dtype=complex)
            mat_groundtruth[pos:pos + 2, pos:pos + 2] = rot_mat
            self.assertTrue(np.allclose(mat_mcu, mat_groundtruth))

    @parameterized.expand(
        product(nums_controls, use_basis_gates_vals)
    )
    def test_mcry(self, num_controls, use_basis_gates):
        num_controls = num_controls[0]
        c = QuantumRegister(num_controls, name='c')
        o = QuantumRegister(1, name='o')
        allsubsets = list(chain(*[combinations(range(num_controls), ni) for
                                  ni in range(num_controls + 1)]))
        for subset in allsubsets:
            control_int = 0
            theta = np.random.random(1)[0] * pi
            qc = QuantumCircuit(o, c)
            for idx in subset:
                control_int += 2**idx
                qc.x(c[idx])
            qc.mcry(theta, [c[i] for i in range(num_controls)], o[0], None,
                    mode='noancilla', use_basis_gates=use_basis_gates)
            for idx in subset:
                qc.x(c[idx])

            mat_mcu = execute(qc, BasicAer.get_backend(
                'unitary_simulator')).result().get_unitary(qc)
            dim = 2**(num_controls+1)
            pos = dim - 2*(control_int+1)
            mat_groundtruth = np.eye(dim, dtype=complex)
            rot_mat = np.array(
                               [[np.cos(theta / 2), - np.sin(theta / 2)],
                                [np.sin(theta / 2), np.cos(theta / 2)]],
                               dtype=complex)
            mat_groundtruth[pos:pos + 2, pos:pos + 2] = rot_mat
            self.assertTrue(np.allclose(mat_mcu, mat_groundtruth))

    @parameterized.expand(
        product(nums_controls_basic, use_basis_gates_vals)
    )
    def test_mcry_basic(self, num_controls, use_basis_gates):
        num_controls = num_controls[0]
        if num_controls <= 2:
            num_ancillae = 0
        else:
            num_ancillae = num_controls - 2
        c = QuantumRegister(num_controls, name='c')
        o = QuantumRegister(1, name='o')
        allsubsets = list(chain(*[combinations(range(num_controls), ni) for
                                  ni in range(num_controls + 1)]))
        for subset in allsubsets:
            control_int = 0
            theta = np.random.random(1)[0] * pi
            qc = QuantumCircuit(o, c)
            if num_ancillae > 0:
                a = QuantumRegister(num_ancillae, name='a')
                qc.add_register(a)
            else:
                a = None
            for idx in subset:
                control_int += 2**idx
                qc.x(c[idx])
            qc.mcry(theta, [c[i] for i in range(num_controls)], o[0],
                    [a[i] for i in range(num_ancillae)], mode='basic',
                    use_basis_gates=use_basis_gates)
            for idx in subset:
                qc.x(c[idx])

            mat_mcu = execute(qc, BasicAer.get_backend(
                'unitary_simulator')).result().get_unitary(qc)
            dim = 2**(num_controls+1)
            mat_mcu = mat_mcu[:dim, :dim]
            pos = dim - 2*(control_int+1)
            mat_groundtruth = np.eye(dim, dtype=complex)
            rot_mat = np.array(
                               [[np.cos(theta / 2), - np.sin(theta / 2)],
                                [np.sin(theta / 2), np.cos(theta / 2)]],
                               dtype=complex)
            mat_groundtruth[pos:pos + 2, pos:pos + 2] = rot_mat
            self.assertTrue(np.allclose(mat_mcu, mat_groundtruth))

    @parameterized.expand(
        product(nums_controls, use_basis_gates_vals)
    )
    def test_mcrz(self, num_controls, use_basis_gates):
        num_controls = num_controls[0]
        c = QuantumRegister(num_controls, name='c')
        o = QuantumRegister(1, name='o')
        allsubsets = list(chain(*[combinations(range(num_controls), ni) for
                                  ni in range(num_controls + 1)]))
        for subset in allsubsets:
            control_int = 0
            lam = np.random.random(1)[0] * pi
            qc = QuantumCircuit(o, c)
            for idx in subset:
                control_int += 2**idx
                qc.x(c[idx])
            qc.mcrz(lam, [c[i] for i in range(num_controls)], o[0],
                    use_basis_gates=use_basis_gates)
            for idx in subset:
                qc.x(c[idx])

            mat_mcu = execute(qc, BasicAer.get_backend(
                'unitary_simulator')).result().get_unitary(qc)

            dim = 2**(num_controls+1)
            pos = dim - 2*(control_int+1)
            mat_groundtruth = np.eye(dim, dtype=complex)
            rot_mat = np.array(
                               [[1, 0],
                                [0, np.exp(1j * lam)]],
                               dtype=complex)
            mat_groundtruth[pos:pos + 2, pos:pos + 2] = rot_mat
            self.assertTrue(np.allclose(mat_mcu, mat_groundtruth))


if __name__ == '__main__':
    unittest.main()

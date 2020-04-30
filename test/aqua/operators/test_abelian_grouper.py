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

""" Test AbelianGrouper. """

from qiskit.aqua.operators import ListOp, AbelianGrouper
from qiskit.aqua.operators import PauliOp
from qiskit.quantum_info.operators import Pauli
from test.aqua import QiskitAquaTestCase


# pylint: disable=invalid-name

class TestAbelianGrouper(QiskitAquaTestCase):
    """AbelianGrouper tests."""

    def test_grouper1(self):
        listop = ListOp([
            PauliOp(Pauli(label='IX'), coeff=1),
            PauliOp(Pauli(label='XX'), coeff=2),
            PauliOp(Pauli(label='ZY'), coeff=3),
        ])
        groups = AbelianGrouper.group_subops(listop)
        self.assertEqual(len(groups), 2)
        if len(groups[0]) > len(groups[1]):
            g_0 = list(sorted(groups[0], key=lambda op: str(op.primitive)))
            g_1 = list(sorted(groups[1], key=lambda op: str(op.primitive)))
        else:
            g_0 = list(sorted(groups[1], key=lambda op: str(op.primitive)))
            g_1 = list(sorted(groups[0], key=lambda op: str(op.primitive)))
        self.assertEqual(len(g_0), 2)
        self.assertEqual(str(g_0[0].primitive), 'IX')
        self.assertEqual(g_0[0].coeff, 1)
        self.assertEqual(str(g_0[1].primitive), 'XX')
        self.assertEqual(g_0[1].coeff, 2)
        self.assertEqual(len(g_1), 1)
        self.assertEqual(str(g_1[0].primitive), 'ZY')
        self.assertEqual(g_1[0].coeff, 3)

    def test_grouper2(self):
        listop = ListOp([
            PauliOp(Pauli(label='X'), coeff=1),
            PauliOp(Pauli(label='Y'), coeff=2),
            PauliOp(Pauli(label='Z'), coeff=3),
        ])
        groups = AbelianGrouper.group_subops(listop)
        self.assertEqual(len(groups), 3)

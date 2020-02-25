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

""" Test Operator construction, including OpPrimitives and singletons. """

import unittest
import itertools
import numpy as np

from qiskit import QuantumCircuit, BasicAer, execute, ClassicalRegister
from qiskit.quantum_info import Statevector

from test.aqua import QiskitAquaTestCase
from qiskit.aqua.operators import StateFn, Zero, One, Plus, Minus, OpPrimitive, H, I, Z, X, Y


class TestStateConstruction(QiskitAquaTestCase):
    """State Construction tests."""

    def test_state_singletons(self):
        self.assertEqual(Zero.primitive, {'0': 1})
        self.assertEqual(One.primitive, {'1': 1})

        self.assertEqual((Zero^5).primitive, {'00000': 1})
        self.assertEqual((One^5).primitive, {'11111': 1})
        self.assertEqual(((Zero^One)^3).primitive, {'010101': 1})

    def test_zero_broadcast(self):
        np.testing.assert_array_almost_equal(((H^5) @ Zero).to_matrix(), (Plus^5).to_matrix())

    def test_state_to_matrix(self):
        np.testing.assert_array_equal(Zero.to_matrix(), np.array([1, 0]))
        np.testing.assert_array_equal(One.to_matrix(), np.array([0, 1]))
        np.testing.assert_array_almost_equal(Plus.to_matrix(), (Zero.to_matrix() + One.to_matrix())/(np.sqrt(2)))
        np.testing.assert_array_almost_equal(Minus.to_matrix(), (Zero.to_matrix() - One.to_matrix())/(np.sqrt(2)))

        # TODO Not a great test because doesn't test against validated values or test internal representation. Fix this.
        gnarly_state = (One^Plus^Zero^Minus * .3) @ StateFn(Statevector.from_label('r0+l')) + (StateFn(X^Z^Y^I)*.1j)
        gnarly_mat = gnarly_state.to_matrix()
        gnarly_mat_separate = (One^Plus^Zero^Minus * .3).to_matrix()
        gnarly_mat_separate = np.dot(gnarly_mat_separate, StateFn(Statevector.from_label('r0+l')).to_matrix())
        gnarly_mat_separate = gnarly_mat_separate + (StateFn(X^Z^Y^I)*.1j).to_matrix()
        np.testing.assert_array_almost_equal(gnarly_mat, gnarly_mat_separate)

    def test_qiskit_result_instantiation(self):
        qc = QuantumCircuit(3)
        # REMEMBER: This is Qubit 2 in Operator land.
        qc.h(0)
        sv_res = execute(qc, BasicAer.get_backend('statevector_simulator')).result()
        sv_vector = sv_res.get_statevector()
        qc_op = OpPrimitive(qc)

        qc.add_register(ClassicalRegister(3))
        qc.measure(range(3),range(3))
        qasm_res = execute(qc, BasicAer.get_backend('qasm_simulator')).result()

        print(sv_res.get_counts())

        np.testing.assert_array_almost_equal(StateFn(sv_res).to_matrix(), [0.5, 0.5, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_almost_equal(StateFn(sv_vector).to_matrix(), [.5**.5, .5**.5, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_almost_equal(StateFn(qasm_res).to_matrix(), [0.5, 0.5, 0, 0, 0, 0, 0, 0], decimal=1)

        np.testing.assert_array_almost_equal(((I^I^H)@Zero).to_matrix(), [.5**.5, .5**.5, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_almost_equal((qc_op@Zero).to_matrix(), [.5**.5, .5**.5, 0, 0, 0, 0, 0, 0])

    def test_state_meas_composition(self):
        print((~Zero^4).eval(Zero^4))
        print((~One^4).eval(Zero^4))
        print((~One ^ 4).eval(One ^ 4))

        # print(StateFn(I^Z, is_measurement=True).eval(One^2))

    def test_add_direct(self):
        wf = StateFn({'101010': .5, '111111': .3}) + (Zero^6)
        self.assertEqual(wf.primitive, {'101010': 0.5, '111111': 0.3, '000000': 1.0})
        wf = (4*StateFn({'101010': .5, '111111': .3})) + ((3+.1j)*(Zero ^ 6))
        self.assertEqual(wf.primitive, {'000000': (3+0.1j), '101010': (2+0j), '111111': (1.2+0j)})

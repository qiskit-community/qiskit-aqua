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

""" Test PauliExpectation """

from test.aqua import QiskitAquaTestCase

import numpy as np
import itertools

from qiskit.aqua.operators import X, Y, Z, I, CX, T, H, S, OpPrimitive, OpSum, OpComposition, OpVec
from qiskit.aqua.operators import StateFn, Zero, Plus, Minus

from qiskit.aqua.algorithms.expectation import ExpectationBase, PauliExpectation
from qiskit import QuantumCircuit, BasicAer


class TestPauliExpectation(QiskitAquaTestCase):
    """Pauli Change of Basis Expectation tests."""

    def test_pauli_expect_single(self):
        op = (Z ^ Z)
        # op = (Z ^ Z)*.5 + (I ^ Z)*.5 + (Z ^ X)*.5
        backend = BasicAer.get_backend('qasm_simulator')
        expect = PauliExpectation(operator=op, backend=backend)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = CX @ (H^I) @ Zero
        mean = expect.compute_expectation(wf)
        self.assertAlmostEqual(mean, 0)

        paulis_op = OpVec([X, Y, Z, I])

        expect = PauliExpectation(operator=paulis_op, backend=backend)
        plus_mean = expect.compute_expectation(Plus)
        np.testing.assert_array_almost_equal(plus_mean, [1, -1j, 0, 2**.5])

        minus_mean = expect.compute_expectation(Minus)
        np.testing.assert_array_almost_equal(minus_mean, [-1, 1j, 2**.5, 0])

        bellish_mean = expect.compute_expectation(Plus+Minus)
        np.testing.assert_array_almost_equal(bellish_mean, [0, 0, 2**.5, 2**.5])
        # TODO Fix Identity!

        # plus_mat = Plus.to_matrix()
        # minus_mat = Minus.to_matrix()
        # bellish_mat = (Plus+Minus).to_matrix()
        for i, op in enumerate(paulis_op.oplist):
            print(op)
            mat_op = op.to_matrix()
            np.testing.assert_array_almost_equal(plus_mean[i],
                                                 Plus.adjoint().to_matrix() @ mat_op @ Plus.to_matrix())
            np.testing.assert_array_almost_equal(minus_mean[i],
                                                 Minus.adjoint().to_matrix() @ mat_op @ Minus.to_matrix())
            np.testing.assert_array_almost_equal(bellish_mean[i],
                                                 (Plus+Minus).adjoint().to_matrix() @ mat_op @ (Plus+Minus).to_matrix())

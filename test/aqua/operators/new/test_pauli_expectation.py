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
from qiskit.aqua.operators import StateFn, Zero, One, Plus, Minus

from qiskit.aqua.algorithms.expectation import ExpectationBase, PauliExpectation
from qiskit import QuantumCircuit, BasicAer


class TestPauliExpectation(QiskitAquaTestCase):
    """Pauli Change of Basis Expectation tests."""

    def test_pauli_expect_pair(self):
        op = (Z ^ Z)
        backend = BasicAer.get_backend('qasm_simulator')
        expect = PauliExpectation(operator=op, backend=backend)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = CX @ (H^I) @ Zero
        mean = expect.compute_expectation(wf)
        self.assertAlmostEqual(mean, 0)

    def test_pauli_expect_single(self):
        backend = BasicAer.get_backend('qasm_simulator')
        paulis = [Z, X, Y, I]
        states = [Zero, One, Plus, Minus, S@Plus, S@Minus]
        for pauli, state in itertools.product(paulis, states):
            expect = PauliExpectation(operator=pauli, backend=backend)
            mean = expect.compute_expectation(state)
            matmulmean = state.adjoint().to_matrix() @ pauli.to_matrix() @ state.to_matrix()
            # print('{}, {}'.format(pauli.primitive, np.round(float(matmulmean[0]), decimals=3)))
            np.testing.assert_array_almost_equal(mean, matmulmean)

    def test_pauli_expect_op_vector(self):
        backend = BasicAer.get_backend('qasm_simulator')
        paulis_op = OpVec([X, Y, Z, I])

        expect = PauliExpectation(operator=paulis_op, backend=backend)
        plus_mean = expect.compute_expectation(Plus)
        np.testing.assert_array_almost_equal(plus_mean, [1, 0, 0, 1])

        minus_mean = expect.compute_expectation(Minus)
        np.testing.assert_array_almost_equal(minus_mean, [-1, 0, 0, 1])

        zero_mean = expect.compute_expectation(Zero)
        np.testing.assert_array_almost_equal(zero_mean, [0, 0, 1, 1])

        sum_zero = (Plus+Minus)*(.5**.5)
        sum_zero_mean = expect.compute_expectation(sum_zero)
        np.testing.assert_array_almost_equal(sum_zero_mean, [0, 0, 1, 1])

        for i, op in enumerate(paulis_op.oplist):
            print(op)
            mat_op = op.to_matrix()
            np.testing.assert_array_almost_equal(plus_mean[i],
                                                 Plus.adjoint().to_matrix() @ mat_op @ Plus.to_matrix())
            np.testing.assert_array_almost_equal(minus_mean[i],
                                                 Minus.adjoint().to_matrix() @ mat_op @ Minus.to_matrix())
            np.testing.assert_array_almost_equal(sum_zero_mean[i],
                                                 sum_zero.adjoint().to_matrix() @ mat_op @ sum_zero.to_matrix())

    def test_pauli_expect_op_vector_state_vector(self):
        backend = BasicAer.get_backend('qasm_simulator')
        paulis_op = OpVec([X, Y, Z, I])
        states_op = OpVec([One, Zero, Plus, Minus])

        expect = PauliExpectation(operator=paulis_op, backend=backend)
        means = expect.compute_expectation(states_op)
        print(means)

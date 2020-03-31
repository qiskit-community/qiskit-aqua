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

""" Test AerPauliExpectation """

from test.aqua import QiskitAquaTestCase

import itertools
import numpy as np

from qiskit.aqua.operators import (X, Y, Z, I, CX, H, S,
                                   ListOp, Zero, One, Plus, Minus,
                                   AerPauliExpectation)

from qiskit import Aer


class TestAerPauliExpectation(QiskitAquaTestCase):
    """Pauli Change of Basis Expectation tests."""

    def test_pauli_expect_pair(self):
        """ Test AerPauli expectation for simple 2-qubit case."""
        op = (Z ^ Z)
        backend = Aer.get_backend('qasm_simulator')
        expect = AerPauliExpectation(operator=op, backend=backend)
        # wf_op = (Pl^Pl) + (Ze^Ze)
        wf_op = CX @ (H ^ I) @ Zero
        mean = expect.compute_expectation(wf_op)
        self.assertAlmostEqual(mean, 0, delta=.1)

    def test_pauli_expect_single(self):
        """ Test AerPauli expectation over all single qubit paulis and eigenstates. """
        backend = Aer.get_backend('qasm_simulator')
        paulis = [Z, X, Y, I]
        states = [Zero, One, Plus, Minus, S @ Plus, S @ Minus]
        for pauli, state in itertools.product(paulis, states):
            expect = AerPauliExpectation(operator=pauli, backend=backend)
            mean = expect.compute_expectation(state)
            matmulmean = state.adjoint().to_matrix() @ pauli.to_matrix() @ state.to_matrix()
            # print('{}, {}'.format(pauli.primitive, np.round(float(matmulmean[0]), decimals=3)))
            np.testing.assert_array_almost_equal(mean, matmulmean, decimal=1)

    def test_pauli_expect_op_vector(self):
        """ Test for expectation over ListOp of observables. """
        backend = Aer.get_backend('qasm_simulator')
        paulis_op = ListOp([X, Y, Z, I])
        expect = AerPauliExpectation(operator=paulis_op, backend=backend)

        plus_mean = expect.compute_expectation(Plus)
        np.testing.assert_array_almost_equal(plus_mean, [1, 0, 0, 1], decimal=1)

        # Note! Also tests reuse of expectation.
        minus_mean = expect.compute_expectation(Minus)
        np.testing.assert_array_almost_equal(minus_mean, [-1, 0, 0, 1], decimal=1)

        zero_mean = expect.compute_expectation(Zero)
        np.testing.assert_array_almost_equal(zero_mean, [0, 0, 1, 1], decimal=1)

        # !!NOTE!!: Depolarizing channel (Sampling) means interference
        # does not happen between circuits in sum, so expectation does
        # not equal expectation for Zero!!
        sum_zero = (Plus + Minus) * (.5 ** .5)
        sum_zero_mean = expect.compute_expectation(sum_zero)
        np.testing.assert_array_almost_equal(sum_zero_mean, [0, 0, 0, 2], decimal=1)

        for i, op in enumerate(paulis_op.oplist):
            mat_op = op.to_matrix()
            np.testing.assert_array_almost_equal(zero_mean[i],
                                                 Zero.adjoint().to_matrix() @
                                                 mat_op @ Zero.to_matrix(),
                                                 decimal=1)
            np.testing.assert_array_almost_equal(plus_mean[i],
                                                 Plus.adjoint().to_matrix() @
                                                 mat_op @ Plus.to_matrix(),
                                                 decimal=1)
            np.testing.assert_array_almost_equal(minus_mean[i],
                                                 Minus.adjoint().to_matrix() @
                                                 mat_op @ Minus.to_matrix(),
                                                 decimal=1)

    def test_pauli_expect_state_vector(self):
        """ Test over ListOp of states """
        backend = Aer.get_backend('qasm_simulator')
        states_op = ListOp([One, Zero, Plus, Minus])

        paulis_op = X
        expect = AerPauliExpectation(operator=paulis_op, backend=backend)
        means = expect.compute_expectation(states_op)
        np.testing.assert_array_almost_equal(means, [0, 0, 1, -1], decimal=1)

    def test_pauli_expect_op_vector_state_vector(self):
        """ Test over ListOp of Observables and ListOp of states."""
        backend = Aer.get_backend('qasm_simulator')
        # TODO Bug in Aer with Y Measurements!!
        # paulis_op = ListOp([X, Y, Z, I])
        paulis_op = ListOp([X, Z, I])
        states_op = ListOp([One, Zero, Plus, Minus])

        expect = AerPauliExpectation(operator=paulis_op, backend=backend)
        means = expect.compute_expectation(states_op)
        valids = [[+0, 0, 1, -1],
                  # [+0, 0, 0, 0],
                  [-1, 1, 0, -0],
                  [+1, 1, 1, 1]]
        np.testing.assert_array_almost_equal(means, valids, decimal=1)

    def test_parameterized_qobj(self):
        """ Test direct-to-aer parameter passing in Qobj header. """
        pass

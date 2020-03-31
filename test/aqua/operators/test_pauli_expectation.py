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

import itertools
import unittest
import numpy as np

from qiskit.aqua.operators import (X, Y, Z, I, CX, H, S,
                                   ListOp, Zero, One, Plus, Minus,
                                   PauliExpectation, AbelianGrouper,
                                   CircuitSampler)

from qiskit import BasicAer, IBMQ


# pylint: disable=invalid-name

class TestPauliExpectation(QiskitAquaTestCase):
    """Pauli Change of Basis Expectation tests."""

    def test_pauli_expect_pair(self):
        """ pauli expect pair test """
        op = (Z ^ Z)
        backend = BasicAer.get_backend('qasm_simulator')
        expect = PauliExpectation(operator=op, backend=backend)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = CX @ (H ^ I) @ Zero
        mean = expect.compute_expectation(wf)
        self.assertAlmostEqual(mean, 0, delta=.1)

    def test_pauli_expect_single(self):
        """ pauli expect single test """
        backend = BasicAer.get_backend('qasm_simulator')
        paulis = [Z, X, Y, I]
        states = [Zero, One, Plus, Minus, S @ Plus, S @ Minus]
        for pauli, state in itertools.product(paulis, states):
            expect = PauliExpectation(operator=pauli, backend=backend)
            mean = expect.compute_expectation(state)
            matmulmean = state.adjoint().to_matrix() @ pauli.to_matrix() @ state.to_matrix()
            # print('{}, {}'.format(pauli.primitive, np.round(float(matmulmean[0]), decimals=3)))
            np.testing.assert_array_almost_equal(mean, matmulmean, decimal=1)

    def test_pauli_expect_op_vector(self):
        """ pauli expect op vector test """
        backend = BasicAer.get_backend('qasm_simulator')
        paulis_op = ListOp([X, Y, Z, I])
        expect = PauliExpectation(operator=paulis_op, backend=backend)

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
        """ pauli expect state vector test """
        backend = BasicAer.get_backend('qasm_simulator')
        states_op = ListOp([One, Zero, Plus, Minus])

        paulis_op = X
        expect = PauliExpectation(operator=paulis_op, backend=backend)
        means = expect.compute_expectation(states_op)
        np.testing.assert_array_almost_equal(means, [0, 0, 1, -1], decimal=1)

    def test_pauli_expect_op_vector_state_vector(self):
        """ pauli expect op vector state vector test """
        backend = BasicAer.get_backend('qasm_simulator')
        paulis_op = ListOp([X, Y, Z, I])
        states_op = ListOp([One, Zero, Plus, Minus])

        expect = PauliExpectation(operator=paulis_op, backend=backend)
        means = expect.compute_expectation(states_op)
        valids = [[+0, 0, 1, -1],
                  [+0, 0, 0, 0],
                  [-1, 1, 0, -0],
                  [+1, 1, 1, 1]]
        np.testing.assert_array_almost_equal(means, valids, decimal=1)

    def test_not_to_matrix_called(self):
        """ 45 qubit calculation - literally will not work if to_matrix is
            somehow called (in addition to massive=False throwing an error)"""

        backend = BasicAer.get_backend('qasm_simulator')
        qs = 45
        states_op = ListOp([Zero ^ qs,
                            One ^ qs,
                            (Zero ^ qs) + (One ^ qs)])
        paulis_op = ListOp([Z ^ qs,
                            (I ^ Z ^ I) ^ int(qs / 3)])
        expect = PauliExpectation(operator=paulis_op, backend=backend)
        means = expect.compute_expectation(states_op)
        np.testing.assert_array_almost_equal(means, [[1, -1, 0],
                                                     [1, -1, 0]])

    def test_abelian_grouper(self):
        """ abelian grouper test """
        two_qubit_H2 = (-1.052373245772859 * I ^ I) + \
                       (0.39793742484318045 * I ^ Z) + \
                       (-0.39793742484318045 * Z ^ I) + \
                       (-0.01128010425623538 * Z ^ Z) + \
                       (0.18093119978423156 * X ^ X)
        grouped_sum = AbelianGrouper().convert(two_qubit_H2)
        self.assertEqual(len(grouped_sum.oplist), 2)
        paulis = (I ^ I ^ X ^ X * 0.2) + \
                 (Z ^ Z ^ X ^ X * 0.3) + \
                 (Z ^ Z ^ Z ^ Z * 0.4) + \
                 (X ^ X ^ Z ^ Z * 0.5) + \
                 (X ^ X ^ X ^ X * 0.6) + \
                 (I ^ X ^ X ^ X * 0.7)
        grouped_sum = AbelianGrouper().convert(paulis)
        self.assertEqual(len(grouped_sum.oplist), 4)

    def test_grouped_pauli_expectation(self):
        """ grouped pauli expectation test """
        two_qubit_H2 = (-1.052373245772859 * I ^ I) + \
                       (0.39793742484318045 * I ^ Z) + \
                       (-0.39793742484318045 * Z ^ I) + \
                       (-0.01128010425623538 * Z ^ Z) + \
                       (0.18093119978423156 * X ^ X)
        wf = CX @ (H ^ I) @ Zero
        backend = BasicAer.get_backend('qasm_simulator')
        expect_op = PauliExpectation(operator=two_qubit_H2,
                                     backend=backend,
                                     group_paulis=False).expectation_op(wf)
        sampler = CircuitSampler.factory(backend)
        sampler._extract_circuitstatefns(expect_op)
        num_circuits_ungrouped = len(sampler._circuit_ops_cache)
        self.assertEqual(num_circuits_ungrouped, 5)

        expect_op_grouped = PauliExpectation(operator=two_qubit_H2,
                                             backend=backend,
                                             group_paulis=True).expectation_op(wf)
        sampler = CircuitSampler.factory(backend)
        sampler._extract_circuitstatefns(expect_op_grouped)
        num_circuits_grouped = len(sampler._circuit_ops_cache)
        self.assertEqual(num_circuits_grouped, 2)

    @unittest.skip(reason="IBMQ testing not available in general.")
    def test_ibmq_grouped_pauli_expectation(self):
        """ pauli expect op vector state vector test """
        p = IBMQ.load_account()
        backend = p.get_backend('ibmq_qasm_simulator')
        paulis_op = ListOp([X, Y, Z, I])
        states_op = ListOp([One, Zero, Plus, Minus])

        expect = PauliExpectation(operator=paulis_op, backend=backend)
        means = expect.compute_expectation(states_op)
        valids = [[+0, 0, 1, -1],
                  [+0, 0, 0, 0],
                  [-1, 1, 0, -0],
                  [+1, 1, 1, 1]]
        np.testing.assert_array_almost_equal(means, valids, decimal=1)

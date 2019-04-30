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

import numpy as np

from test.common import QiskitAquaTestCase
from qiskit.aqua import run_algorithm
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising.vehicle_routing import get_vehiclerouting_qubitops
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.quantum_info import Pauli

# To run only this test, issue:
# python -m unittest test.test_vrp.TestVehicleRouting


class TestVehicleRouting(QiskitAquaTestCase):
    """Tests vehicle routing Ising translator."""

    def setUp(self):
        super().setUp()
        np.random.seed(100)
        self.n = 2
        self.K = 1
        self.instance = np.zeros((self.n, self.n))
        self.instance[0, 1] = 0.8
        self.instance[1, 0] = 0.8
        self.qubit_op = get_vehiclerouting_qubitops(self.instance, self.n,
                                                    self.K)
        self.algo_input = EnergyInput(self.qubit_op)

    def test_simple1(self):
        # Compares the output in terms of Paulis.
        paulis = [(79.6, Pauli(z=[True, False], x=[False, False])),
                  (79.6, Pauli(z=[False, True], x=[False, False])),
                  (160.8, Pauli(z=[False, False], x=[False, False]))]
        # Could also consider op = Operator(paulis) and then __eq__, but 
        # that would not use assert_approx_equal
        for pauliA, pauliB in zip(self.qubit_op._paulis, paulis):
            costA, binaryA = pauliA
            costB, binaryB = pauliB
            # Note that the construction is a bit iffy, i.e., can be a small bit off even when the random seed is fixed,
            # even when the ordering is the same. Obviously, when the ordering changes, the test will become invalid.
            np.testing.assert_approx_equal(costA, costB, 2)
            self.assertEqual(binaryA, binaryB)

    def test_simple2(self):
        # Solve the problem using the exact eigensolver
        params = {
            'problem': {
                'name': 'ising'
            },
            'algorithm': {
                'name': 'ExactEigensolver'
            }
        }
        result = run_algorithm(params, self.algo_input)
        A = np.array([0., 0., 0., 1.])
        np.testing.assert_array_almost_equal(A, result['eigvecs'][0], 4)

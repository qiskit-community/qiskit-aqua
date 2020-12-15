# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Vehicle Routing """

import unittest
from test.optimization import QiskitOptimizationTestCase

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.aqua import aqua_globals
from qiskit.opflow import PauliOp
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.optimization.applications.ising.vehicle_routing import get_operator


# To run only this test, issue:
# python -m unittest test.test_vrp.TestVehicleRouting


class TestVehicleRouting(QiskitOptimizationTestCase):
    """Tests vehicle routing Ising translator."""

    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = 100
        self.n = 2
        self.k = 1
        self.instance = np.zeros((self.n, self.n))
        self.instance[0, 1] = 0.8
        self.instance[1, 0] = 0.8
        self.qubit_op = get_operator(self.instance, self.n, self.k)

    def test_simple1(self):
        """ simple1 test """
        # Compares the output in terms of Paulis.
        paulis = (
            PauliOp(Pauli(z=[True, False], x=[False, False]), 79.6)
            + PauliOp(Pauli(z=[False, True], x=[False, False]), 79.6)
            + PauliOp(Pauli(z=[False, False], x=[False, False]), 160.8)
        )
        self.assertEqual(self.qubit_op, paulis)

    def test_simple2(self):
        """ simple2 test """
        # Solve the problem using the exact eigensolver
        result = NumPyMinimumEigensolver(self.qubit_op).run()
        arr = np.array([0., 0., 0., 1.])
        np.testing.assert_array_almost_equal(arr, np.abs(result.eigenstate.to_matrix()) ** 2, 4)


if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()

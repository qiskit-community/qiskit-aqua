# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np

from test.common import QiskitAquaTestCase
from qiskit import BasicAer

from qiskit.aqua import run_algorithm
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising.vrp import *
from qiskit.aqua.algorithms import ExactEigensolver

# To run only this test, issue:
# python -m unittest test.test_vrp.TestVehicleRouting

class TestVehicleRouting(QiskitAquaTestCase):
    """Tests vehicle routing Ising translator."""

    def setUp(self):
        super().setUp()
        np.random.seed(100)        
        self.n = 2
        self.K = 1
        self.instance = np.zeros((self.n,self.n))
        self.instance[0,1] = 0.8
        self.instance[1,0] = 0.8
        self.qubit_op = get_vehiclerouting_qubitops(self.instance, self.n, self.K)
        self.algo_input = EnergyInput(self.qubit_op)

    def test_simple1(self):
        # Compares the output in terms of Paulis.
        paulis = [(79.6, Pauli(z=[True, False], x=[False, False])), (79.6, Pauli(z=[False, True], x=[False, False])), (160.8, Pauli(z=[False, False], x=[False, False]))]
        op = Operator(paulis)
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
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, self.algo_input)
        A = np.array([0., 0., 0., 1.])
        np.testing.assert_array_almost_equal(A, result['eigvecs'][0], 4)
# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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

import unittest

import numpy as np
from numpy.random import random
from parameterized import parameterized
from test.common import QiskitAquaTestCase
from qiskit_aqua import run_algorithm
from qiskit_aqua.input import LinearSystemInput
from qiskit_aqua.utils import random_matrix_generator as rmg

class TestHHL(QiskitAquaTestCase):
    """HHL tests."""

    def setUp(self):
        super(TestHHL, self).setUp()
        np.random.seed(0)
        self.params = {
            "problem": {
                "name": "linear_system",
            },
            "algorithm": {
                "mode": "evaluate",
                "name": "HHL"
            },
            "eigs": {
                "expansion_mode": "suzuki",
                "expansion_order": 2,
                "hermitian_matrix": "true",
                "iqft": {
                    "name": "STANDARD"
                },
                "name": "QPE",
                "negative_evals": "false",
                "num_ancillae": 3,
                "num_time_slices": 50,
                "paulis_grouping": "random",
                "use_basis_gates": "true"
            },
            "reciprocal": {
                "name": "Lookup",
                "negative_evals": "false",
                "scale": 0.0
            },
            "backend": {
                "name": "statevector_simulator",
                "skip_transpiler": "false"
            }
        }

    @parameterized.expand([[[0, 1]], [[1, 0]], [[1, 1]], [[1, 10]]])
    def test_hhl_diagonal_sv(self, vector):
        self.log.debug('Testing HHL simple test in mode Lookup with '
                       'statevector simulator')

        matrix = [[1, 0], [0, 1]]
        self.params["input"] = {
            "name": "LinearSystemInput",
            "matrix": matrix,
            "vector": vector
        }

        # run hhl
        result = run_algorithm(self.params)
        hhl_solution = result["solution_hhl"]
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)
        # linear algebra solution
        linalg_solution = np.linalg.solve(matrix, vector)
        linalg_normed = linalg_solution/np.linalg.norm(linalg_solution)

        # compare result
        fidelity = abs(linalg_normed.dot(hhl_normed.conj()))**2
        np.testing.assert_approx_equal(fidelity, 1, significant=5)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(linalg_solution))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(result["probability_result"]))

    @parameterized.expand([[[0, 1]], [[1, 0]], [[1, 1]], [[1, 10]]])
    def test_hhl_diagonal_longdivison_sv(self, vector):
        self.log.debug('Testing HHL simple test in mode LongDivision and '
                       'statevector simulator')

        ld_params = self.params
        matrix = [[1, 0], [0, 1]]
        ld_params["input"] = {
            "name": "LinearSystemInput",
            "matrix": matrix,
            "vector": vector
        }
        ld_params["reciprocal"]["name"] = "LongDivision"
        ld_params["reciprocal"]["scale"] = 1.0

        # run hhl
        result = run_algorithm(ld_params)
        hhl_solution = result["solution_hhl"]
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)
        # linear algebra solution
        linalg_solution = np.linalg.solve(matrix, vector)
        linalg_normed = linalg_solution/np.linalg.norm(linalg_solution)

        # compare result
        fidelity = abs(linalg_normed.dot(hhl_normed.conj()))**2
        np.testing.assert_approx_equal(fidelity, 1, significant=5)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(linalg_solution))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(result["probability_result"]))

    @parameterized.expand([[[0, 1]], [[1, 0]], [[1, 1]], [[1, 10]]])
    def test_hhl_diagonal_qasm(self, vector):
        self.log.debug('Testing HHL simple test with qasm simulator')

        qasm_params = self.params
        matrix = [[1, 0], [0, 1]]
        qasm_params["input"] = {
            "name": "LinearSystemInput",
            "matrix": matrix,
            "vector": vector
        }
        qasm_params["backend"]["name"] = "qasm_simulator"
        qasm_params["backend"]["shots"] = 200

        # run hhl
        result = run_algorithm(qasm_params)
        hhl_solution = result["solution_hhl"]
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)
        # linear algebra solution
        linalg_solution = np.linalg.solve(matrix, vector)
        linalg_normed = linalg_solution/np.linalg.norm(linalg_solution)

        # compare result
        fidelity = abs(linalg_normed.dot(hhl_normed.conj()))**2
        np.testing.assert_approx_equal(fidelity, 1, significant=1)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(linalg_solution))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(result["probability_result"]))

    def test_hhl_circuit_diagonal_2x2(self):
        self.log.debug('Testing HHL simple test with qasm simulator')

        circ_params = self.params
        matrix = [[1, 0], [0, 3]]
        vector = [1, 0]
        circ_params["input"] = {
            "name": "LinearSystemInput",
            "matrix": matrix,
            "vector": vector
        }
        circ_params["algorithm"]["mode"] = "circuit"

        # run hhl
        result = run_algorithm(circ_params)
        gate_count = result["gate_count_total"]
        self.assertEqual(gate_count, 12658)

        self.log.debug('HHL total gate count:       {}'.format(gate_count))

    def test_hhl_circuit_diagonal_8x8(self):
        self.log.debug('Testing HHL simple test with qasm simulator')

        circ_params = self.params
        matrix = [[1, 0, 0, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 0, 0, 1], [0, 0, 3, 0, 0, 0, 0, 0], [0, 0, 0, 4, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 0, 0, 4]]
        vector = [1, 0, 1, 0, 1, 0, 1, 0]
        circ_params["input"] = {
            "name": "LinearSystemInput",
            "matrix": matrix,
            "vector": vector
        }
        circ_params["algorithm"]["mode"] = "circuit"

        # run hhl
        result = run_algorithm(circ_params)
        gate_count = result["gate_count_total"]
        self.assertEqual(gate_count, 288668)

        self.log.debug('HHL total gate count:       {}'.format(gate_count))

    def test_hhl_negative_eigs_sv(self):
        self.log.debug('Testing HHL with matrix with negative eigenvalues')

        neg_params = self.params
        neg_params["eigs"]["num_ancillae"] = 4
        neg_params["eigs"]["negative_evals"] = "true"
        neg_params["reciprocal"]["negative_evals"] = "true"

        n = 2
        matrix = rmg.random_diag(n, eigrange=[-1, 1])
        vector = random(2)

        algo_input = LinearSystemInput()
        algo_input.matrix = matrix
        algo_input.vector = vector

        # run hhl
        result = run_algorithm(neg_params, algo_input)
        hhl_solution = result["solution_hhl"]
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)
        # linear algebra solution
        linalg_solution = np.linalg.solve(matrix, vector)
        linalg_normed = linalg_solution/np.linalg.norm(linalg_solution)

        # compare result
        fidelity = abs(linalg_normed.dot(hhl_normed.conj()))**2
        np.testing.assert_approx_equal(fidelity, 1, significant=3)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(linalg_solution))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(result["probability_result"]))

    def test_hhl_random_hermitian_sv(self):
        self.log.debug('Testing HHL with random hermitian matrix')

        hermitian_params = self.params
        hermitian_params["eigs"]["num_ancillae"] = 4

        n = 2
        matrix = rmg.random_hermitian(n, eigrange=[0, 1])
        vector = random(2)

        algo_input = LinearSystemInput()
        algo_input.matrix = matrix
        algo_input.vector = vector

        # run hhl
        result = run_algorithm(hermitian_params, algo_input)
        hhl_solution = result["solution_hhl"]
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)
        # linear algebra solution
        linalg_solution = np.linalg.solve(matrix, vector)
        linalg_normed = linalg_solution/np.linalg.norm(linalg_solution)

        # compare result
        fidelity = abs(linalg_normed.dot(hhl_normed.conj()))**2
        np.testing.assert_approx_equal(fidelity, 1, significant=2)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(linalg_solution))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(result["probability_result"]))


if __name__ == '__main__':
    unittest.main()

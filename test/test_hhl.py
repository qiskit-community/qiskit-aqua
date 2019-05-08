# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest

import numpy as np
from numpy.random import random
from parameterized import parameterized
from test.common import QiskitAquaTestCase
from qiskit.aqua import run_algorithm
from qiskit.aqua.input import LinearSystemInput
from qiskit.aqua.utils import random_matrix_generator as rmg
from qiskit.quantum_info import state_fidelity


class TestHHL(QiskitAquaTestCase):
    """HHL tests."""

    def setUp(self):
        super(TestHHL, self).setUp()
        self.els_params = {
            'algorithm': {
                'name': 'ExactLSsolver'
            },
            'problem': {
                'name': 'linear_system',
                'random_seed': 0
            }
        }
        self.params = {
            'problem': {
                'name': 'linear_system',
                'random_seed': 0
            },
            'algorithm': {
                'name': 'HHL'
            },
            'eigs': {
                'expansion_mode': 'suzuki',
                'expansion_order': 2,
                'name': 'EigsQPE',
                'negative_evals': False,
                'num_ancillae': 3,
                'num_time_slices': 50
            },
            'reciprocal': {
                'name': 'Lookup',
                'negative_evals': False,
                'scale': 0.0
            },
            'backend': {
                'provider': 'qiskit.BasicAer',
                'name': 'statevector_simulator',
                'skip_transpiler': False
            }
        }

    @parameterized.expand([[[0, 1]], [[1, 0]], [[1, 1]], [[1, 10]]])
    def test_hhl_diagonal(self, vector):
        self.log.debug('Testing HHL simple test in mode Lookup with '
                       'statevector simulator')

        matrix = [[1, 0], [0, 1]]
        self.params['input'] = {
            'name': 'LinearSystemInput',
            'matrix': matrix,
            'vector': vector
        }

        # run ExactLSsolver
        self.els_params['input'] = self.params['input']
        ref_result = run_algorithm(self.els_params)
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)
        # run hhl
        hhl_result = run_algorithm(self.params)
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare results
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=5)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(ref_solution))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(hhl_result["probability_result"]))

    @parameterized.expand([[[-1, 0]], [[0, -1]], [[-1, -1]]])
    def test_hhl_diagonal_negative(self, vector):
        self.log.debug('Testing HHL simple test in mode Lookup with '
                       'statevector simulator')

        neg_params = self.params
        matrix = [[1, 0], [0, 1]]
        neg_params['input'] = {
            'name': 'LinearSystemInput',
            'matrix': matrix,
            'vector': vector
        }
        neg_params['eigs']['negative_evals'] = True
        neg_params['reciprocal']['negative_evals'] = True
        neg_params['eigs']['num_ancillae'] = 4

        # run ExactLSsolver
        self.els_params['input'] = neg_params['input']
        ref_result = run_algorithm(self.els_params)
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)
        # run hhl
        hhl_result = run_algorithm(neg_params)
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare results
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=5)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(ref_normed))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(hhl_result["probability_result"]))

    @parameterized.expand([[[0, 1]], [[1, 0]], [[1, 1]], [[1, 10]]])
    def test_hhl_diagonal_longdivison(self, vector):
        self.log.debug('Testing HHL simple test in mode LongDivision and '
                       'statevector simulator')

        ld_params = self.params
        matrix = [[1, 0], [0, 1]]
        ld_params['input'] = {
            'name': 'LinearSystemInput',
            'matrix': matrix,
            'vector': vector
        }
        ld_params['reciprocal']['name'] = 'LongDivision'
        ld_params['reciprocal']['scale'] = 1.0

        # run ExactLSsolver
        self.els_params['input'] = ld_params['input']
        ref_result = run_algorithm(self.els_params)
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)
        # run hhl
        hhl_result = run_algorithm(ld_params)
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare results
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=5)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(ref_normed))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(hhl_result["probability_result"]))

    @parameterized.expand([[[0, 1]], [[1, 0]], [[1, 1]], [[1, 10]]])
    def test_hhl_diagonal_qasm(self, vector):
        self.log.debug('Testing HHL simple test with qasm simulator')

        qasm_params = self.params
        matrix = [[1, 0], [0, 1]]
        qasm_params['input'] = {
            'name': 'LinearSystemInput',
            'matrix': matrix,
            'vector': vector
        }
        qasm_params['reciprocal']['scale'] = 0.5
        qasm_params['backend']['name'] = 'qasm_simulator'
        qasm_params['backend']['shots'] = 1000

        # run ExactLSsolver
        self.els_params['input'] = qasm_params['input']
        ref_result = run_algorithm(self.els_params)
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)
        # run hhl
        hhl_result = run_algorithm(qasm_params)
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare results
        fidelity = state_fidelity(ref_normed, hhl_normed)
        self.assertGreater(fidelity, 0.8)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(ref_normed))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(hhl_result["probability_result"]))

    @parameterized.expand([[3, 4], [5, 5]])
    def test_hhl_diagonal_other_dim(self, n, num_ancillary):
        self.log.debug('Testing HHL with matrix dimension other than 2**n')

        dim_params = self.params
        dim_params['eigs']['num_ancillae'] = num_ancillary
        dim_params['eigs']['negative_evals'] = True
        dim_params['reciprocal']['negative_evals'] = True

        np.random.seed(0)
        matrix = rmg.random_diag(n, eigrange=[0, 1])
        vector = random(n)

        algo_input = LinearSystemInput()
        algo_input.matrix = matrix
        algo_input.vector = vector

        # run ExactLSsolver
        ref_result = run_algorithm(self.els_params, algo_input)
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)
        # run hhl
        hhl_result = run_algorithm(dim_params, algo_input)
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare result
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=1)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(ref_solution))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(hhl_result["probability_result"]))

    def test_hhl_negative_eigs(self):
        self.log.debug('Testing HHL with matrix with negative eigenvalues')

        neg_params = self.params
        neg_params['eigs']['num_ancillae'] = 4
        neg_params['eigs']['negative_evals'] = True
        neg_params['reciprocal']['negative_evals'] = True

        n = 2
        np.random.seed(0)
        matrix = rmg.random_diag(n, eigrange=[-1, 1])
        vector = random(n)

        algo_input = LinearSystemInput()
        algo_input.matrix = matrix
        algo_input.vector = vector

        # run ExactLSsolver
        ref_result = run_algorithm(self.els_params, algo_input)
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)
        # run hhl
        hhl_result = run_algorithm(neg_params, algo_input)
        hhl_solution = hhl_result["solution"]
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare results
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=3)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(ref_normed))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(hhl_result["probability_result"]))

    def test_hhl_random_hermitian(self):
        self.log.debug('Testing HHL with random hermitian matrix')

        hermitian_params = self.params
        hermitian_params['eigs']['num_ancillae'] = 4

        n = 2
        np.random.seed(0)
        matrix = rmg.random_hermitian(n, eigrange=[0, 1])
        vector = random(n)

        algo_input = LinearSystemInput()
        algo_input.matrix = matrix
        algo_input.vector = vector

        # run ExactLSsolver
        ref_result = run_algorithm(self.els_params, algo_input)
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)
        # run hhl
        hhl_result = run_algorithm(hermitian_params, algo_input)
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare result
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=2)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(ref_normed))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(hhl_result["probability_result"]))

    def test_hhl_non_hermitian(self):
        self.log.debug('Testing HHL with simple non-hermitian matrix')

        nonherm_params = self.params
        nonherm_params['eigs']['num_ancillae'] = 6
        nonherm_params['eigs']['num_time_slices'] = 80
        nonherm_params['eigs']['negative_evals'] = True
        nonherm_params['reciprocal']['negative_evals'] = True

        matrix = [[1, 1], [2, 1]]
        vector = [1, 0]

        algo_input = LinearSystemInput()
        algo_input.matrix = matrix
        algo_input.vector = vector

        # run ExactLSsolver
        ref_result = run_algorithm(self.els_params, algo_input)
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)
        # run hhl
        hhl_result = run_algorithm(nonherm_params, algo_input)
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)
        # compare result
        fidelity = state_fidelity(ref_normed, hhl_normed)
        self.assertGreater(fidelity, 0.8)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(ref_solution))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(hhl_result["probability_result"]))

if __name__ == '__main__':
    unittest.main()

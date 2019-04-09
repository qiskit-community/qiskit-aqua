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
from qiskit.aqua import run_algorithm
from qiskit.aqua.input import LinearSystemInput
from qiskit.aqua.utils import random_matrix_generator as rmg
from qiskit.quantum_info import state_fidelity


class TestHHL(QiskitAquaTestCase):
    """HHL tests."""

    def setUp(self):
        super(TestHHL, self).setUp()
        np.random.seed(0)
        self.elp_params = {
            'algorithm': {
                'name': 'ExactLPsolver'
            },
            'problem': {
                'name': 'linear_system',
                'random_seed': 0
            }
        }
        self.params = {
            'problem': {
                'name': 'linear_system',
                'circuit_caching': False
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

    def test_hhl_matrix_other_dim(self):
        self.log.debug('Testing HHL with matrix dimension other than 2**n')

        mat_params = self.params
        mat_params['eigs']['num_ancillae'] = 4
        mat_params['eigs']['negative_evals'] = True
        mat_params['reciprocal']['negative_evals'] = True
        mat_params['algorithm']['auto_resize'] = True

        n = 3
        matrix = rmg.random_diag(n, eigrange=[0, 1])
        vector = random(n)

        algo_input = LinearSystemInput()
        algo_input.matrix = matrix
        algo_input.vector = vector

        # run ExactLPsolver
        ref_result = run_algorithm(self.elp_params, algo_input)
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)
        # run hhl
        hhl_result = run_algorithm(mat_params, algo_input)
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare result
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=3)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(ref_solution))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(hhl_result["probability_result"]))

    def test_hhl_hermitian(self):
        self.log.debug('Testing HHL with hermitian matrix')

        herm_params = self.params
        herm_params['eigs']['num_ancillae'] = 6
        #herm_params['eigs']['expansion_order'] = 3
        herm_params['eigs']['num_time_slices'] = 80
        herm_params['eigs']['negative_evals'] = True
        herm_params['reciprocal']['negative_evals'] = True
        #herm_params['reciprocal']['scale'] = 0.5

        matrix = [[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
                  [0.+0.j, 0.+0.j, 2.+0.j, 1.+0.j],
                  [1.+0.j, 2.+0.j, 0.+0.j, 0.+0.j],
                  [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]]
        vector = [1, 0, 1, 0]

        algo_input = LinearSystemInput()
        algo_input.matrix = matrix
        algo_input.vector = vector

        # run ExactLPsolver
        ref_result = run_algorithm(self.elp_params, algo_input)
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)
        # run hhl
        hhl_result = run_algorithm(herm_params, algo_input)
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)
        print(ref_result)
        # compare result
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=2)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(ref_solution))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(hhl_result["probability_result"]))

    def test_hhl_non_hermitian(self):
        self.log.debug('Testing HHL with simple non-hermitian matrix')

        herm_params = self.params
        herm_params['eigs']['num_ancillae'] = 4
        herm_params['reciprocal']['negative_evals'] = True
        herm_params['algorithm']['auto_hermitian'] = True

        matrix = [[1, 1], [0, 1]]
        vector = [1, 0]

        algo_input = LinearSystemInput()
        algo_input.matrix = matrix
        algo_input.vector = vector

        # run ExactLPsolver
        ref_result = run_algorithm(self.elp_params, algo_input)
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)
        # run hhl
        hhl_result = run_algorithm(herm_params, algo_input)
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare result
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=2)

        self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector: {}'.format(ref_solution))
        self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
        self.log.debug('probability of result:     {}'.
                       format(hhl_result["probability_result"]))

    # def test_hhl_random_non_hermitian(self):
    #     self.log.debug('Testing HHL with random non-hermitian matrix')
    #
    #     herm_params = self.params
    #     herm_params['eigs']['num_ancillae'] = 5
    #     herm_params['reciprocal']['negative_evals'] = True
    #     herm_params['algorithm']['auto_hermitian'] = True
    #
    #     n = 2
    #     matrix = rmg.random_non_hermitian(n, srange=[0, 1])
    #     vector = random(n)
    #
    #     algo_input = LinearSystemInput()
    #     algo_input.matrix = matrix
    #     algo_input.vector = vector
    #
    #     # run ExactLPsolver
    #     ref_result = run_algorithm(self.elp_params, algo_input)
    #     ref_solution = ref_result['solution']
    #     ref_normed = ref_solution/np.linalg.norm(ref_solution)
    #     # run hhl
    #     hhl_result = run_algorithm(herm_params, algo_input)
    #     hhl_solution = hhl_result['solution']
    #     hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)
    #
    #     # compare result
    #     fidelity = state_fidelity(ref_normed, hhl_normed)
    #     np.testing.assert_approx_equal(fidelity, 1, significant=2)
    #
    #     self.log.debug('HHL solution vector:       {}'.format(hhl_solution))
    #     self.log.debug('algebraic solution vector: {}'.format(ref_solution))
    #     self.log.debug('fidelity HHL to algebraic: {}'.format(fidelity))
    #     self.log.debug('probability of result:     {}'.
    #                    format(hhl_result["probability_result"]))

if __name__ == '__main__':
    unittest.main()

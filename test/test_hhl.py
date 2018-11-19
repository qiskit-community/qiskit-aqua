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
from parameterized import parameterized

from test.common import QiskitAquaTestCase
from qiskit_aqua import run_algorithm

class TestHHL(QiskitAquaTestCase):
    """HHL tests."""

    def test_hhl_direct(self):
        params = {
            "problem": {
                "name": "linear_system",
            },
            "algorithm": {
                "mode": "state_tomography",
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

        matrix = [[1, 0], [0, 3]]
        vector = [1, 0]
        params["input"] = {
            "name": "LinearSystemInput",
            "matrix": matrix,
            "vector": vector
        }
        self.algorithm = 'HHL'
        self.log.debug('Testing HHL')

        # run hhl
        result = run_algorithm(params)

        # compare result
        linalg_solution = np.linalg.solve(matrix, vector)
        hhl_solution = result["solution_scaled"]
        fidelity = abs(hhl_solution.conj().dot(linalg_solution))**2

        np.testing.assert_approx_equal(fidelity, 1, significant=5)
        # np.testing.assert_approx_equal(classical_solution, hhl_solution,
        #                               significant=5)

        self.log.debug('HHL solution vector:        {}'.format(hhl_solution))
        self.log.debug('algebraic solution vector:  {}'.format(linalg_solution))
        self.log.debug('fidelity HHL to algebraic:  {}'.format(fidelity))

if __name__ == '__main__':
    unittest.main()
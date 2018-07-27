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

import unittest
import numpy as np

from test.common import QiskitAquaTestCase
from qiskit_aqua import Operator, run_algorithm
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import get_algorithm_instance


class TestExactEignesolver(QiskitAquaTestCase):

    def setUp(self):
        np.random.seed(50)
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        qubitOp = Operator.load_from_dict(pauli_dict)
        self.algo_input = get_input_instance('EnergyInput')
        self.algo_input.qubit_op = qubitOp

    def test_ee_via_run_algorithm(self):
        params = {
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, self.algo_input)
        self.assertAlmostEqual(result['energy'], -1.85727503)
        np.testing.assert_array_almost_equal(result['energies'], [-1.85727503])
        np.testing.assert_array_almost_equal(result['eigvals'], [-1.85727503+0j])

    def test_ee_via_run_algorithm_k4(self):
        params = {
            'algorithm': {'name': 'ExactEigensolver', 'k': 4}
        }
        result = run_algorithm(params, self.algo_input)
        self.assertAlmostEqual(result['energy'], -1.85727503)
        self.assertEqual(len(result['eigvals']), 4)
        self.assertEqual(len(result['eigvecs']), 4)
        np.testing.assert_array_almost_equal(result['energies'], [-1.85727503, -1.24458455, -0.88272215, -0.22491125])

    def test_ee_direct(self):
        algo = get_algorithm_instance('ExactEigensolver')
        algo.init_args(self.algo_input.qubit_op, k=1, aux_operators=[])
        result = algo.run()
        self.assertAlmostEqual(result['energy'], -1.85727503)
        np.testing.assert_array_almost_equal(result['energies'], [-1.85727503])
        np.testing.assert_array_almost_equal(result['eigvals'], [-1.85727503+0j])

    def test_ee_direct_k4(self):
        algo = get_algorithm_instance('ExactEigensolver')
        algo.init_args(self.algo_input.qubit_op, k=4, aux_operators=[])
        result = algo.run()
        self.assertAlmostEqual(result['energy'], -1.85727503)
        self.assertEqual(len(result['eigvals']), 4)
        self.assertEqual(len(result['eigvecs']), 4)
        np.testing.assert_array_almost_equal(result['energies'], [-1.85727503, -1.24458455, -0.88272215, -0.22491125])


if __name__ == '__main__':
    unittest.main()

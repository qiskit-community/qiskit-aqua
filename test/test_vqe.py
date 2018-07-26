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
from parameterized import parameterized

from test.common import QiskitAquaTestCase
from qiskit_aqua import Operator, run_algorithm
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import get_algorithm_instance, get_initial_state_instance, \
                         get_variational_form_instance, get_optimizer_instance


class TestVQE(QiskitAquaTestCase):

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

    def test_vqe_via_run_algorithm(self):
        params = {
            'algorithm': {'name': 'VQE'},
            'backend': {'name': 'local_statevector_simulator'}
        }
        result = run_algorithm(params, self.algo_input)
        self.assertAlmostEqual(result['energy'], -1.85727503)
        np.testing.assert_array_almost_equal(result['eigvals'], [-1.85727503])
        np.testing.assert_array_almost_equal(result['opt_params'], [-0.58294401, -1.86141794, -1.97209632, -0.54796022,
                                                                    -0.46945572, 2.60114794, -1.15637845,  1.40498879,
                                                                    1.14479635, -0.48416694, -0.66608349, -1.1367579 ,
                                                                    -2.67097002,  3.10214631,  3.10000313, 0.37235089])
        self.assertIn('eval_count', result)
        self.assertIn('eval_time', result)

    @parameterized.expand([
        ['CG', 5],
        ['COBYLA', 5],
        ['L_BFGS_B', 5],
        ['NELDER_MEAD', 5],
        ['POWELL', 5],
        ['SLSQP', 5],
        ['SPSA', 5],
        ['TNC', 2]
    ])
    def test_vqe_optimzers(self, name, places):
        params = {
            'algorithm': {'name': 'VQE'},
            'optimizer': {'name': name},
            'backend': {'name': 'local_statevector_simulator'}
        }
        result = run_algorithm(params, self.algo_input)
        self.assertAlmostEqual(result['energy'], -1.85727503, places=places)

    @parameterized.expand([
        ['RY', 5],
        ['RYRZ', 5]
    ])
    def test_vqe_var_forms(self, name, places):
        params = {
            'algorithm': {'name': 'VQE'},
            'variational_form': {'name': name},
            'backend': {'name': 'local_statevector_simulator'}
        }
        result = run_algorithm(params, self.algo_input)
        self.assertAlmostEqual(result['energy'], -1.85727503, places=places)

    def test_vqe_direct(self):
        num_qbits = self.algo_input.qubit_op.num_qubits
        init_state = get_initial_state_instance('ZERO')
        init_state.init_args(num_qbits)
        var_form = get_variational_form_instance('RY')
        var_form.init_args(num_qbits, 3, initial_state=init_state)
        optimizer = get_optimizer_instance('L_BFGS_B')
        optimizer.init_args()
        algo = get_algorithm_instance('VQE')
        algo.setup_quantum_backend(backend='local_statevector_simulator')
        algo.init_args(self.algo_input.qubit_op, 'matrix', var_form, optimizer)
        result = algo.run()
        self.assertAlmostEqual(result['energy'], -1.85727503)


if __name__ == '__main__':
    unittest.main()

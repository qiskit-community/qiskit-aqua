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

from test.common import QISKitAcquaTestCase
from qiskit_acqua import Operator, run_algorithm
from qiskit_acqua.input import get_input_instance
from qiskit_acqua import get_algorithm_instance, get_initial_state_instance, \
                         get_variational_form_instance, get_optimizer_instance
from qiskit_acqua.utils import decimal_to_binary


class TestVQE(QISKitAcquaTestCase):

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

    def test_vqe_direct(self):
        num_qbits = self.algo_input.qubit_op.num_qubits
        init_state = get_initial_state_instance('ZERO')
        init_state.init_args(num_qbits)
        var_form = get_variational_form_instance('RY')
        var_form.init_args(num_qbits, 3, initial_state=init_state)
        optimizer = get_optimizer_instance('L_BFGS_B')
        optimizer.init_args()
        optimizer.set_options(**{'maxfun': 20})
        algo = get_algorithm_instance('VQE')
        algo.setup_quantum_backend(backend='local_statevector_simulator')
        algo.init_args(self.algo_input.qubit_op, 'matrix', var_form, optimizer)
        result = algo.run()

        print(result)

        self.ref_eigenval = -1.85727503


        num_time_slices = 50
        num_iterations = 12

        iqpe = get_algorithm_instance('IQPE')
        iqpe.setup_quantum_backend(backend='local_qasm_simulator', shots=100, skip_transpiler=True)

        state_in = get_initial_state_instance('VarForm')
        state_in.init_args(var_form, result['opt_params'])

        iqpe.init_args(
            self.algo_input.qubit_op, state_in, num_time_slices, num_iterations,
            paulis_grouping='random',
            expansion_mode='suzuki',
            expansion_order=2,
        )

        result = iqpe.run()
        # self.log.debug('operator paulis:\n{}'.format(self.qubitOp.print_operators('paulis')))
        # self.log.debug('qpe circuit:\n\n{}'.format(result['circuit']['complete'].qasm()))

        print('top result str label:         {}'.format(result['top_measurement_label']))
        print('top result in decimal:        {}'.format(result['top_measurement_decimal']))
        print('stretch:                      {}'.format(result['stretch']))
        print('translation:                  {}'.format(result['translation']))
        print('final eigenvalue from QPE:    {}'.format(result['energy']))
        print('reference eigenvalue:         {}'.format(self.ref_eigenval))
        print('ref eigenvalue (transformed): {}'.format(
            (self.ref_eigenval + result['translation']) * result['stretch'])
        )
        print('reference binary str label:   {}'.format(decimal_to_binary(
            (self.ref_eigenval + result['translation']) * result['stretch'],
            max_num_digits=num_iterations + 3,
            fractional_part_only=True
        )))

        np.testing.assert_approx_equal(self.ref_eigenval, result['energy'], significant=2)



if __name__ == '__main__':
    unittest.main()

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
from qiskit.transpiler import PassManager

from test.common import QiskitAquaTestCase
from qiskit.aqua import get_aer_backend
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.utils import decimal_to_binary
from qiskit.aqua.components.initial_states import VarFormBased
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.algorithms import IQPE


class TestVQE2IQPE(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        self.random_seed = 0
        np.random.seed(self.random_seed)
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        qubit_op = Operator.load_from_dict(pauli_dict)
        self.algo_input = EnergyInput(qubit_op)

    def test_vqe_2_iqpe(self):
        backend = get_aer_backend('qasm_simulator')
        num_qbits = self.algo_input.qubit_op.num_qubits
        var_form = RYRZ(num_qbits, 3)
        optimizer = SPSA(max_trials=10)
        # optimizer.set_options(**{'max_trials': 500})
        algo = VQE(self.algo_input.qubit_op, var_form, optimizer, 'paulis')
        quantum_instance = QuantumInstance(backend)
        result = algo.run(quantum_instance)

        self.log.debug('VQE result: {}.'.format(result))

        self.ref_eigenval = -1.85727503

        num_time_slices = 50
        num_iterations = 11

        state_in = VarFormBased(var_form, result['opt_params'])
        iqpe = IQPE(self.algo_input.qubit_op, state_in, num_time_slices, num_iterations,
                    expansion_mode='suzuki', expansion_order=2, shallow_circuit_concat=True)
        quantum_instance = QuantumInstance(backend, shots=100, pass_manager=PassManager(), seed_mapper=self.random_seed)
        result = iqpe.run(quantum_instance)

        self.log.debug('top result str label:         {}'.format(result['top_measurement_label']))
        self.log.debug('top result in decimal:        {}'.format(result['top_measurement_decimal']))
        self.log.debug('stretch:                      {}'.format(result['stretch']))
        self.log.debug('translation:                  {}'.format(result['translation']))
        self.log.debug('final eigenvalue from QPE:    {}'.format(result['energy']))
        self.log.debug('reference eigenvalue:         {}'.format(self.ref_eigenval))
        self.log.debug('ref eigenvalue (transformed): {}'.format(
            (self.ref_eigenval + result['translation']) * result['stretch'])
        )
        self.log.debug('reference binary str label:   {}'.format(decimal_to_binary(
            (self.ref_eigenval + result['translation']) * result['stretch'],
            max_num_digits=num_iterations + 3,
            fractional_part_only=True
        )))

        np.testing.assert_approx_equal(self.ref_eigenval, result['energy'], significant=2)


if __name__ == '__main__':
    unittest.main()

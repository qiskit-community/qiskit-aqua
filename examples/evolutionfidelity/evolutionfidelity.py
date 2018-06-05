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
"""
The Fidelity of Quantum Dynamics.
This is a toy example to show how to build an algorithm under the current lib stack.
The entire parent directory 'evolutionfidelity' is to be moved under the 'qiskit_acqua' directory for demo.
"""

import logging
import numpy as np
from qiskit import QuantumRegister
from qiskit.tools.qi.qi import state_fidelity

from qiskit_acqua import QuantumAlgorithm
from qiskit_acqua import AlgorithmError
from qiskit_acqua import get_initial_state_instance


logger = logging.getLogger(__name__)


class EvolutionFidelity(QuantumAlgorithm):
    """The Toy Demo EvolutionFidelity algorithm."""
    PROP_EXPANSION_ORDER = 'expansion_order'

    def __init__(self, configuration=None):
        """
        Args:
            configuration (dict): algorithm configuration
        """
        super(EvolutionFidelity, self).__init__(configuration)
        if configuration is None:
            self._configuration = {
                'name': 'EvolutionFidelity',
                'description': 'Toy Demo EvolutionFidelity Algorithm for Quantum Systems',
                'input_schema': {
                    '$schema': 'http://json-schema.org/schema#',
                    'id': 'evolution_fidelity_schema',
                    'type': 'object',
                    'properties': {
                        EvolutionFidelity.PROP_EXPANSION_ORDER: {
                            'type': 'integer',
                            'default': 1,
                            'minimum': 1
                        },
                    },
                    'additionalProperties': False
                },
                'problems': []
            }
        else:
            self._configuration = configuration

    def init_params(self, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: EnergyInput instance
        """
        if algo_input is None:
            raise AlgorithmError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        evolution_fidelity_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        expansion_order = evolution_fidelity_params.get(EvolutionFidelity.PROP_EXPANSION_ORDER)

        # Set up initial state, we need to add computed num qubits to params
        initial_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        initial_state_params['num_qubits'] = operator.num_qubits
        initial_state = get_initial_state_instance(initial_state_params['name'])
        initial_state.init_params(initial_state_params)

        self.init_args(operator, initial_state, expansion_order)

    def init_args(self, operator, initial_state, expansion_order):
        self._operator = operator
        self._initial_state = initial_state
        self._expansion_order = expansion_order
        self._ret = {}

    def run(self):
        evo_time = 1
        # get the groundtruth via simple matrix * vector
        state_out_exact = self._operator.evolve(self._initial_state.construct_circuit('vector'), evo_time, 'matrix', 0)

        qr = QuantumRegister(self._operator.num_qubits, name='q')
        circuit = self._initial_state.construct_circuit('circuit', qr)
        circuit += self._operator.evolve(
            None, evo_time, 'circuit', 1,
            quantum_registers=qr,
            expansion_mode='suzuki',
            expansion_order=self._expansion_order
        )

        result = self.execute(circuit)
        state_out_dynamics = np.asarray(result.get_statevector(circuit))

        self._ret['score'] = state_fidelity(state_out_exact, state_out_dynamics)

        return self._ret

# -- end class

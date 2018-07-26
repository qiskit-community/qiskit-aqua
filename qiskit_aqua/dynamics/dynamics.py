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
The Quantum Dynamics algorithm.
"""

import logging

from qiskit import QuantumRegister

from qiskit_aqua import QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_initial_state_instance

logger = logging.getLogger(__name__)


class Dynamics(QuantumAlgorithm):
    """
    The Quantum Dynamics algorithm.
    """

    PROP_OPERATOR_MODE = 'operator_mode'
    PROP_EVO_TIME = 'evo_time'
    PROP_NUM_TIME_SLICES = 'num_time_slices'
    PROP_PAULIS_GROUPING = 'paulis_grouping'
    PROP_EXPANSION_MODE = 'expansion_mode'
    PROP_EXPANSION_ORDER = 'expansion_order'

    DYNAMICS_CONFIGURATION = {
        'name': 'Dynamics',
        'description': 'Dynamics for Quantum Systems',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'Dynamics_schema',
            'type': 'object',
            'properties': {
                PROP_OPERATOR_MODE: {
                    'type': 'string',
                    'default': 'paulis',
                    'oneOf': [
                        {'enum': [
                            'paulis',
                            'grouped_paulis',
                            'matrix'
                        ]}
                    ]
                },
                PROP_EVO_TIME: {
                    'type': 'number',
                    'default': 1,
                    'minimum': 0
                },
                PROP_NUM_TIME_SLICES: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
                PROP_PAULIS_GROUPING: {
                    'type': 'string',
                    'default': 'random',
                    'oneOf': [
                        {'enum': [
                            'default',
                            'random'
                        ]}
                    ]
                },
                PROP_EXPANSION_MODE: {
                    'type': 'string',
                    'default': 'trotter',
                    'oneOf': [
                        {'enum': [
                            'trotter',
                            'suzuki'
                        ]}
                    ]
                },
                PROP_EXPANSION_ORDER: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['dynamics'],
        'depends': ['initial_state'],
        'defaults': {
            'initial_state': {
                'name': 'ZERO'
            }
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DYNAMICS_CONFIGURATION.copy())
        self._operator = None
        self._operator_mode = None
        self._initial_state = None
        self._evo_operator = None
        self._evo_time = 0
        self._num_time_slices = 0
        self._paulis_grouping = None
        self._expansion_mode = None
        self._expansion_order = None
        self._ret = {}

    def init_params(self, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: EnergyInput instance
        """
        if algo_input is None:
            raise AlgorithmError("EnergyInput instance is required.")

        # For getting the extra operator, caller has to do something like: algo_input.add_aux_op(evo_op)
        operator = algo_input.qubit_op
        aux_ops = algo_input.aux_ops
        if aux_ops is None or len(aux_ops) != 1:
            raise AlgorithmError("EnergyInput, a single aux op is required for evaluation.")
        evo_operator = aux_ops[0]
        if evo_operator is None:
            raise AlgorithmError("EnergyInput, invalid aux op.")

        dynamics_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        operator_mode = dynamics_params.get(Dynamics.PROP_OPERATOR_MODE)
        evo_time = dynamics_params.get(Dynamics.PROP_EVO_TIME)
        num_time_slices = dynamics_params.get(Dynamics.PROP_NUM_TIME_SLICES)
        paulis_grouping = dynamics_params.get(Dynamics.PROP_PAULIS_GROUPING)
        expansion_mode = dynamics_params.get(Dynamics.PROP_EXPANSION_MODE)
        expansion_order = dynamics_params.get(Dynamics.PROP_EXPANSION_ORDER)

        # Set up initial state, we need to add computed num qubits to params
        initial_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        initial_state_params['num_qubits'] = operator.num_qubits
        initial_state = get_initial_state_instance(initial_state_params['name'])
        initial_state.init_params(initial_state_params)

        self.init_args(
            operator, operator_mode, initial_state, evo_operator, evo_time, num_time_slices,
            paulis_grouping=paulis_grouping, expansion_mode=expansion_mode, expansion_order=expansion_order
        )

    def init_args(
            self, operator, operator_mode, initial_state, evo_operator, evo_time, num_time_slices,
            paulis_grouping='default', expansion_mode='trotter', expansion_order=1):
        self._operator = operator
        self._operator_mode = operator_mode
        self._initial_state = initial_state
        self._evo_operator = evo_operator
        self._evo_time = evo_time
        self._num_time_slices = num_time_slices
        self._paulis_grouping = paulis_grouping
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._ret = {}

    def run(self):
        quantum_registers = QuantumRegister(self._operator.num_qubits, name='q')
        qc = self._initial_state.construct_circuit('circuit', quantum_registers)

        qc.data += self._evo_operator.evolve(
            None,
            self._evo_time,
            'circuit',
            self._num_time_slices,
            quantum_registers=quantum_registers,
            paulis_grouping=self._paulis_grouping,
            expansion_mode=self._expansion_mode,
            expansion_order=self._expansion_order,
        ).data

        self._ret['avg'], self._ret['std_dev'] = self._operator.eval(self._operator_mode, qc, self._backend)
        return self._ret

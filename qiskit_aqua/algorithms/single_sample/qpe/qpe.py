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
The Quantum Phase Estimation Algorithm.
"""

import copy
import logging
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit_aqua.algorithms.components.phase_estimation import PhaseEstimation
from qiskit_aqua import Operator, QuantumAlgorithm, AquaError
from qiskit_aqua import PluggableType, get_pluggable_class


logger = logging.getLogger(__name__)


class QPE(QuantumAlgorithm):
    """The Quantum Phase Estimation algorithm."""

    PROP_NUM_TIME_SLICES = 'num_time_slices'
    PROP_PAULIS_GROUPING = 'paulis_grouping'
    PROP_EXPANSION_MODE = 'expansion_mode'
    PROP_EXPANSION_ORDER = 'expansion_order'
    PROP_NUM_ANCILLAE = 'num_ancillae'

    CONFIGURATION = {
        'name': 'QPE',
        'description': 'Quantum Phase Estimation for Quantum Systems',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'qpe_schema',
            'type': 'object',
            'properties': {
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
                            'random',
                            'default'
                        ]}
                    ]
                },
                PROP_EXPANSION_MODE: {
                    'type': 'string',
                    'default': 'suzuki',
                    'oneOf': [
                        {'enum': [
                            'suzuki',
                            'trotter'
                        ]}
                    ]
                },
                PROP_EXPANSION_ORDER: {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                PROP_NUM_ANCILLAE: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy'],
        'depends': ['initial_state', 'iqft'],
        'defaults': {
            'initial_state': {
                'name': 'ZERO'
            },
            'iqft': {
                'name': 'STANDARD'
            }
        }
    }

    def __init__(
            self, operator, state_in, iqft, num_time_slices=1, num_ancillae=1,
            paulis_grouping='random', expansion_mode='trotter', expansion_order=1,
            shallow_circuit_concat=False
    ):
        super().__init__()
        super().validate({
            QPE.PROP_NUM_TIME_SLICES: num_time_slices,
            QPE.PROP_PAULIS_GROUPING: paulis_grouping,
            QPE.PROP_EXPANSION_MODE: expansion_mode,
            QPE.PROP_EXPANSION_ORDER: expansion_order,
            QPE.PROP_NUM_ANCILLAE: num_ancillae
        })

        self._ret = {}
        self._operator = copy.deepcopy(operator)
        self._operator.to_paulis()
        self._ret['translation'] = sum([abs(p[0]) for p in self._operator.paulis])
        self._ret['stretch'] = 0.5 / self._ret['translation']

        # translate the operator
        self._operator._simplify_paulis()
        translation_op = Operator([
            [
                self._ret['translation'],
                Pauli(
                    np.zeros(self._operator.num_qubits),
                    np.zeros(self._operator.num_qubits)
                )
            ]
        ])
        translation_op._simplify_paulis()
        self._operator += translation_op

        # stretch the operator
        for p in self._operator._paulis:
            p[0] = p[0] * self._ret['stretch']

        self._phase_estimation_component = PhaseEstimation(
            self._operator, state_in, iqft, num_time_slices=num_time_slices, num_ancillae=num_ancillae,
            paulis_grouping=paulis_grouping, expansion_mode=expansion_mode, expansion_order=expansion_order,
            state_in_circuit_factory=None,
            operator_circuit_factory=None,
            additional_params=None,
            shallow_circuit_concat=shallow_circuit_concat
        )
        self._circuit = None
        self._binary_fractions = [1 / 2 ** p for p in range(1, num_ancillae + 1)]

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: EnergyInput instance
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        qpe_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        num_time_slices = qpe_params.get(QPE.PROP_NUM_TIME_SLICES)
        paulis_grouping = qpe_params.get(QPE.PROP_PAULIS_GROUPING)
        expansion_mode = qpe_params.get(QPE.PROP_EXPANSION_MODE)
        expansion_order = qpe_params.get(QPE.PROP_EXPANSION_ORDER)
        num_ancillae = qpe_params.get(QPE.PROP_NUM_ANCILLAE)

        # Set up initial state, we need to add computed num qubits to params
        init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        init_state_params['num_qubits'] = operator.num_qubits
        init_state = get_pluggable_class(PluggableType.INITIAL_STATE,
                                         init_state_params['name']).init_params(init_state_params)

        # Set up iqft, we need to add num qubits to params which is our num_ancillae bits here
        iqft_params = params.get(QuantumAlgorithm.SECTION_KEY_IQFT)
        iqft_params['num_qubits'] = num_ancillae
        iqft = get_pluggable_class(PluggableType.IQFT, iqft_params['name']).init_params(iqft_params)

        return cls(operator, init_state, iqft, num_time_slices, num_ancillae,
                   paulis_grouping=paulis_grouping, expansion_mode=expansion_mode,
                   expansion_order=expansion_order)

    def _compute_energy(self):
        if QuantumAlgorithm.is_statevector_backend(self.backend):
            raise ValueError('Selected backend does not support measurements.')

        if self._circuit is None:
            self._circuit = self._phase_estimation_component.construct_circuit(measure=True)

        result = self.execute(self._circuit)

        rd = result.get_counts(self._circuit)
        rets = sorted([(rd[k], k) for k in rd])[::-1]
        ret = rets[0][-1][::-1]
        retval = sum([t[0] * t[1] for t in zip(self._binary_fractions, [int(n) for n in ret])])

        self._ret['measurements'] = rets
        self._ret['top_measurement_label'] = ret
        self._ret['top_measurement_decimal'] = retval
        self._ret['energy'] = retval / self._ret['stretch'] - self._ret['translation']

    def run(self):
        self._compute_energy()
        return self._ret

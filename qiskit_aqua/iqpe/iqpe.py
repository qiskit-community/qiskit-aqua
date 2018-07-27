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
The Iterative Quantum Phase Estimation Algorithm.
See https://arxiv.org/abs/quant-ph/0610214
"""

import logging

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.qi.pauli import Pauli

from qiskit_aqua import Operator, QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_initial_state_instance

logger = logging.getLogger(__name__)


class IQPE(QuantumAlgorithm):
    """
    The Iterative Quantum Phase Estimation algorithm.
    See https://arxiv.org/abs/quant-ph/0610214
    """

    PROP_NUM_TIME_SLICES = 'num_time_slices'
    PROP_PAULIS_GROUPING = 'paulis_grouping'
    PROP_EXPANSION_MODE = 'expansion_mode'
    PROP_EXPANSION_ORDER = 'expansion_order'
    PROP_NUM_ITERATIONS = 'num_iterations'

    IQPE_CONFIGURATION = {
        'name': 'IQPE',
        'description': 'Iterative Quantum Phase Estimation for Quantum Systems',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'Dynamics_schema',
            'type': 'object',
            'properties': {
                PROP_NUM_TIME_SLICES: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
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
                PROP_NUM_ITERATIONS: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy'],
        'depends': ['initial_state'],
        'defaults': {
            'initial_state': {
                'name': 'ZERO'
            },
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.IQPE_CONFIGURATION.copy())
        self._operator = None
        self._state_in = None
        self._num_time_slices = 0
        self._paulis_grouping = None
        self._expansion_mode = None
        self._expansion_order = None
        self._num_iterations = 0
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

        operator = algo_input.qubit_op

        iqpe_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        num_time_slices = iqpe_params.get(IQPE.PROP_NUM_TIME_SLICES)
        paulis_grouping = iqpe_params.get(IQPE.PROP_PAULIS_GROUPING)
        expansion_mode = iqpe_params.get(IQPE.PROP_EXPANSION_MODE)
        expansion_order = iqpe_params.get(IQPE.PROP_EXPANSION_ORDER)
        num_iterations = iqpe_params.get(IQPE.PROP_NUM_ITERATIONS)

        # Set up initial state, we need to add computed num qubits to params
        init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        init_state_params['num_qubits'] = operator.num_qubits
        init_state = get_initial_state_instance(init_state_params['name'])
        init_state.init_params(init_state_params)

        self.init_args(
            operator, init_state, num_time_slices, num_iterations,
            paulis_grouping=paulis_grouping, expansion_mode=expansion_mode,
            expansion_order=expansion_order)

    def init_args(self, operator, state_in, num_time_slices, num_iterations,
                  paulis_grouping='default', expansion_mode='trotter', expansion_order=1):
        if self._backend.find('statevector') >= 0:
            raise ValueError('Selected backend does not support measurements.')
        self._operator = operator
        self._state_in = state_in
        self._num_time_slices = num_time_slices
        self._num_iterations = num_iterations
        self._paulis_grouping = paulis_grouping
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._ret = {}

    def _construct_kth_evolution(self, slice_pauli_list, k, omega):
        """Construct the kth iteration Quantum Phase Estimation circuit"""
        a = QuantumRegister(1, name='a')
        c = ClassicalRegister(1, name='c')
        q = QuantumRegister(self._operator.num_qubits, name='q')
        qc = QuantumCircuit(a, c, q)
        qc.data += self._state_in.construct_circuit('circuit', q).data
        # hadamard on a[0]
        qc.u2(0, np.pi, a[0])
        # controlled-U
        qc.data += self._operator.construct_evolution_circuit(
            slice_pauli_list, -2 * np.pi, self._num_time_slices, q, a, unitary_power=2 ** (k - 1)
        ).data
        # global phase due to identity pauli
        qc.u1(2 * np.pi * self._ancilla_phase_coef * (2 ** (k - 1)), a[0])
        # rz on a[0]
        qc.u1(omega, a[0])
        # hadamard on a[0]
        qc.u2(0, np.pi, a[0])
        qc.measure(a, c)
        return qc

    def _estimate_phase_iteratively(self):
        """Iteratively construct the different order of controlled evolution circuit to carry out phase estimation"""
        pauli_list = self._operator.reorder_paulis(grouping=self._paulis_grouping)
        if len(pauli_list) == 1:
            slice_pauli_list = pauli_list
        else:
            if self._expansion_mode == 'trotter':
                slice_pauli_list = pauli_list
            else:
                slice_pauli_list = Operator._suzuki_expansion_slice_pauli_list(pauli_list, 1, self._expansion_order)

        self._ret['top_measurement_label'] = ''

        omega_coef = 0
        # k runs from the number of iterations back to 1
        for k in range(self._num_iterations, 0, -1):
            omega_coef /= 2
            qc = self._construct_kth_evolution(slice_pauli_list, k, -2 * np.pi * omega_coef)
            measurements = self.execute(qc).get_counts(qc)

            if '0' not in measurements:
                if '1' in measurements:
                    x = 1
                else:
                    raise RuntimeError('Unexpected measurement {}.'.format(measurements))
            else:
                if '1' not in measurements:
                    x = 0
                else:
                    x = 1 if measurements['1'] > measurements['0'] else 0
            self._ret['top_measurement_label'] = '{}{}'.format(x, self._ret['top_measurement_label'])
            omega_coef = omega_coef + x / 2
            logger.info('Reverse iteration {} of {} with measured bit {}'.format(k, self._num_iterations, x))
        return omega_coef

    def _compute_energy(self):
        self._operator._check_representation('paulis')
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

        # check for identify paulis to get its coef for applying global phase shift on ancilla later
        num_identities = 0
        for p in self._operator.paulis:
            if np.all(p[1].v == 0) and np.all(p[1].w == 0):
                num_identities += 1
                if num_identities > 1:
                    raise RuntimeError('Multiple identity pauli terms are present.')
                self._ancilla_phase_coef = p[0].real if isinstance(p[0], complex) else p[0]

        self._ret['phase'] = self._estimate_phase_iteratively()
        self._ret['top_measurement_decimal'] = sum([t[0] * t[1] for t in zip(
            [1 / 2 ** p for p in range(1, self._num_iterations + 1)],
            [int(n) for n in self._ret['top_measurement_label']]
        )])
        self._ret['energy'] = self._ret['phase'] / self._ret['stretch'] - self._ret['translation']

    def run(self):
        self._compute_energy()
        return self._ret

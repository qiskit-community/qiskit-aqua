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
from qiskit.qasm import pi

from qiskit_acqua import Operator, QuantumAlgorithm, AlgorithmError
from qiskit_acqua import get_initial_state_instance

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

    DEFAULT_PROP_NUM_TIME_SLICES = 1
    DEFAULT_PROP_PAULIS_GROUPING = 'default'        # grouped_paulis
    ALTERNATIVE_PROP_PAULIS_GROUPING = 'random'     # paulis
    DEFAULT_PROP_EXPANSION_MODE = 'trotter'
    ALTERNATIVE_PROP_EXPANSION_MODE = 'suzuki'
    DEFAULT_PROP_EXPANSION_ORDER = 2
    DEFAULT_PROP_NUM_ITERATIONS = 1

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
                    'default': DEFAULT_PROP_NUM_TIME_SLICES,
                    'minimum': 0
                },
                PROP_PAULIS_GROUPING: {
                    'type': 'string',
                    'default': DEFAULT_PROP_PAULIS_GROUPING,
                    'oneOf': [
                        {'enum': [
                            DEFAULT_PROP_PAULIS_GROUPING,
                            ALTERNATIVE_PROP_PAULIS_GROUPING
                        ]}
                    ]
                },
                PROP_EXPANSION_MODE: {
                    'type': 'string',
                    'default': DEFAULT_PROP_EXPANSION_MODE,
                    'oneOf': [
                        {'enum': [
                            DEFAULT_PROP_EXPANSION_MODE,
                            ALTERNATIVE_PROP_EXPANSION_MODE
                        ]}
                    ]
                },
                PROP_EXPANSION_ORDER: {
                    'type': 'integer',
                    'default': DEFAULT_PROP_EXPANSION_ORDER,
                    'minimum': 1
                },
                PROP_NUM_ITERATIONS: {
                    'type': 'integer',
                    'default': DEFAULT_PROP_NUM_ITERATIONS,
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

    def _construct_kth_evolution(self, slice_pauli_list, k, omega, use_basis_gates=True):
        """Construct the kth iteration Quantum Phase Estimation circuit"""
        a = QuantumRegister(1, name='a')
        c = ClassicalRegister(1, name='c')
        q = QuantumRegister(self._operator.num_qubits, name='q')
        qc = QuantumCircuit(a, c, q)
        qc += self._state_in.construct_circuit('circuit', q)
        qc.h(a[0])
        qc += self._operator.construct_evolution_circuit(
            slice_pauli_list, 1, self._num_time_slices, q, a, unitary_power=2**(k-1)
        )
        qc.u1(omega, a[0]) if use_basis_gates else qc.rz(omega, a[0])
        qc.h(a[0])
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

        k, omega_coef = self._num_iterations, 0
        while True:
            qc = self._construct_kth_evolution(slice_pauli_list, k, -2 * pi * omega_coef)
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
            omega_coef = omega_coef / 2 + x / 4
            # print('k:{} measurements:{} x:{} omega:{}'.format(k, measurements, x, omega_coef))
            k -= 1
            if k <= 0:
                return 2 * omega_coef

    def _compute_energy(self):
        bound = sum([abs(p[0].real) for p in self._operator.paulis])
        translation = bound
        stretch = np.pi / bound
        self._ret['phase'] = self._estimate_phase_iteratively()
        self._ret['energy'] = self._ret['phase'] * 2 * np.pi / stretch - translation

    def run(self):
        self._compute_energy()
        return self._ret

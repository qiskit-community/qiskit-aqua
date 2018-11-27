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

from qiskit_aqua import Pluggable
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aqua import Operator, QuantumAlgorithm, AquaError
from qiskit_aqua import PluggableType, get_pluggable_class


class PhaseEstimation(Pluggable):
    """The Quantum Phase Estimation algorithm."""

    PROP_NUM_TIME_SLICES = 'num_time_slices'
    PROP_PAULIS_GROUPING = 'paulis_grouping'
    PROP_EXPANSION_MODE = 'expansion_mode'
    PROP_EXPANSION_ORDER = 'expansion_order'
    PROP_NUM_ANCILLAE = 'num_ancillae'

    CONFIGURATION = {
        'name': 'PhaseEstimation',
        'description': 'Quantum Phase Estimation',
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
            state_in_circuit_factory=None,
            operator_circuit_factory=None,
            additional_params=None,
            shallow_circuit_concat=False):

        super().__init__()
        super().validate({
            PhaseEstimation.PROP_NUM_TIME_SLICES: num_time_slices,
            PhaseEstimation.PROP_PAULIS_GROUPING: paulis_grouping,
            PhaseEstimation.PROP_EXPANSION_MODE: expansion_mode,
            PhaseEstimation.PROP_EXPANSION_ORDER: expansion_order,
            PhaseEstimation.PROP_NUM_ANCILLAE: num_ancillae
        })
        self._operator = operator
        self._operator_circuit_factory = operator_circuit_factory
        self._state_in = state_in
        self._state_in_circuit_factory = state_in_circuit_factory
        self._iqft = iqft
        self._num_time_slices = num_time_slices
        self._num_ancillae = num_ancillae
        self._paulis_grouping = paulis_grouping
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._shallow_circuit_concat = shallow_circuit_concat
        self._ancilla_phase_coef = 1
        self._circuit = {True: None, False: None}
        self._additional_params = additional_params
        self._ret = {}

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
        num_time_slices = qpe_params.get(PhaseEstimation.PROP_NUM_TIME_SLICES)
        paulis_grouping = qpe_params.get(PhaseEstimation.PROP_PAULIS_GROUPING)
        expansion_mode = qpe_params.get(PhaseEstimation.PROP_EXPANSION_MODE)
        expansion_order = qpe_params.get(PhaseEstimation.PROP_EXPANSION_ORDER)
        num_ancillae = qpe_params.get(PhaseEstimation.PROP_NUM_ANCILLAE)

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

    def construct_circuit(self, measure=False):
        """Construct the Phase Estimation circuit"""

        if self._circuit[measure] is None:
            if self._operator is not None:
                # check for identify paulis to get its coef for applying global phase shift on ancillae later
                num_identities = 0
                for p in self._operator.paulis:
                    if np.all(np.logical_not(p[1].z)) and np.all(np.logical_not(p[1].x)):
                        num_identities += 1
                        if num_identities > 1:
                            raise RuntimeError('Multiple identity pauli terms are present.')
                        self._ancilla_phase_coef = p[0].real if isinstance(p[0], complex) else p[0]

            a = QuantumRegister(self._num_ancillae, name='a')
            if self._operator is not None:
                q = QuantumRegister(self._operator.num_qubits, name='q')
            elif self._operator_circuit_factory is not None:
                q = QuantumRegister(self._operator_circuit_factory.num_target_qubits, name='q')
            else:
                raise RuntimeError('Missing operator specification.')
            qc = QuantumCircuit(a, q)

            num_aux_qubits, aux = 0, None
            if self._state_in_circuit_factory is not None:
                num_aux_qubits = self._state_in_circuit_factory.required_ancillas()
            if self._operator_circuit_factory is not None:
                num_aux_qubits = max(num_aux_qubits, self._operator_circuit_factory.required_ancillas_controlled())

            if num_aux_qubits > 0:
                aux = QuantumRegister(num_aux_qubits, name='aux')
                qc.add_register(aux)

            # initialize state_in
            if self._state_in is not None:
                qc.data += self._state_in.construct_circuit('circuit', q).data
            elif self._state_in_circuit_factory is not None:
                self._state_in_circuit_factory.build(qc, q, aux, self._additional_params)
            else:
                raise RuntimeError('Missing initial state specification.')

            # Put all ancillae in uniform superposition
            qc.u2(0, np.pi, a)

            # phase kickbacks via dynamics
            if self._operator is not None:
                pauli_list = self._operator.reorder_paulis(grouping=self._paulis_grouping)
                if len(pauli_list) == 1:
                    slice_pauli_list = pauli_list
                else:
                    if self._expansion_mode == 'trotter':
                        slice_pauli_list = pauli_list
                    elif self._expansion_mode == 'suzuki':
                        slice_pauli_list = Operator._suzuki_expansion_slice_pauli_list(
                            pauli_list,
                            1,
                            self._expansion_order
                        )
                    else:
                        raise ValueError('Unrecognized expansion mode {}.'.format(self._expansion_mode))
                for i in range(self._num_ancillae):
                    qc_evolutions = Operator.construct_evolution_circuit(
                        slice_pauli_list, -2 * np.pi, self._num_time_slices, q, a, ctl_idx=i,
                        shallow_slicing=self._shallow_circuit_concat
                    )
                    if self._shallow_circuit_concat:
                        qc.data += qc_evolutions.data
                    else:
                        qc += qc_evolutions
                    # global phase shift for the ancilla due to the identity pauli term
                    qc.u1(2 * np.pi * self._ancilla_phase_coef * (2 ** i), a[i])
            elif self._operator_circuit_factory is not None:
                for i in range(self._num_ancillae):
                    self._operator_circuit_factory.build_controlled_power(qc, q, a[i], 2 ** i, aux, self._additional_params)

            # inverse qft on ancillae
            self._iqft.construct_circuit('circuit', a, qc)

            # measuring ancillae
            if measure:
                c = ClassicalRegister(self._num_ancillae, name='c')
                qc.add_register(c)
                qc.barrier(a)
                qc.measure(a, c)

            self._circuit[measure] = qc

        return self._circuit[measure]

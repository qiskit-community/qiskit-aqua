
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
The Quantum Phase Estimation Subroutine.
"""

import logging

from functools import reduce
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.tools.qi.pauli import Pauli
from qiskit_aqua import Operator, QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_initial_state_instance, get_iqft_instance

logger = logging.getLogger(__name__)

class QPE():
    """The Quantum Phase Estimation subroutine modified for HHL needs"""

    PROP_NUM_TIME_SLICES = 'num_time_slices'
    PROP_PAULIS_GROUPING = 'paulis_grouping'
    PROP_EXPANSION_MODE = 'expansion_mode'
    PROP_EXPANSION_ORDER = 'expansion_order'
    PROP_NUM_ANCILLAE = 'num_ancillae'
    PROP_EVO_TIME = 'evo_time'

    QPE_CONFIGURATION = {
        'name': 'QPE_HHL',
        'description': 'Quantum Phase Estimation for HHL',
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
                },
                PROP_EVO_TIME: {
                    'type': 'float',
                    'minimum': 1.0
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

    def __init__(self, configuration=None):
        self._configuration = configuration or self.QPE_CONFIGURATION.copy()
        self._operator = None
        self._state_in = None
        self._num_time_slices = 0
        self._paulis_grouping = None
        self._expansion_mode = None
        self._expansion_order = None
        self._num_ancillae = 0
        self._ancilla_phase_coef = 1
        self._circuit = None
        self._ret = {}

    def init_params(self, params, qubit_op):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            qubit_op: Operator instance
        """
        if qubit_op is None:
            raise AlgorithmError("Operator instance is required.")

        operator = qubit_op

        qpe_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        for k, p in self._configuration.get("input_schema").get("properties").items():
            if qpe_params.get(k) == None:
                qpe_params[k] = p.get("default")

        num_time_slices = qpe_params.get(QPE.PROP_NUM_TIME_SLICES)
        paulis_grouping = qpe_params.get(QPE.PROP_PAULIS_GROUPING)
        expansion_mode = qpe_params.get(QPE.PROP_EXPANSION_MODE)
        expansion_order = qpe_params.get(QPE.PROP_EXPANSION_ORDER)
        num_ancillae = qpe_params.get(QPE.PROP_NUM_ANCILLAE)
        evo_time = qpe_params.get(QPE.PROP_EVO_TIME)

        # Set up initial state, we need to add computed num qubits to params
        init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        init_state_params['num_qubits'] = operator.num_qubits
        init_state = get_initial_state_instance(init_state_params['name'])
        init_state.init_params(init_state_params)

        # Set up iqft, we need to add num qubits to params which is our num_ancillae bits here
        iqft_params = params.get(QuantumAlgorithm.SECTION_KEY_IQFT)
        iqft_params['num_qubits'] = num_ancillae
        iqft = get_iqft_instance(iqft_params['name'])
        iqft.init_params(iqft_params)

        self.init_args(
            operator, init_state, iqft, num_time_slices, num_ancillae,
            paulis_grouping=paulis_grouping, expansion_mode=expansion_mode,
            expansion_order=expansion_order, evo_time=evo_time)

    def init_args(
            self, operator, state_in, iqft, num_time_slices, num_ancillae,
            paulis_grouping='random', expansion_mode='trotter', expansion_order=1,
            evo_time=None):
        # if self._backend.find('statevector') >= 0:
        #     raise ValueError('Selected backend does not support measurements.')
        self._operator = operator
        self._state_in = state_in
        self._iqft = iqft
        self._num_time_slices = num_time_slices
        self._num_ancillae = num_ancillae
        self._paulis_grouping = paulis_grouping
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._evo_time = evo_time
        self._ret = {}

    def _construct_phase_estimation_circuit(self, measure=False):
        """Implement the Quantum Phase Estimation algorithm"""

        a = QuantumRegister(self._num_ancillae, name='a')
        q = QuantumRegister(self._operator.num_qubits, name='q')
        qc = QuantumCircuit(a, q)
        if measure:
            c = ClassicalRegister(self._num_ancillae, name='c')
            qc.add(c)

        # initialize state_in
        qc += self._state_in.construct_circuit('circuit', q)

        # Put all ancillae in uniform superposition
        qc.u2(0, np.pi, a)

        # phase kickbacks via dynamics
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
            qc += self._operator.construct_evolution_circuit(
                slice_pauli_list, self._evo_time, self._num_time_slices, q, a, ctl_idx=i
            )
            # global phase shift for the ancilla due to the identity pauli term
            qc.u1(self._evo_time * self._ancilla_phase_coef * (2 ** i), a[i])

        # inverse qft on ancillae
        self._iqft.construct_circuit('circuit', a, qc)

        if measure:
            qc.measure(a, c)

        self._circuit = qc
        return qc


    def _setup_qpe(self, measure=False):
        self._operator._check_representation('paulis')
        if self._evo_time == None:
            self._evo_time = (1-2**-self._num_ancillae)*2*np.pi/sum([abs(p[0]) for p in self._operator.paulis])


        # check for identify paulis to get its coef for applying global phase shift on ancillae later
        num_identities = 0
        for p in self._operator.paulis:
            if np.all(p[1].v == 0) and np.all(p[1].w == 0):
                num_identities += 1
                if num_identities > 1:
                    raise RuntimeError('Multiple identity pauli terms are present.')
                self._ancilla_phase_coef = p[0].real if isinstance(p[0], complex) else p[0]

        self._construct_phase_estimation_circuit(measure=measure)
        logger.info('QPE circuit qasm length is roughly {}.'.format(
            len(self._circuit.qasm().split('\n'))
        ))
        return self._circuit

    def _compute_eigenvalue(self):
        if self._circuit is None:
            self._setup_qpe(measure=True)
        result = execute(self._circuit, backend="local_qasm_simulator").result()
        print(result)
        counts = result.get_counts(self._circuit)

        rd = result.get_counts(self._circuit)
        rets = sorted([[rd[k], k, k] for k in rd])[::-1]
        for d in rets:
            d[2] = sum([2**-(i+1) for i, e in enumerate(reversed(d[2])) if e == "1"])*2*np.pi/self._evo_time

        self._ret['measurements'] = rets
        self._ret['evo_time'] = self._evo_time


    def run(self):
        self._compute_eigenvalue()
        return self._ret

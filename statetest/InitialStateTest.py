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
import logging

from functools import reduce
import numpy as np
from math import log
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.tools.qi.pauli import Pauli
from qiskit_aqua import Operator, QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_initial_state_instance, get_iqft_instance
from qiskit.extensions.simulator import snapshot
import matplotlib.pyplot as plt

from qiskit.tools.visualization._circuit_visualization import matplotlib_circuit_drawer

logger = logging.getLogger(__name__)

class IST():

    PROP_NUM_ANCILLAE = 'num_ancillae'
    PROP_USE_BASIS_GATES = 'use_basis_gates'
    PROP_BACKEND = 'backend'

    IST_CONFIGURATION = {
        'name': 'State_Test',
        'description': 'Test for InitialState',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'initialstate_schema',
            'type': 'object',
            'properties': {
                PROP_NUM_ANCILLAE: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                PROP_USE_BASIS_GATES: {
                    'type': 'boolean',
                    'default': True,
                },
                PROP_BACKEND: {
                    'type': 'string',
                    'default': 'local_qasm_simulator'
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy'],
        'depends': ['initial_state'],
        'defaults': {
            'initial_state': {
                'name': 'CUSTOM'
            }
        }
    }

    def __init__(self, configuration=None):
        self._configuration = configuration or self.IST_CONFIGURATION.copy()
        self._state_in = None
        self._num_ancillae = 0
        self._ancilla_phase_coef = 0
        self._circuit = None
        self._ret = {}
        self._backend = None
        self._num_qubits = 0
        self._globphase = False

    def init_params(self, params, num_qubits):
        
        ist_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        for k, p in self._configuration.get("input_schema").get("properties").items():
            if ist_params.get(k) == None:
                ist_params[k] = p.get("default")

        num_ancillae = ist_params.get(IST.PROP_NUM_ANCILLAE)
        use_basis_gates = ist_params.get(IST.PROP_USE_BASIS_GATES)
        backend = ist_params.get(IST.PROP_BACKEND)
        self._num_qubits = num_qubits

        # Set up initial state, we need to add computed num qubits to params, check the length of the vector
        init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        init_state_params['num_qubits'] = num_qubits
        vector = init_state_params['state_vector']
        globphase = False
        if vector[0] < 0:
            globphase = True
        init_state = get_initial_state_instance(init_state_params['name'])
        init_state.init_params(init_state_params)
        
        self.init_args(
            init_state, num_ancillae, globphase, use_basis_gates=use_basis_gates, backend = backend, 
            )

    def init_args(
            self, state_in, num_ancillae, globphase,
            use_basis_gates=True, backend='local_qasm_simulator'):
        #if self._backend.find('statevector') >= 0:
        #     raise ValueError('Selected backend does not support measurements.')
        self._state_in = state_in
        self._num_ancillae = num_ancillae
        self._use_basis_gates = use_basis_gates
        self._backend = backend
        self._globphase = globphase

    def _construct_state_circuit(self, measure=False):

        q = QuantumRegister(self._num_qubits, name='q')
        qc = QuantumCircuit(q)

        # initialize state_in
        
        qc.snapshot("2")

        qc += self._state_in.construct_circuit('circuit', q)
        if self._globphase:
            qc.u1(np.pi,q[0])
            qc.x(q[0])
            qc.u1(np.pi,q[0])
            qc.x(q[0])
        qc.barrier(q)
        #matplotlib_circuit_drawer(qc)

        qc.snapshot("1")
        self._circuit = qc
        return qc


    def _setup_ist(self, measure=False):

        self._construct_state_circuit(measure=measure)
        logger.info('InitialStateTest circuit qasm length is roughly {}.'.format(
            len(self._circuit.qasm().split('\n'))
        ))
        print('InitialStateTest circuit qasm length is roughly {}.'.format(
            len(self._circuit.qasm().split('\n'))
        ))
        return self._circuit

    def run(self):
        qc = self._setup_ist()
        #matplotlib_circuit_drawer(qc)
        #plt.show()
        result = execute(qc,backend=self._backend,config={"data":["quantum_state_ket"]},shots=1)
        #result = execute(qc, backend=self._backend).result()
        res = result.result().get_snapshot("1")["quantum_state_ket"]
        res2 = result.result().get_snapshot("2")["quantum_state_ket"]

        return res, res2, result

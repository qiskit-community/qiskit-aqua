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
The HHL algorithm.
"""

import logging

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from qiskit_aqua import QuantumAlgorithm
from qiskit_aqua import get_eigs_instance, get_reciprocal_instance, get_initial_state_instance
import numpy as np

import qiskit.extensions.simulator

logger = logging.getLogger(__name__)


class HHL(QuantumAlgorithm):
    """The HHL algorithm."""

    HHL_CONFIGURATION = {
        'name': 'HHL',
        'description': 'The HHL Algorithm for Solving Linear Systems of equations',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'hhl_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'problems': ['energy'],
        'depends': ['eigs', 'initial_state', 'reciprocal'],
        'defaults': {
            'eigs': {
                'name': 'QPE',
                'num_ancillae': 6,
                'num_time_slices': 50,
                'expansion_mode': 'suzuki',
                'expansion_order': 2,
                'qft': {'name': 'STANDARD'}
            },
            'initial_state': {
                'name': 'ZERO'
            },
            'reciprocal': {
                'name': 'LOOKUP'
            }
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.HHL_CONFIGURATION.copy())
        self._matrix = None
        self._invec = None

        self._eigs = None
        self._init_state = None
        self._reciprocal = None

        self._circuit = None
        self._io_register = None
        self._eigenvalue_register = None
        self._ancilla_register = None
        self._success_bit = None

        self._num_q = 0
        self._num_a = 0

        self._ret = {}


    def init_params(self, params, matrix):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: list or np.array instance
        """
        if matrix is None:
            raise AlgorithmError("Matrix instance is required.")
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
    
        eigs_params = params.get(QuantumAlgorithm.SECTION_KEY_EIGS) or {}
        eigs = get_eigs_instance(eigs_params["name"])
        eigs.init_params(eigs_params, matrix)

        num_q, num_a = eigs.get_register_sizes()

        init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE) or {}
        

        # Fix invec for nonhermitian/non 2**n size matrices
        if init_state_params.get("name") == "CUSTOM":
            invec = init_state_params['state_vector']
            assert(matrix.shape[0] == len(init_state_params['state_vector']), 
                    "Check input vector size!")
            tmpvec = invec + (2**num_q - len(invec))*[0]
            init_state_params['state_vector'] = tmpvec
        else:
            invec = [1, 0]
        init_state_params["num_qubits"] = num_q
        init_state = get_initial_state_instance(init_state_params["name"])
        init_state.init_params(init_state_params)
        invec = np.array(list(map(lambda x: x[0]+1j*x[1] if isinstance(x, list)
            else x, invec)))
        
        reciprocal_params = params.get(QuantumAlgorithm.SECTION_KEY_RECIPROCAL) or {}
        reciprocal_params["negative_evals"] = eigs._negative_evals
        reci = get_reciprocal_instance(reciprocal_params["name"])
        reci.init_params(reciprocal_params)

        self.init_args(matrix, invec, eigs, init_state, reci, num_q, num_a)


    def init_args(self, matrix, invec, eigs, init_state, reciprocal, num_q, num_a):
        self._matrix = matrix
        self._invec = invec
        self._eigs = eigs
        self._init_state = init_state
        self._reciprocal = reciprocal
        self._num_q = num_q
        self._num_a = num_a

       
    def _construct_circuit(self):
        snap = True

        q = QuantumRegister(self._num_q)
        c = ClassicalRegister(1)
        qc = QuantumCircuit(q, c)

        # InitialState
        qc += self._init_state.construct_circuit("circuit", q)

        if snap: qc.snapshot("0")

        # EigenvalueEstimation (QPE)
        qc += self._eigs.construct_circuit("circuit", q)
        a = self._eigs._output_register

        if snap: qc.snapshot("1")

        # Reciprocal calculation with rotation
        qc += self._reciprocal.construct_circuit("circuit", a)
        s = self._reciprocal._anc

        if snap: qc.snapshot("2")

        # Inverse EigenvalueEstimation
        qc += self._eigs.construct_inverse("circuit")
        
        if snap: qc.snapshot("3")

        # Measurement of the ancilla qubit
        qc.measure(s, c)

        self._io_register = q
        self._eigenvalue_register = a
        self._ancilla_register = s
        self._success_bit = c
        self._circuit = qc


    def run(self):
        self._construct_circuit()
        return self._circuit

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
The Quantum Eigenvalue Estimation algorithm.
"""

import logging

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from qiskit_aqua import QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_initial_state_instance, get_eigs_instance

import numpy as np

logger = logging.getLogger(__name__)


class EigenvalueEstimation(QuantumAlgorithm):
    """
    The Eigenvalue Estimation algorithm.
    """
    
    EIGENVALUE_ESTIMATION_CONFIGURATION = {
        'name': 'EigenvalueEstimation',
        'description': 'Eigenvalue Estimation algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'ee_scheme',
            'type': 'object',
            'properties': {
                },
            'additionalProperties': False
        },
        'problems': ['energy'],
        'depends': ['eigs', 'initial_state'],
        'defaults': {
            'eigs': {
                'name': 'QPE',
                'num_ancillae': 6,
                'num_time_slices': 20,
                'expansion_mode': 'trotter',
                'expansion_order': 2
            },
            'initial_state': {
                'name': 'ZERO'
            }
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.EIGENVALUE_ESTIMATION_CONFIGURATION.copy())
        self._qpe = None
        self._state_in = None
        self._num_q = 0
        self._num_a = 0
        self._circuit = None
        self._shots = 0
        self._matrix = None
        self._ret = {}

    def init_params(self, params, matrix):

        if matrix is None:
            raise ValueError("Matrix needed")
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        # Extending the operator matrix, if the dimension is not in 2**n
        if np.log2(matrix.shape[0]) % 1 != 0:
            matrix_dim = True
            next_higher = int(np.ceil(np.log2(matrix.shape[0])))
            new_matrix = np.identity(2**next_higher)
            new_matrix = np.array(new_matrix, dtype = complex)
            new_matrix[:matrix.shape[0], :matrix.shape[0]] = matrix[:,:]
            matrix = new_matrix
        num_q = int(np.log2(matrix.shape[0]))

        qpe_params = params.get(QuantumAlgorithm.SECTION_KEY_EIGS) or {}
        qpe = get_eigs_instance(qpe_params["name"])
        qpe.init_params(qpe_params, matrix)

        hermitian_matrix = qpe._hermitian_matrix
        
        init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE) or {}
        
        # Fix invector for nonhermitian/non 2**n size matrices
        if init_state_params.get("name") == "CUSTOM":
            vector = init_state_params['state_vector']
            if len(vector) < matrix.shape[0] and hermitian_matrix:
                vector = np.append(vector, (matrix.shape[0] - len(vector)) * [0])
            if not hermitian_matrix:
                help_vector = np.zeros(matrix.shape[0] - len(vector))
                vector = np.append(help_vector, vector)
            init_state_params['state_vector'] = vector
        init_state_params["num_qubits"] = num_q
        state_in = get_initial_state_instance(init_state_params["name"])
        state_in.init_params(init_state_params)

        shots = params.get("backend").get("shots")
        
        self.init_args(qpe, state_in, num_q, shots, matrix)

    def init_args(self, qpe, state_in, num_q, shots, matrix):
        self._qpe = qpe
        self._state_in = state_in
        self._num_q = num_q
        self._shots = shots
        self._matrix = matrix

    def _construct_circuit(self):
        q = QuantumRegister(self._num_q)
        qc = QuantumCircuit(q)

        qc += self._state_in.construct_circuit('circuit', q)
        
        qc += self._qpe.construct_circuit('circuit', q)

        a = self._qpe._output_register
        self._num_a = len(a)
        c = ClassicalRegister(self._num_a)
        qc.add(c)
        qc.measure(a, c)
        self._circuit = qc

    def visualization(self, res):
        from numpy.linalg import eig
        import matplotlib.pyplot as plt
        w, v = eig(self._matrix)
        
        print(dir(self._state_in))
        print(self._state_in._state)
        vt = v.T.conj().dot(vec)
        print(vt)
        print(v.dot(vt))

        x = []
        y = []
        for c, _, l in res["measurements"]:
            x.append(l)
            y.append(c)

        return w, v

    def _compute_eigenvalue(self):
        if self._circuit is None:
            self._construct_circuit()
        result = self.execute(self._circuit)

        rd = result.get_counts(self._circuit)
        rets = sorted([[rd[k], k, k] for k in rd])[::-1]
       
        for d in rets:
            d[0] /= self._shots
            offset = 1
            sgn = 1
            if self._qpe._negative_evals:
                sgn = -1 if d[2][-1] == "1" else 1
                d[2] = d[2][:-1]
                offset = 2
            d[2] = sgn*sum([2**-(i+offset) for i, e in enumerate(reversed(d[2])) if e ==
                "1"])*2*np.pi/self._qpe._evo_time

        self._ret['measurements'] = rets
        self._ret['evo_time'] = self._qpe._evo_time
        self._ret['visualization'] = lambda: self.visualization(rets)

    def run(self):
        self._compute_eigenvalue()
        return self._ret

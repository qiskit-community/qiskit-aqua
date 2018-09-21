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

from scipy import linalg
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit_aqua import Operator
from copy import deepcopy

from qiskit_aqua.algorithms.components.eigs import Eigenvalues

try:
    from qiskit_aqua import get_iqft_instance, get_qft_instance
except ImportError as e:
    # Initialization run for get_pluggable_instance results in error
    pass

class QPE(Eigenvalues):
    """A QPE for getting the eigenvalues."""

    PROP_NUM_TIME_SLICES = 'num_time_slices'
    PROP_PAULIS_GROUPING = 'paulis_grouping'
    PROP_EXPANSION_MODE = 'expansion_mode'
    PROP_EXPANSION_ORDER = 'expansion_order'
    PROP_NUM_ANCILLAE = 'num_ancillae'
    PROP_EVO_TIME = 'evo_time'
    PROP_USE_BASIS_GATES = 'use_basis_gates'
    PROP_HERMITIAN_MATRIX ='hermitian_matrix'
    PROP_NEGATIVE_EVALS = 'negative_evals'
    PROP_IQFT = 'iqft'

    QPE_CONFIGURATION = {
        'name': 'QPE',
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
                    'default': 'trotter',
                    'oneOf': [
                        {'enum': [
                            'suzuki',
                            'trotter'
                        ]}
                    ]
                },
                PROP_EXPANSION_ORDER: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                PROP_NUM_ANCILLAE: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                PROP_EVO_TIME: {
                    'type': ['number', 'null'],
                    'default': None
                },
                PROP_USE_BASIS_GATES: {
                    'type': 'boolean',
                    'default': True,
                },
                PROP_HERMITIAN_MATRIX: {
                    'type': 'boolean',
                    'default': True
                },
                PROP_NEGATIVE_EVALS: {
                    'type': 'boolean',
                    'default': False
                },
                PROP_IQFT: {
                    'type': 'object',
                    'default': {'name': 'STANDARD'}
                },
            },
            'additionalProperties': False
        },
    }

    def __init__(self, configuration=None):
        self._configuration = configuration or self.QPE_CONFIGURATION.copy()
        self._operator = None
        self._num_time_slices = 0
        self._paulis_grouping = None
        self._expansion_mode = None
        self._expansion_order = None
        self._num_ancillae = 0
        self._ancilla_phase_coef = 0
        self._input_register = None
        self._circuit = None
        self._circuit_data = None
        self._inverse = None
        self._state_circuit_length = 0
        self._ret = {}
        self._matrix_dim = True
        self._hermitian_matrix = True
        self._ne_qfts = [None, None]

    def init_params(self, params, matrix):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            matrix: two dimensional array which represents the operator
        """
        if matrix is None:
            raise AlgorithmError("Operator instance is required.")

        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        num_time_slices = params.get(QPE.PROP_NUM_TIME_SLICES)
        paulis_grouping = params.get(QPE.PROP_PAULIS_GROUPING)
        expansion_mode = params.get(QPE.PROP_EXPANSION_MODE)
        expansion_order = params.get(QPE.PROP_EXPANSION_ORDER)
        num_ancillae = params.get(QPE.PROP_NUM_ANCILLAE)
        evo_time = params.get(QPE.PROP_EVO_TIME)
        use_basis_gates = params.get(QPE.PROP_USE_BASIS_GATES)
        hermitian_matrix = params.get(QPE.PROP_HERMITIAN_MATRIX)
        negative_evals = params.get(QPE.PROP_NEGATIVE_EVALS)
        iqft_params = params.get(QPE.PROP_IQFT)

        if hermitian_matrix:
            negative_evals = True

        # Extending the operator matrix, if the dimension is not in 2**n
        if np.log2(matrix.shape[0]) % 1 != 0:
            matrix_dim = True
            next_higher = int(np.ceil(np.log2(matrix.shape[0])))
            new_matrix = np.identity(2**next_higher)
            new_matrix = np.array(new_matrix, dtype = complex)
            new_matrix[:matrix.shape[0], :matrix.shape[0]] = matrix[:,:]
            matrix = new_matrix

        # If operator matrix is not hermitian, extending it to B = ((0, A), (A⁺, 0)), which is hermitian
        # In this case QPE will give singular values
        if not hermitian_matrix:
            negative_evals = True
            new_matrix = np.zeros((2*matrix.shape[0], 2*matrix.shape[0]), dtype=complex)
            new_matrix[matrix.shape[0]:,:matrix.shape[0]] = np.matrix.getH(matrix)[:,:]
            new_matrix[:matrix.shape[0],matrix.shape[0]:] = matrix[:,:]
            matrix = new_matrix
        operator = Operator(matrix=matrix)

        # Set up iqft, we need to add num qubits to params which is our num_ancillae bits here
        iqft_params['num_qubits'] = num_ancillae
        iqft = get_iqft_instance(iqft_params['name'])
        iqft.init_params(iqft_params)
        
        # For converting the encoding of the negative eigenvalues, we need two
        # additional QFTs
        if negative_evals:
            ne_qft_params = iqft_params
            ne_qft_params['num_qubits'] -= 1
            ne_qfts = [ get_qft_instance(ne_qft_params['name']),
                    get_iqft_instance(ne_qft_params['name'])]
            ne_qfts[0].init_params(ne_qft_params)
            ne_qfts[1].init_params(ne_qft_params)
        else:
            ne_qfts = [None, None]


        self.init_args(
            operator, iqft, num_time_slices, num_ancillae,
            paulis_grouping=paulis_grouping, expansion_mode=expansion_mode,
            expansion_order=expansion_order, evo_time=evo_time,
            use_basis_gates=use_basis_gates, hermitian_matrix=hermitian_matrix,
            negative_evals=negative_evals, ne_qfts=ne_qfts)

    def init_args(
            self, operator, iqft, num_time_slices, num_ancillae,
            paulis_grouping='random', expansion_mode='trotter', expansion_order=1,
            evo_time=None, use_basis_gates=True, hermitian_matrix=True,
            negative_evals=False, ne_qfts=[None, None]):
        self._operator = operator
        self._iqft = iqft
        self._num_time_slices = num_time_slices
        self._num_ancillae = num_ancillae
        self._paulis_grouping = paulis_grouping
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._evo_time = evo_time
        self._use_basis_gates = use_basis_gates
        self._hermitian_matrix = hermitian_matrix
        self._negative_evals = negative_evals
        self._ne_qfts = ne_qfts
        self._ret = {}

    def get_register_sizes(self):
        return self._operator.num_qubits, self._num_ancillae

    def get_scaling(self):
        return self._evo_time

    def construct_circuit(self, mode, register):
        """Implement the Quantum Phase Estimation algorithm"""

        if mode == 'vector':
            raise ValueError("QPE only posslible as circuit not vector.")

        a = QuantumRegister(self._num_ancillae)
        q = register

        qc = QuantumCircuit(a, q)

        self._setup_constants()

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
                slice_pauli_list, -self._evo_time, self._num_time_slices, q, a,
                ctl_idx=i, use_basis_gates=self._use_basis_gates
            )
            # global phase shift for the ancilla due to the identity pauli term
            if self._ancilla_phase_coef != 0:
                qc.u1(self._evo_time * self._ancilla_phase_coef * (2 ** i), a[i])

        # inverse qft on ancillae
        self._iqft.construct_circuit('circuit', a, qc)

        # handle negative eigenvalues
        if self._negative_evals:
            self._handle_negative_evals(qc, a)

        self._circuit = qc
        self._output_register = a
        self._input_register = q
        self._circuit_data = deepcopy(qc.data)
        return self._circuit

    def _setup_constants(self):
        # estimate evolution time
        self._operator._check_representation('paulis')
        paulis = self._operator.paulis
        if self._evo_time == None:
            lmax = sum([abs(p[0]) for p in self._operator.paulis])
            if not self._negative_evals:
                self._evo_time = (1-2**-self._num_ancillae)*2*np.pi/lmax
            else:
                self._evo_time = (1/2-2**-self._num_ancillae)*2*np.pi/lmax

        # check for identify paulis to get its coef for applying global phase shift on ancillae later
        num_identities = 0
        for p in self._operator.paulis:
            if np.all(p[1].v == 0) and np.all(p[1].w == 0):
                num_identities += 1
                if num_identities > 1:
                    raise RuntimeError('Multiple identity pauli terms are present.')
                self._ancilla_phase_coef = p[0].real if isinstance(p[0], complex) else p[0]

    def _handle_negative_evals(self, qc, q):
        sgn = q[0]
        qs = [q[i] for i in range(1, len(q))]
        for qi in qs:
            qc.cx(sgn, qi)
        self._ne_qfts[0].construct_circuit('circuit', qs, qc)
        for i, qi in enumerate(reversed(qs)):
            qc.cu1(2*np.pi/2**(i+1), sgn, qi)
        self._ne_qfts[1].construct_circuit('circuit', qs, qc)

    def construct_inverse(self, mode):
        if mode == "vector":
            raise NotImplementedError("Mode vector not supported for construct_inverse")
        elif mode == "circuit":
            return QuantumCircuit(self._input_register, self._output_register)

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

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit_aqua.utils.initial_states import InitialState


class HartreeFock(InitialState):
    """A Hartree-Fock initial state."""

    HARTREEFOCK_CONFIGURATION = {
        'name': 'HartreeFock',
        'description': 'Hartree-Fock initial state',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'hf_state_schema',
            'type': 'object',
            'properties': {
                'qubit_mapping': {
                    'type': 'string',
                    'default': 'parity',
                    'oneOf': [
                        {'enum': ['jordan_wigner', 'parity', 'bravyi_kitaev']}
                    ]
                },
                'two_qubit_reduction': {
                    'type': 'boolean',
                    'default': True
                },
                'num_particles': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                'num_orbitals': {
                    'type': 'integer',
                    'default': 4,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.HARTREEFOCK_CONFIGURATION.copy())
        self._num_qubits = 0
        self._qubit_mapping = 'parity'
        self._two_qubit_reduction = True
        self._num_particles = 2
        self._num_orbitals = 1
        self._bitstr = None

    def init_args(self, num_qubits, num_orbitals, qubit_mapping, two_qubit_reduction, num_particles):
        """

        Args:
            num_qubits (int): number of qubits
            num_orbitals (int): number of spin orbitals
            qubit_mapping (str): mapping type for qubit operator
            two_qubit_reduction (bool): flag indicating whether or not two qubit is reduced
            num_particles (int): number of particles
        """
        self._qubit_mapping = qubit_mapping.lower()
        self._two_qubit_reduction = two_qubit_reduction
        if self._qubit_mapping != 'parity':
            self._two_qubit_reduction = False
        self._num_orbitals = num_orbitals
        self._num_particles = num_particles
        self._num_qubits = num_orbitals - 2 if self._two_qubit_reduction else self._num_orbitals
        if self._num_qubits != num_qubits:
            raise ValueError('Computed num qubits {} does not match actual {}'.format(self._num_qubits, num_qubits))

    def _build_bitstr(self):
        self._num_particles = self._num_particles
        if self._num_particles > self._num_orbitals:
            raise ValueError('# of particles must be less than or equal to # of orbitals.')

        bitstr = np.zeros(self._num_orbitals, np.bool)
        bitstr[:int(np.ceil(self._num_particles / 2))] = True
        bitstr[self._num_orbitals // 2:self._num_orbitals // 2 + int(np.floor(self._num_particles / 2))] = True

        if self._qubit_mapping == 'parity':
            new_bitstr = bitstr.copy()
            for new_k in range(1, new_bitstr.size):
                new_bitstr[new_k] = np.logical_xor(new_bitstr[new_k-1], bitstr[new_k])

            bitstr = np.append(new_bitstr[:self._num_orbitals//2-1], new_bitstr[self._num_orbitals//2:-1]) \
                if self._two_qubit_reduction else new_bitstr

        elif self._qubit_mapping == 'bravyi_kitaev':
            binary_superset_size = int(np.ceil(np.log2(self._num_orbitals)))
            beta = 1
            basis = np.asarray([[1, 0], [0, 1]])
            for i in range(binary_superset_size):
                beta = np.kron(basis, beta)
                beta[-1, :] = 1

            beta = beta[:self._num_orbitals, :self._num_orbitals]
            new_bitstr = beta.dot(bitstr.astype(int)) % 2
            bitstr = new_bitstr.astype(np.bool)

        self._bitstr = bitstr

    def construct_circuit(self, mode, register=None):
        """
        Construct the statevector of desired initial state.

        Args:
            mode (string): `vector` or `circuit`. The `vector` mode produces the vector.
                            While the `circuit` constructs the quantum circuit corresponding that
                            vector.
            register (QuantumRegister): register for circuit construction.

        Returns:
            numpy.ndarray or QuantumCircuit: statevector
        """
        if self._bitstr is None:
            self._build_bitstr()
        if mode == 'vector':
            state = 1.0
            one = np.asarray([0.0, 1.0])
            zero = np.asarray([1.0, 0.0])
            for k in self._bitstr:
                state = np.kron(one if k else zero, state)
            return state
        elif mode == 'circuit':
            if register is None:
                register = QuantumRegister(self._num_qubits, name='q')
            quantum_circuit = QuantumCircuit(register)
            for idx, bit in enumerate(self._bitstr):
                if bit:
                    quantum_circuit.u3(np.pi, 0.0, np.pi, register[idx])
            return quantum_circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')

    @property
    def bitstr(self):
        """Getter of the bit string represented the statevector"""
        if self._bitstr is None:
            self._build_bitstr()
        return self._bitstr

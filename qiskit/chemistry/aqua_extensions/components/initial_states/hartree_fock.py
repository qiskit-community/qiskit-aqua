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

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua.components.initial_states import InitialState

logger = logging.getLogger(__name__)


class HartreeFock(InitialState):
    """A Hartree-Fock initial state."""

    CONFIGURATION = {
        'name': 'HartreeFock',
        'description': 'Hartree-Fock initial state',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'hf_state_schema',
            'type': 'object',
            'properties': {
                'num_orbitals': {
                    'type': 'integer',
                    'default': 4,
                    'minimum': 1
                },
                'num_particles': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
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
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, num_orbitals=4, num_particles=2,
                 qubit_mapping='parity', two_qubit_reduction=True, sq_list=None):
        """Constructor.

        Args:
            num_qubits (int): number of qubits
            num_orbitals (int): number of spin orbitals
            qubit_mapping (str): mapping type for qubit operator
            two_qubit_reduction (bool): flag indicating whether or not two qubit is reduced
            num_particles (int): number of particles
            sq_list ([int]): position of the single-qubit operators that anticommute
                        with the cliffords

        Raises:
            ValueError: wrong setting in num_particles and num_orbitals.
            ValueError: wrong setting for computed num_qubits and supplied num_qubits.
        """
        self.validate(locals())
        super().__init__()
        self._sq_list = sq_list
        self._qubit_tapering = False if self._sq_list is None else True
        self._qubit_mapping = qubit_mapping.lower()
        self._two_qubit_reduction = two_qubit_reduction
        if self._qubit_mapping != 'parity':
            if self._two_qubit_reduction:
                logger.warning("two_qubit_reduction only works with parity qubit mapping "
                               "but you have {}. We switch two_qubit_reduction "
                               "to False.".format(self._qubit_mapping))
                self._two_qubit_reduction = False

        self._num_orbitals = num_orbitals
        self._num_particles = num_particles

        if self._num_particles > self._num_orbitals:
            raise ValueError("# of particles must be less than or equal to # of orbitals.")

        self._num_qubits = num_orbitals - 2 if self._two_qubit_reduction else self._num_orbitals
        self._num_qubits = self._num_qubits \
            if not self._qubit_tapering else self._num_qubits - len(sq_list)
        if self._num_qubits != num_qubits:
            raise ValueError("Computed num qubits {} does not match "
                             "actual {}".format(self._num_qubits, num_qubits))

        self._bitstr = None

    def _build_bitstr(self):

        half_orbitals = self._num_orbitals // 2
        bitstr = np.zeros(self._num_orbitals, np.bool)
        bitstr[-int(np.ceil(self._num_particles / 2)):] = True
        bitstr[-(half_orbitals + int(np.floor(self._num_particles / 2))):-half_orbitals] = True

        if self._qubit_mapping == 'parity':
            new_bitstr = bitstr.copy()

            t = np.triu(np.ones((self._num_orbitals, self._num_orbitals)))
            new_bitstr = t.dot(new_bitstr.astype(np.int)) % 2

            bitstr = np.append(new_bitstr[1:half_orbitals], new_bitstr[half_orbitals + 1:]) \
                if self._two_qubit_reduction else new_bitstr

        elif self._qubit_mapping == 'bravyi_kitaev':
            binary_superset_size = int(np.ceil(np.log2(self._num_orbitals)))
            beta = 1
            basis = np.asarray([[1, 0], [0, 1]])
            for i in range(binary_superset_size):
                beta = np.kron(basis, beta)
                beta[0, :] = 1

            beta = beta[:self._num_orbitals, :self._num_orbitals]
            new_bitstr = beta.dot(bitstr.astype(int)) % 2
            bitstr = new_bitstr.astype(np.bool)

        if self._qubit_tapering:
            sq_list = (len(bitstr) - 1) - np.asarray(self._sq_list)
            bitstr = np.delete(bitstr, sq_list)

        self._bitstr = bitstr.astype(np.bool)

    def construct_circuit(self, mode, register=None):
        """
        Construct the statevector of desired initial state.

        Args:
            mode (string): `vector` or `circuit`. The `vector` mode produces the vector.
                            While the `circuit` constructs the quantum circuit corresponding that
                            vector.
            register (QuantumRegister): register for circuit construction.

        Returns:
            QuantumCircuit or numpy.ndarray: statevector.

        Raises:
            ValueError: when mode is not 'vector' or 'circuit'.
        """
        if self._bitstr is None:
            self._build_bitstr()
        if mode == 'vector':
            state = 1.0
            one = np.asarray([0.0, 1.0])
            zero = np.asarray([1.0, 0.0])
            for k in self._bitstr[::-1]:
                state = np.kron(one if k else zero, state)
            return state
        elif mode == 'circuit':
            if register is None:
                register = QuantumRegister(self._num_qubits, name='q')
            quantum_circuit = QuantumCircuit(register)
            for qubit_idx, bit in enumerate(self._bitstr[::-1]):
                if bit:
                    quantum_circuit.u3(np.pi, 0.0, np.pi, register[qubit_idx])
            return quantum_circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')

    @property
    def bitstr(self):
        """Getter of the bit string represented the statevector."""
        if self._bitstr is None:
            self._build_bitstr()
        return self._bitstr

# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Hartree-Fock initial state."""

from typing import Optional, Union, List
import logging
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from qiskit.aqua.components.initial_states import InitialState

logger = logging.getLogger(__name__)


class HartreeFock(InitialState):
    """A Hartree-Fock initial state."""

    def __init__(self, num_qubits: int,
                 num_orbitals: int,
                 num_particles: Union[List[int], int],
                 qubit_mapping: str = 'parity',
                 two_qubit_reduction: bool = True,
                 sq_list: Optional[List[int]] = None) -> None:
        """Constructor.

        Args:
            num_qubits: number of qubits, has a min. value of 1.
            num_orbitals: number of spin orbitals, has a min. value of 1.
            num_particles: number of particles, if it is a list, the first number
                            is alpha and the second number if beta.
            qubit_mapping: mapping type for qubit operator
            two_qubit_reduction: flag indicating whether or not two qubit is reduced
            sq_list: position of the single-qubit operators that
                    anticommute with the cliffords

        Raises:
            ValueError: wrong setting in num_particles and num_orbitals.
            ValueError: wrong setting for computed num_qubits and supplied num_qubits.
        """
        validate_min('num_qubits', num_qubits, 1)
        validate_min('num_orbitals', num_orbitals, 1)
        if isinstance(num_particles, list) and len(num_particles) != 2:
            raise ValueError('Num particles value {}. Number of values allowed is 2'.format(
                num_particles))
        validate_in_set('qubit_mapping', qubit_mapping,
                        {'jordan_wigner', 'parity', 'bravyi_kitaev'})
        super().__init__()
        self._sq_list = sq_list
        self._qubit_tapering = bool(self._sq_list)
        self._qubit_mapping = qubit_mapping.lower()
        self._two_qubit_reduction = two_qubit_reduction
        if self._qubit_mapping != 'parity':
            if self._two_qubit_reduction:
                logger.warning("two_qubit_reduction only works with parity qubit mapping "
                               "but you have %s. We switch two_qubit_reduction "
                               "to False.", self._qubit_mapping)
                self._two_qubit_reduction = False

        self._num_orbitals = num_orbitals
        if isinstance(num_particles, list):
            self._num_alpha = num_particles[0]
            self._num_beta = num_particles[1]
        else:
            logger.info("We assume that the number of alphas and betas are the same.")
            self._num_alpha = num_particles // 2
            self._num_beta = num_particles // 2

        self._num_particles = self._num_alpha + self._num_beta

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
        bitstr[-self._num_alpha:] = True
        bitstr[-(half_orbitals + self._num_beta):-half_orbitals] = True

        if self._qubit_mapping == 'parity':
            new_bitstr = bitstr.copy()

            t_r = np.triu(np.ones((self._num_orbitals, self._num_orbitals)))
            new_bitstr = t_r.dot(new_bitstr.astype(np.int)) % 2  # pylint: disable=no-member

            bitstr = np.append(new_bitstr[1:half_orbitals], new_bitstr[half_orbitals + 1:]) \
                if self._two_qubit_reduction else new_bitstr

        elif self._qubit_mapping == 'bravyi_kitaev':
            binary_superset_size = int(np.ceil(np.log2(self._num_orbitals)))
            beta = 1
            basis = np.asarray([[1, 0], [0, 1]])
            for _ in range(binary_superset_size):
                beta = np.kron(basis, beta)
                beta[0, :] = 1
            start_idx = beta.shape[0] - self._num_orbitals
            beta = beta[start_idx:, start_idx:]
            new_bitstr = beta.dot(bitstr.astype(int)) % 2
            bitstr = new_bitstr.astype(np.bool)

        if self._qubit_tapering:
            sq_list = (len(bitstr) - 1) - np.asarray(self._sq_list)
            bitstr = np.delete(bitstr, sq_list)

        self._bitstr = bitstr.astype(np.bool)

    def construct_circuit(self, mode='circuit', register=None):
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

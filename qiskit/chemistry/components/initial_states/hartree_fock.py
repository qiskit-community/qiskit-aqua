# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Hartree-Fock initial state."""

import warnings
from typing import Optional, Union, List
import logging
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from qiskit.aqua.components.initial_states import InitialState
from qiskit.chemistry.circuit.library.initial_states.hartree_fock import hartree_fock_bitstring

logger = logging.getLogger(__name__)


class HartreeFock(InitialState):
    """A Hartree-Fock initial state."""

    def __init__(self,
                 num_orbitals: int,
                 num_particles: Union[List[int], int],
                 qubit_mapping: str = 'parity',
                 two_qubit_reduction: bool = True,
                 sq_list: Optional[List[int]] = None) -> None:
        """
        Args:
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
        # validate the input
        validate_min('num_orbitals', num_orbitals, 1)
        validate_in_set('qubit_mapping', qubit_mapping,
                        {'jordan_wigner', 'parity', 'bravyi_kitaev'})

        if qubit_mapping != 'parity' and two_qubit_reduction:
            warnings.warn('two_qubit_reduction only works with parity qubit mapping '
                          'but you have %s. We switch two_qubit_reduction '
                          'to False.' % qubit_mapping)
            two_qubit_reduction = False

        super().__init__()

        # get the bitstring encoding the Hartree Fock state
        if isinstance(num_particles, list):
            num_particles = tuple(num_particles)  # type: ignore

        bitstr = hartree_fock_bitstring(num_orbitals, num_particles, qubit_mapping,
                                        two_qubit_reduction, sq_list)
        self._bitstr = bitstr

    def construct_circuit(self, mode='circuit', register=None):
        """Construct the statevector of desired initial state.

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
        if mode == 'vector':
            state = 1.0
            one = np.asarray([0.0, 1.0])
            zero = np.asarray([1.0, 0.0])
            for k in self._bitstr[::-1]:
                state = np.kron(one if k else zero, state)
            return state
        elif mode == 'circuit':
            if register is None:
                register = QuantumRegister(len(self.bitstr), name='q')
            quantum_circuit = QuantumCircuit(register, name='HF')
            for i, bit in enumerate(reversed(self.bitstr)):
                if bit:
                    quantum_circuit.x(i)
            return quantum_circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')

    @property
    def bitstr(self):
        """Getter of the bit string represented the statevector."""
        return self._bitstr

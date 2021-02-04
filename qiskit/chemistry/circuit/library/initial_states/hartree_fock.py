# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
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
from typing import Optional, Union, List, Tuple
import logging
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.utils.validation import validate_min, validate_in_set

logger = logging.getLogger(__name__)


class HartreeFock(QuantumCircuit):
    """A Hartree-Fock initial state."""

    def __init__(self,
                 num_orbitals: int,
                 num_particles: Union[Tuple[int, int], int],
                 qubit_mapping: str = 'parity',
                 two_qubit_reduction: bool = True,
                 sq_list: Optional[List[int]] = None) -> None:
        """
        Args:
            num_orbitals: The number of spin orbitals, has a min. value of 1.
            num_particles: The number of particles. If this is an integer, it is the total (even)
                number of particles. If a tuple, the first number is alpha and the second number is
                beta.
            qubit_mapping: The mapping type for qubit operator.
            two_qubit_reduction: A flag indicating whether or not two qubit is reduced.
            sq_list: The position of the single-qubit operators that anticommute with the cliffords.
        """
        # get the bitstring encoding the Hartree Fock state
        bitstr = hartree_fock_bitstring(num_orbitals, num_particles, qubit_mapping,
                                        two_qubit_reduction, sq_list)

        # construct the circuit
        qr = QuantumRegister(len(bitstr), 'q')
        super().__init__(qr, name='HF')

        # add gates in the right positions
        for i, bit in enumerate(reversed(bitstr)):
            if bit:
                self.x(i)


def hartree_fock_bitstring(num_orbitals: int,
                           num_particles: Union[Tuple[int, int], int],
                           qubit_mapping: str = 'parity',
                           two_qubit_reduction: bool = True,
                           sq_list: Optional[List[int]] = None) -> List[bool]:
    """Compute the bitstring representing the Hartree-Fock state for the specified system.

    Args:
        num_orbitals: The number of spin orbitals, has a min. value of 1.
        num_particles: The number of particles. If this is an integer, it is the total (even) number
            of particles. If a tuple, the first number is alpha and the second number is beta.
        qubit_mapping: The mapping type for qubit operator.
        two_qubit_reduction: A flag indicating whether or not two qubit is reduced.
        sq_list: The position of the single-qubit operators that anticommute with the cliffords.

    Returns:
        The bitstring representing the state of the Hartree-Fock state as array of bools.

    Raises:
        ValueError: If the total number of particles is larger than the number of orbitals.
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

    if isinstance(num_particles, tuple):
        num_alpha, num_beta = num_particles
    else:
        logger.info('We assume that the number of alphas and betas are the same.')
        num_alpha = num_beta = num_particles // 2

    num_particles = num_alpha + num_beta

    if num_particles > num_orbitals:
        raise ValueError('# of particles must be less than or equal to # of orbitals.')

    half_orbitals = num_orbitals // 2
    bitstr = np.zeros(num_orbitals, bool)
    bitstr[-num_alpha:] = True
    bitstr[-(half_orbitals + num_beta):-half_orbitals] = True

    if qubit_mapping == 'parity':
        new_bitstr = bitstr.copy()

        t_r = np.triu(np.ones((num_orbitals, num_orbitals)))
        new_bitstr = t_r.dot(new_bitstr.astype(int)) % 2  # pylint: disable=no-member

        bitstr = np.append(new_bitstr[1:half_orbitals], new_bitstr[half_orbitals + 1:]) \
            if two_qubit_reduction else new_bitstr

    elif qubit_mapping == 'bravyi_kitaev':
        binary_superset_size = int(np.ceil(np.log2(num_orbitals)))
        beta = np.array([1])
        basis = np.asarray([[1, 0], [0, 1]])
        for _ in range(binary_superset_size):
            beta = np.kron(basis, beta)
            beta[0, :] = 1
        start_idx = beta.shape[0] - num_orbitals
        beta = beta[start_idx:, start_idx:]
        new_bitstr = beta.dot(bitstr.astype(int)) % 2  # type: ignore
        bitstr = new_bitstr.astype(bool)

    if sq_list is not None:
        sq_list = [len(bitstr) - 1 - position for position in sq_list]
        bitstr = np.delete(bitstr, sq_list)

    return bitstr.astype(bool).tolist()

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

"""
Generate Number Partitioning (Partition) instances, and convert them
into a Hamiltonian given as a Pauli list.
"""

import logging

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator

logger = logging.getLogger(__name__)


def get_operator(values):
    """Construct the Hamiltonian for a given Partition instance.

    Given a list of numbers for the Number Partitioning problem, we
    construct the Hamiltonian described as a list of Pauli gates.

    Args:
        values (numpy.ndarray): array of values.

    Returns:
        tuple(WeightedPauliOperator, float): operator for the Hamiltonian and a
        constant shift for the obj function.

    """
    n = len(values)
    # The Hamiltonian is:
    # \sum_{i,j=1,\dots,n} ij z_iz_j + \sum_{i=1,\dots,n} i^2
    pauli_list = []
    for i in range(n):
        for j in range(i):
            x_p = np.zeros(n, dtype=bool)
            z_p = np.zeros(n, dtype=bool)
            z_p[i] = True
            z_p[j] = True
            pauli_list.append([2. * values[i] * values[j], Pauli((z_p, x_p))])
    return WeightedPauliOperator(paulis=pauli_list), sum(values * values)


def partition_value(x, number_list):
    """Compute the value of a partition.

    Args:
        x (numpy.ndarray): binary string as numpy array.
        number_list (numpy.ndarray): list of numbers in the instance.

    Returns:
        float: difference squared between the two sides of the number
            partition.
    """
    diff = np.sum(number_list[x == 0]) - np.sum(number_list[x == 1])
    return diff * diff

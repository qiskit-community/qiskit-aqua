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

"""
Generate Number Partitioning (Partition) instances, and convert them
into a Hamiltonian given as a Pauli list.
"""

import logging
import warnings

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
            x_p = np.zeros(n, dtype=np.bool)
            z_p = np.zeros(n, dtype=np.bool)
            z_p[i] = True
            z_p[j] = True
            pauli_list.append([2. * values[i] * values[j], Pauli(z_p, x_p)])
    return WeightedPauliOperator(paulis=pauli_list), sum(values*values)


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


def random_number_list(n, weight_range=100, savefile=None):
    """ random number list """
    # pylint: disable=import-outside-toplevel
    from .common import random_number_list as redirect_func
    warnings.warn("random_number_list function has been moved to "
                  "qiskit.optimization.ising.common,, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(n=n, weight_range=weight_range, savefile=savefile)


def read_numbers_from_file(filename):
    """ read numbers from file """
    # pylint: disable=import-outside-toplevel
    from .common import read_numbers_from_file as redirect_func
    warnings.warn("read_numbers_from_file function has been moved to "
                  "qiskit.optimization.ising.common, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(filename)


def sample_most_likely(state_vector):
    """ sample most likely """
    # pylint: disable=import-outside-toplevel
    from .common import sample_most_likely as redirect_func
    warnings.warn("sample_most_likely function has been moved "
                  "to qiskit.optimization.ising.common,, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(state_vector=state_vector)


def get_partition_qubitops(values):
    """ get partition qubit ops """
    warnings.warn("get_partition_qubitops function has been changed to get_operator"
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return get_operator(values)

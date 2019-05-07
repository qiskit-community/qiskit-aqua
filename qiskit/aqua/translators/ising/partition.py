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

# Generate Number Partitioning (Partition) instances, and convert them
# into a Hamiltonian given as a Pauli list.


import logging
from collections import OrderedDict

import numpy as np

from qiskit.quantum_info import Pauli
from qiskit.aqua import Operator

logger = logging.getLogger(__name__)


def random_number_list(n, weight_range=100, savefile=None):
    """Generate a set of positive integers within the given range.

    Args:
        n (int): size of the set of numbers.
        weight_range (int): maximum absolute value of the numbers.
        savefile (str or None): write numbers to this file.

    Returns:
        numpy.ndarray: the list of integer numbers.
    """
    number_list = np.random.randint(low=1, high=(weight_range + 1), size=n)
    if savefile:
        with open(savefile, 'w') as outfile:
            for i in range(n):
                outfile.write('{}\n'.format(number_list[i]))
    return number_list


def get_partition_qubitops(values):
    """Construct the Hamiltonian for a given Partition instance.

    Given a list of numbers for the Number Partitioning problem, we
    construct the Hamiltonian described as a list of Pauli gates.

    Args:
        values (numpy.ndarray): array of values.

    Returns:
        operator.Operator, float: operator for the Hamiltonian and a
        constant shift for the obj function.

    """
    n = len(values)
    # The Hamiltonian is:
    # \sum_{i,j=1,\dots,n} ij z_iz_j + \sum_{i=1,\dots,n} i^2
    pauli_list = []
    for i in range(n):
        for j in range(i):
            xp = np.zeros(n, dtype=np.bool)
            zp = np.zeros(n, dtype=np.bool)
            zp[i] = True
            zp[j] = True
            pauli_list.append([2. * values[i] * values[j], Pauli(zp, xp)])
    return Operator(paulis=pauli_list), sum(values*values)


def read_numbers_from_file(filename):
    """Read numbers from a file

    Args:
        filename (str): name of the file.

    Returns:
        numpy.ndarray: list of numbers as a numpy.ndarray.
    """
    numbers = []
    with open(filename) as infile:
        for line in infile:
            assert(int(round(float(line))) == float(line))
            numbers.append(int(round(float(line))))
    return np.array(numbers)


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


def sample_most_likely(state_vector):
    """Compute the most likely binary string from state vector.

    Args:
        state_vector (numpy.ndarray or dict): state vector or counts.

    Returns:
        numpy.ndarray: binary string as numpy.ndarray of ints.
    """
    if isinstance(state_vector, dict) or isinstance(state_vector, OrderedDict):
        # get the binary string with the largest count
        binary_string = sorted(state_vector.items(), key=lambda kv: kv[1])[-1][0]
        x = np.asarray([int(y) for y in reversed(list(binary_string))])
        return x
    else:
        n = int(np.log2(state_vector.shape[0]))
        k = np.argmax(np.abs(state_vector))
        x = np.zeros(n)
        for i in range(n):
            x[i] = k % 2
            k >>= 1
        return x

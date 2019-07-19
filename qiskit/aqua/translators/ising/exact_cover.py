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
    number_list = np.random.randint(low=1, high=(weight_range+1), size=n)
    if savefile:
        with open(savefile, 'w') as outfile:
            for i in range(n):
                outfile.write('{}\n'.format(number_list[i]))
    return number_list


def get_exact_cover_qubitops(list_of_subsets):
    """Construct the Hamiltonian for the exact solver problem


    Args:
        list_of_subsets: list of lists (i.e., subsets)

    Returns:
        operator.Operator, float: operator for the Hamiltonian and a
        constant shift for the obj function.

    Assumption:
        the union of the subsets contains all the elements to cover

    The Hamiltonian is:
       sum_{each element e}{(1-sum_{every subset_i that contains e}{Xi})^2},
       where Xi (Xi=1 or 0) means whether should include the subset i.

    """
    n = len(list_of_subsets)

    U = []
    for sub in list_of_subsets:
        U.extend(sub)
    U = np.unique(U)   # U is the universe

    shift = 0
    pauli_list = []

    for e in U:
        cond = [True if e in sub else False for sub in list_of_subsets]
        indices_has_e = np.arange(n)[cond]
        num_has_e = len(indices_has_e)
        Y = 1-0.5*num_has_e
        shift += Y*Y

        for i in indices_has_e:
            for j in indices_has_e:
                if i != j:
                    wp = np.zeros(n)
                    vp = np.zeros(n)
                    vp[i] = 1
                    vp[j] = 1
                    pauli_list.append([0.25, Pauli(vp, wp)])
                else:
                    shift += 0.25

        for i in indices_has_e:
            wp = np.zeros(n)
            vp = np.zeros(n)
            vp[i] = 1
            pauli_list.append([-Y, Pauli(vp, wp)])

    return Operator(paulis=pauli_list), shift


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


def sample_most_likely(n, state_vector):
    """Compute the most likely binary string from state vector.

    Args:
        n (int): number of  qubits.
        state_vector (numpy.ndarray or dict): state vector or counts.

    Returns:
        numpy.ndarray: binary string as numpy.ndarray of ints.
    """
    if isinstance(state_vector, dict) or isinstance(state_vector, OrderedDict):
        temp_vec = np.zeros(2**n)
        total = 0
        for i in range(2**n):
            state = np.binary_repr(i, n)
            count = state_vector.get(state, 0)
            temp_vec[i] = count
            total += count
        state_vector = temp_vec / float(total)

    k = np.argmax(np.abs(state_vector))
    x = np.zeros(n)
    for i in range(n):
        x[i] = k % 2
        k >>= 1
    return x


def get_solution(x):
    """

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        numpy.ndarray: graph solution as binary numpy array.
    """
    return 1 - x


def check_solution_satisfiability(sol, list_of_subsets):
    n = len(list_of_subsets)
    U = []
    for sub in list_of_subsets:
        U.extend(sub)
    U = np.unique(U)  # U is the universe

    U2 = []
    selected_subsets = []
    for i in range(n):
        if sol[i] == 1:
            selected_subsets.append(list_of_subsets[i])
            U2.extend(list_of_subsets[i])

    U2 = np.unique(U2)
    if set(U) != set(U2):
        return False

    tmplen = len(selected_subsets)
    for i in range(tmplen):
        for j in range(i):
            L = selected_subsets[i]
            R = selected_subsets[j]

            if set(L) & set(R):  # should be empty
                return False

    return True

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

""" set packing module """

import logging
import warnings

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator

logger = logging.getLogger(__name__)


def get_operator(list_of_subsets):
    """Construct the Hamiltonian for the set packing.

    Notes:
        find the maximal number of subsets which are disjoint pairwise.

        Hamiltonian:
        H = A Ha + B Hb
        Ha = sum_{Si and Sj overlaps}{XiXj}
        Hb = -sum_{i}{Xi}

        Ha is to ensure the disjoint condition, while Hb is to achieve the maximal number.
        Ha is hard constraint that must be satisfied. Therefore A >> B.
        In the following, we set A=10 and B = 1

        where Xi = (Zi + 1)/2

    Args:
        list_of_subsets (list): list of lists (i.e., subsets)

    Returns:
        tuple(WeightedPauliOperator, float): operator for the Hamiltonian,
                                        a constant shift for the obj function.
    """
    # pylint: disable=invalid-name
    shift = 0
    pauli_list = []
    A = 10
    n = len(list_of_subsets)
    for i in range(n):
        for j in range(i):
            if set(list_of_subsets[i]) & set(list_of_subsets[j]):
                wp = np.zeros(n)
                vp = np.zeros(n)
                vp[i] = 1
                vp[j] = 1
                pauli_list.append([A*0.25, Pauli(vp, wp)])

                vp2 = np.zeros(n)
                vp2[i] = 1
                pauli_list.append([A*0.25, Pauli(vp2, wp)])

                vp3 = np.zeros(n)
                vp3[j] = 1
                pauli_list.append([A*0.25, Pauli(vp3, wp)])

                shift += A*0.25

    for i in range(n):
        wp = np.zeros(n)
        vp = np.zeros(n)
        vp[i] = 1
        pauli_list.append([-0.5, Pauli(vp, wp)])
        shift += -0.5

    return WeightedPauliOperator(paulis=pauli_list), shift


def get_solution(x):
    """

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        numpy.ndarray: graph solution as binary numpy array.
    """
    return 1 - x


def check_disjoint(sol, list_of_subsets):
    """ check disjoint """
    # pylint: disable=invalid-name
    n = len(list_of_subsets)
    selected_subsets = []
    for i in range(n):
        if sol[i] == 1:
            selected_subsets.append(list_of_subsets[i])
    tmplen = len(selected_subsets)
    for i in range(tmplen):
        for j in range(i):
            L = selected_subsets[i]
            R = selected_subsets[j]
            if set(L) & set(R):
                return False

    return True


def random_number_list(n, weight_range=100, savefile=None):
    """ random number list """
    # pylint: disable=import-outside-toplevel
    from .common import random_number_list as redirect_func
    warnings.warn("random_number_list function has been moved to "
                  "qiskit.optimization.ising.common, "
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


def sample_most_likely(n=None, state_vector=None):
    """ sample most likely """
    # pylint: disable=import-outside-toplevel
    from .common import sample_most_likely as redirect_func
    if n is not None:
        warnings.warn("n argument is not need and it will be removed after Aqua 0.7+",
                      DeprecationWarning)
    warnings.warn("sample_most_likely function has been moved to "
                  "qiskit.optimization.ising.common, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(state_vector=state_vector)


def get_set_packing_qubitops(list_of_subsets):
    """ get set packing qubit ops """
    warnings.warn("get_set_packing_qubitops function has been changed to get_operator"
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return get_operator(list_of_subsets)

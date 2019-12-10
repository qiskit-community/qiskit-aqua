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
Convert vertex cover instances into Pauli list
Deal with Gset format. See https://web.stanford.edu/~yyye/yyye/Gset/
"""

import logging
import warnings

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator

logger = logging.getLogger(__name__)


def get_operator(weight_matrix):
    r"""Generate Hamiltonian for the vertex cover
    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.

    Returns:
        tuple(WeightedPauliOperator, float): operator for the Hamiltonian and a
        constant shift for the obj function.

    Goals:
    1 color some vertices as red such that every edge is connected to some red vertex
    2 minimize the vertices to be colored as red

    Hamiltonian:
    H = A * H_A + H_B
    H_A = sum\_{(i,j)\in E}{(1-Xi)(1-Xj)}
    H_B = sum_{i}{Zi}

    H_A is to achieve goal 1 while H_b is to achieve goal 2.
    H_A is hard constraint so we place a huge penality on it. A=5.
    Note Xi = (Zi+1)/2

    """
    n = len(weight_matrix)
    pauli_list = []
    shift = 0
    a__ = 5

    for i in range(n):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                w_p = np.zeros(n)
                v_p = np.zeros(n)
                v_p[i] = 1
                v_p[j] = 1
                pauli_list.append([a__*0.25, Pauli(v_p, w_p)])

                v_p2 = np.zeros(n)
                v_p2[i] = 1
                pauli_list.append([-a__*0.25, Pauli(v_p2, w_p)])

                v_p3 = np.zeros(n)
                v_p3[j] = 1
                pauli_list.append([-a__*0.25, Pauli(v_p3, w_p)])

                shift += a__*0.25

    for i in range(n):
        w_p = np.zeros(n)
        v_p = np.zeros(n)
        v_p[i] = 1
        pauli_list.append([0.5, Pauli(v_p, w_p)])
        shift += 0.5
    return WeightedPauliOperator(paulis=pauli_list), shift


def check_full_edge_coverage(x, w):
    """
    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix.

    Returns:
        float: value of the cut.
    """
    first = w.shape[0]
    second = w.shape[1]
    for i in range(first):
        for j in range(second):
            if w[i, j] != 0:
                if x[i] != 1 and x[j] != 1:
                    return False

    return True


def get_graph_solution(x):
    """Get graph solution from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        numpy.ndarray: graph solution as binary numpy array.
    """
    return 1 - x


def random_graph(n, weight_range=10, edge_prob=0.3, savefile=None, seed=None):
    """ random graph """
    # pylint: disable=import-outside-toplevel
    from .common import random_graph as redirect_func
    warnings.warn("random_graph function has been moved to "
                  "qiskit.optimization.ising.common, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(n=n, weight_range=weight_range, edge_prob=edge_prob,
                         savefile=savefile, seed=seed)


def parse_gset_format(filename):
    """ parse gset format """
    # pylint: disable=import-outside-toplevel
    from .common import parse_gset_format as redirect_func
    warnings.warn("parse_gset_format function has been moved to "
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


def get_gset_result(x):
    """ get gset result """
    # pylint: disable=import-outside-toplevel
    from .common import get_gset_result as redirect_func
    warnings.warn("get_gset_result function has been moved to "
                  "qiskit.optimization.ising.common, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(x)


def get_vertex_cover_qubitops(weight_matrix):
    """ get vertex cover qubit ops """
    warnings.warn("get_vertex_cover_qubitops function has been changed to get_operator"
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return get_operator(weight_matrix)

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
Convert vertex cover instances into Pauli list
Deal with Gset format. See https://web.stanford.edu/~yyye/yyye/Gset/
"""

import logging

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
                pauli_list.append([a__ * 0.25, Pauli((v_p, w_p))])

                v_p2 = np.zeros(n)
                v_p2[i] = 1
                pauli_list.append([-a__ * 0.25, Pauli((v_p2, w_p))])

                v_p3 = np.zeros(n)
                v_p3[j] = 1
                pauli_list.append([-a__ * 0.25, Pauli((v_p3, w_p))])

                shift += a__ * 0.25

    for i in range(n):
        w_p = np.zeros(n)
        v_p = np.zeros(n)
        v_p[i] = 1
        pauli_list.append([0.5, Pauli((v_p, w_p))])
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

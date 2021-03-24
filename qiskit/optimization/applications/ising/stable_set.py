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
Convert stable set instances into Pauli list.  We read instances in
the Gset format, see https://web.stanford.edu/~yyye/yyye/Gset/ , for
compatibility with the maxcut format, but the weights on the edges
as they are not really used and are always assumed to be 1.  The
graph is represented by an adjacency matrix.
"""

import logging

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator

logger = logging.getLogger(__name__)


def get_operator(w):
    """Generate Hamiltonian for the maximum stable set in a graph.

    Args:
        w (numpy.ndarray) : adjacency matrix.

    Returns:
        tuple(WeightedPauliOperator, float): operator for the Hamiltonian and a
        constant shift for the obj function.

    """
    num_nodes = len(w)
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if w[i, j] != 0:
                x_p = np.zeros(num_nodes, dtype=bool)
                z_p = np.zeros(num_nodes, dtype=bool)
                z_p[i] = True
                z_p[j] = True
                pauli_list.append([1/2, Pauli((z_p, x_p))])
                shift += 1/2
    for i in range(num_nodes):
        degree = np.count_nonzero(w[i, :] != 0)
        x_p = np.zeros(num_nodes, dtype=bool)
        z_p = np.zeros(num_nodes, dtype=bool)
        z_p[i] = True
        pauli_list.append([1/2 - degree/2, Pauli((z_p, x_p))])
    return WeightedPauliOperator(paulis=pauli_list), shift - num_nodes / 2


def stable_set_value(x, w):
    """Compute the value of a stable set, and its feasibility.

    Args:
        x (numpy.ndarray): binary string in original format -- not
            graph solution!.
        w (numpy.ndarray): adjacency matrix.

    Returns:
        tuple(float, bool): size of the stable set, and Boolean indicating
            feasibility.
    """
    assert len(x) == w.shape[0]
    feasible = True
    num_nodes = w.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if w[i, j] != 0 and x[i] == 1 and x[j] == 1:
                feasible = False
                break
    return np.sum(x), feasible


def get_graph_solution(x):
    """Get graph solution from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        numpy.ndarray: graph solution as binary numpy array.
    """
    return x

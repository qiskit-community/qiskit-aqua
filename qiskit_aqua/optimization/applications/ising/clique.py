# -*- coding: utf-8 -*-

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

"""Convert clique instances into Pauli list

Deal with Gset format. See https://web.stanford.edu/~yyye/yyye/Gset/
"""

import logging

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator

logger = logging.getLogger(__name__)


def get_operator(weight_matrix, K):  # pylint: disable=invalid-name
    r"""
    Generate Hamiltonian for the clique.

    The goals is can we find a complete graph of size K?

    To build the Hamiltonian the following logic is applied.

    | Suppose Xv denotes whether v should appear in the clique (Xv=1 or 0)\n
    | H = Ha + Hb\n
    | Ha = (K-sum_{v}{Xv})\^2
    | Hb = K(K−1)/2 - sum_{(u,v)\in E}{XuXv}

    | Besides, Xv = (Zv+1)/2
    | By replacing Xv with Zv and simplifying it, we get what we want below.

    Note: in practice, we use H = A\*Ha + Bb, where A is a large constant such as 1000.

    A is like a huge penality over the violation of Ha,
    which forces Ha to be 0, i.e., you have exact K vertices selected.
    Under this assumption, Hb = 0 starts to make sense,
    it means the subgraph constitutes a clique or complete graph.
    Note the lowest possible value of Hb is 0.

    Without the above assumption, Hb may be negative (say you select all).
    In this case, one needs to use Hb\^2 in the hamiltonian to minimize the difference.

    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.
        K (numpy.ndarray): K

    Returns:
        tuple(WeightedPauliOperator, float):
            The operator for the Hamiltonian and a constant shift for the obj function.
    """
    # pylint: disable=invalid-name
    num_nodes = len(weight_matrix)
    pauli_list = []
    shift = 0

    Y = K - 0.5 * num_nodes  # Y = K - sum_{v}{1 / 2}

    A = 1000
    # Ha part:
    shift += A * Y * Y

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                xp = np.zeros(num_nodes, dtype=np.bool)
                zp = np.zeros(num_nodes, dtype=np.bool)
                zp[i] = True
                zp[j] = True
                pauli_list.append([A * 0.25, Pauli(zp, xp)])
            else:
                shift += A * 0.25
    for i in range(num_nodes):
        xp = np.zeros(num_nodes, dtype=np.bool)
        zp = np.zeros(num_nodes, dtype=np.bool)
        zp[i] = True
        pauli_list.append([-A * Y, Pauli(zp, xp)])

    shift += 0.5 * K * (K - 1)

    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                xp = np.zeros(num_nodes, dtype=np.bool)
                zp = np.zeros(num_nodes, dtype=np.bool)
                zp[i] = True
                zp[j] = True
                pauli_list.append([-0.25, Pauli(zp, xp)])

                zp2 = np.zeros(num_nodes, dtype=np.bool)
                zp2[i] = True
                pauli_list.append([-0.25, Pauli(zp2, xp)])

                zp3 = np.zeros(num_nodes, dtype=np.bool)
                zp3[j] = True
                pauli_list.append([-0.25, Pauli(zp3, xp)])

                shift += -0.25

    return WeightedPauliOperator(paulis=pauli_list), shift


def satisfy_or_not(x, w, K):  # pylint: disable=invalid-name
    """Compute the value of a cut.

    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix.
        K (numpy.ndarray): K

    Returns:
        float: value of the cut.
    """
    # pylint: disable=invalid-name
    X = np.outer(x, x)
    w_01 = np.where(w != 0, 1, 0)

    return np.sum(w_01 * X) == K * (K - 1)  # note sum() count the same edge twice


def get_graph_solution(x):
    """Get graph solution from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        numpy.ndarray: graph solution as binary numpy array.
    """
    return 1 - x

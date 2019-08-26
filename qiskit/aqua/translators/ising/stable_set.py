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
Convert stable set instances into Pauli list.  We read instances in
the Gset format, see https://web.stanford.edu/~yyye/yyye/Gset/ , for
compatibility with the maxcut format, but the weights on the edges
as they are not really used and are always assumed to be 1.  The
graph is represented by an adjacency matrix.
"""

import logging
import warnings

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator

logger = logging.getLogger(__name__)


def get_qubit_op(w):
    """Generate Hamiltonian for the maximum stable set in a graph.

    Args:
        w (numpy.ndarray) : adjacency matrix.

    Returns:
        WeightedPauliOperatorOperator, float: operator for the Hamiltonian and a
        constant shift for the obj function.

    """
    num_nodes = len(w)
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if w[i, j] != 0:
                xp = np.zeros(num_nodes, dtype=np.bool)
                zp = np.zeros(num_nodes, dtype=np.bool)
                zp[i] = True
                zp[j] = True
                pauli_list.append([1.0, Pauli(zp, xp)])
                shift += 1
    for i in range(num_nodes):
        degree = np.sum(w[i, :])
        xp = np.zeros(num_nodes, dtype=np.bool)
        zp = np.zeros(num_nodes, dtype=np.bool)
        zp[i] = True
        pauli_list.append([degree - 1/2, Pauli(zp, xp)])
    return WeightedPauliOperator(paulis=pauli_list), shift - num_nodes/2


def stable_set_value(x, w):
    """Compute the value of a stable set, and its feasibility.

    Args:
        x (numpy.ndarray): binary string in original format -- not
            graph solution!.
        w (numpy.ndarray): adjacency matrix.

    Returns:
        float, bool: size of the stable set, and Boolean indicating
            feasibility.
    """
    assert(len(x) == w.shape[0])
    feasible = True
    num_nodes = w.shape[0]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if w[i, j] != 0 and x[i] == 0 and x[j] == 0:
                feasible = False
                break
    return len(x) - np.sum(x), feasible


def get_graph_solution(x):
    """Get graph solution from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        numpy.ndarray: graph solution as binary numpy array.
    """
    return 1 - x


def random_graph(n, edge_prob=0.5, savefile=None, seed=None):
    from .common import random_graph as redirect_func
    warnings.warn("random_graph function has been moved to "
                  "qiskit.aqua.translators.ising.common, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(n=n, weight_range=2, edge_prob=edge_prob,
                         negative_weight=False, savefile=savefile, seed=seed)


def parse_gset_format(filename):
    from .common import parse_gset_format as redirect_func
    warnings.warn("parse_gset_format function has been moved to "
                  "qiskit.aqua.translators.ising.common, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(filename)


def sample_most_likely(state_vector):
    from .common import sample_most_likely as redirect_func
    warnings.warn("sample_most_likely function has been moved to "
                  "qiskit.aqua.translators.ising.common, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(state_vector=state_vector)


def get_stable_set_qubitops(w):
    warnings.warn("get_stable_set_qubitops function has been changed to get_qubit_op"
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return get_qubit_op(w)

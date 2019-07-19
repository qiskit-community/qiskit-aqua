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

# Convert graph partitioning instances into Pauli list
# Deal with Gset format. See https://web.stanford.edu/~yyye/yyye/Gset/


import logging
from collections import OrderedDict

import numpy as np
import numpy.random as rand

from qiskit.quantum_info import Pauli
from qiskit.aqua import Operator

logger = logging.getLogger(__name__)


def random_graph(n, weight_range=10, edge_prob=0.3, savefile=None, seed=None):
    """Generate random Erdos-Renyi graph.

    Args:
        n (int): number of nodes.
        weight_range (int): weights will be smaller than this value,
            in absolute value.
        edge_prob (float): probability of edge appearing.
        savefile (str or None): name of file where to save graph.
        seed (int or None): random seed - if None, will not initialize.

    Returns:
        numpy.ndarray: adjacency matrix (with weights).

    """
    assert(weight_range >= 0)
    if seed:
        rand.seed(seed)
    w = np.zeros((n, n))
    m = 0
    for i in range(n):
        for j in range(i+1, n):
            if rand.rand() <= edge_prob:
                w[i, j] = rand.randint(1, weight_range)
                if rand.rand() >= 0.5:
                    w[i, j] *= -1
                m += 1
    w += w.T
    if savefile:
        with open(savefile, 'w') as outfile:
            outfile.write('{} {}\n'.format(n, m))
            for i in range(n):
                for j in range(i+1, n):
                    if w[i, j] != 0:
                        outfile.write('{} {} {}\n'.format(i + 1, j + 1, w[i, j]))
    return w


def get_clique_qubitops(weight_matrix, K):
    r"""
    Generate Hamiltonian for the clique

    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.

    Returns:
        operator.Operator, float: operator for the Hamiltonian and a
        constant shift for the obj function.

    Goals:
        can we find a complete graph of size K?

    Hamiltonian:
    suppose Xv denotes whether v should appear in the clique (Xv=1 or 0)
    H = Ha + Hb
    Ha = (K-sum_{v}{Xv})^2
    Hb = K(K−1)/2 􏰏- sum_{(u,v)\in E}{XuXv}

    Besides, Xv = (Zv+1)/2
    By replacing Xv with Zv and simplifying it, we get what we want below.

    Note: in practice, we use H = A*Ha + Bb,
        where A is a large constant such as 1000.
    A is like a huge penality over the violation of Ha,
    which forces Ha to be 0, i.e., you have exact K vertices selected.
    Under this assumption, Hb = 0 starts to make sense,
    it means the subgraph constitutes a clique or complete graph.
    Note the lowest possible value of Hb is 0.

    Without the above assumption, Hb may be negative (say you select all).
    In this case, one needs to use Hb^2 in the hamiltonian to minimize the difference.
    """
    num_nodes = len(weight_matrix)
    pauli_list = []
    shift = 0

    Y = K - 0.5*num_nodes  # Y = K-sum_{v}{1/2}

    A = 1000
    # Ha part:
    shift += A*Y*Y

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                xp = np.zeros(num_nodes, dtype=np.bool)
                zp = np.zeros(num_nodes, dtype=np.bool)
                zp[i] = True
                zp[j] = True
                pauli_list.append([A*0.25, Pauli(zp, xp)])
            else:
                shift += A*0.25
    for i in range(num_nodes):
        xp = np.zeros(num_nodes, dtype=np.bool)
        zp = np.zeros(num_nodes, dtype=np.bool)
        zp[i] = True
        pauli_list.append([-A*Y, Pauli(zp, xp)])

    shift += 0.5*K*(K-1)

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

    return Operator(paulis=pauli_list), shift


def parse_gset_format(filename):
    """Read graph in Gset format from file.

    Args:
        filename (str): name of the file.

    Returns:
        numpy.ndarray: adjacency matrix as a 2D numpy array.
    """
    n = -1
    with open(filename) as infile:
        header = True
        m = -1
        count = 0
        for line in infile:
            v = map(lambda e: int(e), line.split())
            if header:
                n, m = v
                w = np.zeros((n, n))
                header = False
            else:
                s, t, x = v
                s -= 1  # adjust 1-index
                t -= 1  # ditto
                w[s, t] = t
                count += 1
        assert m == count
    w += w.T
    return w


def satisfy_or_not(x, w, K):
    """Compute the value of a cut.

    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix.

    Returns:
        float: value of the cut.
    """
    X = np.outer(x, x)
    w_01 = np.where(w != 0, 1, 0)

    return np.sum(w_01 * X) == K*(K-1)  # note sum() count the same edge twice


def get_graph_solution(x):
    """Get graph solution from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        numpy.ndarray: graph solution as binary numpy array.
    """
    return 1 - x


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

    k = np.argmax(state_vector)
    x = np.zeros(n)
    for i in range(n):
        x[i] = k % 2
        k >>= 1
    return x


def get_gset_result(x):
    """Get graph solution in Gset format from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        Dict[int, int]: graph solution in Gset format.
    """
    return {i + 1: 1 - x[i] for i in range(len(x))}

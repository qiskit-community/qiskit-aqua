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


def get_vertex_cover_qubitops(weight_matrix):
    r"""Generate Hamiltonian for the vertex cover
    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.

    Returns:
        operator.Operator, float: operator for the Hamiltonian and a
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
    A = 5

    for i in range(n):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                wp = np.zeros(n)
                vp = np.zeros(n)
                vp[i] = 1
                vp[j] = 1
                pauli_list.append([A*0.25, Pauli(vp, wp)])

                vp2 = np.zeros(n)
                vp2[i] = 1
                pauli_list.append([-A*0.25, Pauli(vp2, wp)])

                vp3 = np.zeros(n)
                vp3[j] = 1
                pauli_list.append([-A*0.25, Pauli(vp3, wp)])

                shift += A*0.25

    for i in range(n):
        wp = np.zeros(n)
        vp = np.zeros(n)
        vp[i] = 1
        pauli_list.append([0.5, Pauli(vp, wp)])
        shift += 0.5
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


def sample_most_likely(n, state_vector):
    """Compute the most likely binary string from state vector.
    Args:
        state_vector (numpy.ndarray or dict): state vector or counts.
    Returns:
        numpy.ndarray: binary string as numpy.ndarray of ints.
    """
    if isinstance(state_vector, dict) or isinstance(state_vector, OrderedDict):
        # get the binary string with the largest count
        binary_string = sorted(state_vector.items(), key=lambda kv: kv[1])[-1][0]
        x = np.asarray([int(y) for y in list(binary_string)])
        return x
    else:
        n = int(np.log2(state_vector.shape[0]))
        k = np.argmax(np.abs(state_vector))
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

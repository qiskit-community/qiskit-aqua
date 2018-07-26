# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Convert stable set instances into Pauli list.  We read instances in
# the Gset format, see https://web.stanford.edu/~yyye/yyye/Gset/ , for
# compatibility with the maxcut format, but the weights on the edges
# as they are not really used and are always assumed to be 1.  The
# graph is represented by an adjacency matrix.


import logging
from collections import OrderedDict

import numpy as np
import numpy.random as rand
from qiskit.tools.qi.pauli import Pauli

from qiskit_aqua import Operator

logger = logging.getLogger(__name__)


def random_graph(n, edge_prob = 0.5, savefile=None):
    """Generate a random Erdos-Renyi graph on n nodes.

    Args:
        n (int): number of nodes.
        edge_prob (float): probability of edge appearing.
        savefile (str or None): write graph to this file.

    Returns:
        numpy.ndarray: adjacency matrix (with weights).
    """
    w = np.zeros((n, n))
    m = 0
    for i in range(n):
        for j in range(i+1, n):
            if rand.rand() <= edge_prob:
                w[i, j] = 1
                m += 1
    w += w.T

    if savefile:
        with open(savefile, 'w') as outfile:
            outfile.write('{} {}\n'.format(n, m))
            for i in range(n):
                for j in range(i+1, n):
                    if w[i, j] != 0:
                        outfile.write('{} {} {}\n'.format(i + 1, j + 1, t))
    return w


def get_stableset_qubitops(w):
    """Generate Hamiltonian for the maximum stableset in a graph.

    Args:
        w (numpy.ndarray) : adjacency matrix.

    Returns:
        operator.Operator, float: operator for the Hamiltonian and a
        constant shift for the obj function.

    """
    num_nodes = len(w)
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if (w[i, j] != 0):
                wp = np.zeros(num_nodes)
                vp = np.zeros(num_nodes)
                vp[i] = 1
                vp[j] = 1
                pauli_list.append([1.0, Pauli(vp, wp)])
                shift += 1
    for i in range(num_nodes):
        degree = sum(w[i, :])
        wp = np.zeros(num_nodes)
        vp = np.zeros(num_nodes)
        vp[i] = 1
        pauli_list.append([degree - 1/2, Pauli(vp, wp)])
    return Operator(paulis=pauli_list), shift - num_nodes/2


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
                w[s, t] = 1 if x != 0 else 0
                count += 1
        assert m == count
    w += w.T
    return w

def stableset_value(x, w):
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



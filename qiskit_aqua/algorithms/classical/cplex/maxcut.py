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

# Convert maxcut instances into Pauli list
# Deal with Gset format. See https://web.stanford.edu/~yyye/yyye/Gset/
# Design the maxcut object `w` as a two-dimensional np.array
# e.g., w[i, j] = x means that the weight of a edge between i and j is x
# Note that the weights are symmetric, i.e., w[j, i] = x always holds.

import logging
from collections import OrderedDict

import numpy as np
import numpy.random as rand
from qiskit.tools.qi.pauli import Pauli

from qiskit_aqua import Operator

logger = logging.getLogger(__name__)


def random_graph(n, weight_range=10, edge_prob=0.3, savefile=None,
                  seed=None):
    """Generate random Erdos-Renyi graph for MaxCut.

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


def get_maxcut_qubitops(weight_matrix):
    """Generate Hamiltonian for the maximum stableset in a graph.

    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.

    Returns:
        operator.Operator, float: operator for the Hamiltonian and a
        constant shift for the obj function.

    """
    num_nodes = weight_matrix.shape[0]
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(i):
            if (weight_matrix[i,j] != 0):
                wp = np.zeros(num_nodes)
                vp = np.zeros(num_nodes)
                vp[i] = 1
                vp[j] = 1
                pauli_list.append([0.5 * weight_matrix[i, j], Pauli(vp, wp)])
                shift -= 0.5 * weight_matrix[i, j]
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

def maxcut_value(x, w):
    """Compute the value of a cut.

    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix.

    Returns:
        float: value of the cut.
    """
    X = np.outer(x, (1 - x))
    return np.sum(w * X)

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

def get_gset_result(x):
    """Get graph solution in Gset format from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        Dict[int, int]: graph solution in Gset format.
    """
    return {i + 1: 1 - x[i] for i in range(len(x))}

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

# Convert symmetric TSP instances into Pauli list
# Deal with TSPLIB format.
# See https://wwwproxy.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/
# and http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html
# Design the tsp object `w` as a two-dimensional np.array
# e.g., w[i, j] = x means that the length of a edge between i and j is x
# Note that the weights are symmetric, i.e., w[j, i] = x always holds.

import logging
from collections import OrderedDict, namedtuple

import numpy as np
import numpy.random as rand
from qiskit.tools.qi.pauli import Pauli

from qiskit_acqua import Operator

logger = logging.getLogger(__name__)


TspData = namedtuple('TspData', 'name dim coord w')


def calc_distance(coord, name='tmp'):
    assert coord.shape[1] == 2
    dim = coord.shape[0]
    w = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i + 1, dim):
            delta = coord[i] - coord[j]
            w[i, j] = np.hypot(delta[0], delta[1])
    w += w.T
    return TspData(name=name, dim=dim, coord=coord, w=w)


def random_tsp(n, low=0, high=100, savefile=None, seed=None, name='tmp'):
    """Generate a random instance for TSP.

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
    assert n > 0
    if seed:
        rand.seed(seed)
    coord = rand.uniform(low, high, (n, 2))
    ins = calc_distance(coord, name)
    if savefile:
        with open(savefile, 'w') as outfile:
            outfile.write('NAME : {}\n'.format(ins.name))
            outfile.write('COMMENT : random data\n')
            outfile.write('TYPE : TSP\n')
            outfile.write('DIMENSION : {}\n'.format(ins.dim))
            outfile.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
            outfile.write('NODE_COORD_SECTION\n')
            for i in range(ins.dim):
                x = ins.coord[i]
                outfile.write('{} {:.4f} {:.4f}\n'.format(i + 1, x[0], x[1]))
    return ins


def parse_tsplib_format(filename):
    """Read graph in TSPLIB format from file.

    Args:
        filename (str): name of the file.

    Returns:
        numpy.ndarray: adjacency matrix as a 2D numpy array.
    """
    name = ''
    coord = None
    with open(filename) as infile:
        coord_section = False
        for line in infile:
            if line.startswith('NAME'):
                name = line.split(':')[1]
                name.strip()
            elif line.startswith('TYPE'):
                typ = line.split(':')[1]
                typ.strip()
                if typ != 'TSP':
                    logger.warning('This supports only "TSP" type. Actual: {}'.format(typ))
            elif line.startswith('DIMENSION'):
                dim = int(line.split(':')[1])
                coord = np.zeros((dim, 2))
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                typ = line.split(':')[1]
                typ.strip()
                if typ != 'EUC_2D':
                    logger.warning('This supports only "EUC_2D" edge weight. Actual: {}'.format(typ))
            elif line.startswith('NODE_COORD_SECTION'):
                coord_section = True
            elif coord_section:
                v = line.split()
                index = int(v[0]) - 1
                coord[index][0] = float(v[1])
                coord[index][1] = float(v[2])
    return calc_distance(coord, name)


def get_tsp_qubitops(ins):
    """Generate Hamiltonian for TSP of a graph.

    Args:
        ins (TspData) : TSP data including coordinates and distances.

    Returns:
        operator.Operator, float: operator for the Hamiltonian and a
        constant shift for the obj function.

    """
    num_nodes = ins.dim
    num_qubits = num_nodes ** 2
    zero = np.zeros(num_qubits)
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(i):
            for p in range(num_nodes):
                q = (p + 1) % num_nodes
                vp = np.zeros(num_qubits)
                wp = np.zeros(num_qubits)
                vp[i * num_nodes + p] = 1
                wp[j * num_nodes + q] = 1
                pauli_list.append((ins.w[i, j] / 4, Pauli(vp, wp)))
                pauli_list.append([-ins.w[i, j] / 4, Pauli(vp, zero)])  # use list intentionally
                pauli_list.append([-ins.w[i, j] / 4, Pauli(wp, zero)])  # use list intentionally
            shift += num_nodes * ins.w[i, j] / 4
    coef = 1e5
    for i in range(num_nodes):
        for p in range(num_nodes):
            vp = np.zeros(num_qubits)
            vp[i * num_nodes + p] = 1
            pauli_list.append((2 * coef, Pauli(vp, vp)))
            pauli_list.append((-4 * coef, Pauli(vp, zero)))
    for p in range(num_nodes):
        for i in range(num_nodes):
            for j in range(i):
                vp = np.zeros(num_qubits)
                wp = np.zeros(num_qubits)
                vp[i * num_nodes + p] = 1
                wp[j * num_nodes + p] = 1
                pauli_list.append((2 * coef, Pauli(vp, wp)))
    for i in range(num_nodes):
        for p in range(num_nodes):
            for q in range(p):
                vp = np.zeros(num_qubits)
                wp = np.zeros(num_qubits)
                vp[i * num_nodes + p] = 1
                wp[i * num_nodes + q] = 1
                pauli_list.append((2 * coef, Pauli(vp, wp)))
    shift += 2 * coef * num_nodes
    return Operator(paulis=pauli_list), shift


def tsp_value(x, w):
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


def sample_most_likely(n, state_vector):
    """Compute the most likely binary string from state vector.

    Args:
        n (int): number of  qubits.
        state_vector (numpy.ndarray or dict): state vector or counts.

    Returns:
        numpy.ndarray: binary string as numpy.ndarray of ints.
    """
    if isinstance(state_vector, dict) or isinstance(state_vector, OrderedDict):
        temp_vec = np.zeros(2 ** n)
        total = 0
        for i in range(2 ** n):
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

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

""" Convert symmetric TSP instances into Pauli list
Deal with TSPLIB format. It supports only EUC_2D edge weight type.
See https://wwwproxy.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/
and http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html
Design the tsp object `w` as a two-dimensional np.array
e.g., w[i, j] = x means that the length of a edge between i and j is x
Note that the weights are symmetric, i.e., w[j, i] = x always holds.
"""

import logging
from collections import OrderedDict, namedtuple

import numpy as np
import numpy.random as rand
from qiskit.tools.qi.pauli import Pauli

from qiskit_aqua import Operator

logger = logging.getLogger(__name__)

"""Instance data of TSP"""
TspData = namedtuple('TspData', 'name dim coord w')


def calc_distance(coord, name='tmp'):
    assert coord.shape[1] == 2
    dim = coord.shape[0]
    w = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i + 1, dim):
            delta = coord[i] - coord[j]
            w[i, j] = np.rint(np.hypot(delta[0], delta[1]))
    w += w.T
    return TspData(name=name, dim=dim, coord=coord, w=w)


def random_tsp(n, low=0, high=100, savefile=None, seed=None, name='tmp'):
    """Generate a random instance for TSP.

    Args:
        n (int): number of nodes.
        low (float): lower bound of coordinate.
        high (float): uppper bound of coordinate.
        savefile (str or None): name of file where to save graph.
        seed (int or None): random seed - if None, will not initialize.
        name (str): name of an instance

    Returns:
        TspData: instance data.

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
        TspData: instance data.

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


def get_tsp_qubitops(ins, penalty=1e5):
    """Generate Hamiltonian for TSP of a graph.

    Args:
        ins (TspData) : TSP data including coordinates and distances.
        penalty (float) : Penalty coefficient for the constraints

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
        for j in range(num_nodes):
            if i == j:
                continue
            for p in range(num_nodes):
                q = (p + 1) % num_nodes
                shift += ins.w[i, j] / 4

                vp = np.zeros(num_qubits)
                vp[i * num_nodes + p] = 1
                pauli_list.append([-ins.w[i, j] / 4, Pauli(vp, zero)])

                vp = np.zeros(num_qubits)
                vp[j * num_nodes + q] = 1
                pauli_list.append([-ins.w[i, j] / 4, Pauli(vp, zero)])

                vp = np.zeros(num_qubits)
                vp[i * num_nodes + p] = 1
                vp[j * num_nodes + q] = 1
                pauli_list.append([ins.w[i, j] / 4, Pauli(vp, zero)])

    for i in range(num_nodes):
        for p in range(num_nodes):
            vp = np.zeros(num_qubits)
            vp[i * num_nodes + p] = 1
            pauli_list.append([penalty, Pauli(vp, zero)])
            shift += -penalty

    for p in range(num_nodes):
        for i in range(num_nodes):
            for j in range(i):
                shift += penalty / 2

                vp = np.zeros(num_qubits)
                vp[i * num_nodes + p] = 1
                pauli_list.append([-penalty / 2, Pauli(vp, zero)])

                vp = np.zeros(num_qubits)
                vp[j * num_nodes + p] = 1
                pauli_list.append([-penalty / 2, Pauli(vp, zero)])

                vp = np.zeros(num_qubits)
                vp[i * num_nodes + p] = 1
                vp[j * num_nodes + p] = 1
                pauli_list.append([penalty / 2, Pauli(vp, zero)])

    for i in range(num_nodes):
        for p in range(num_nodes):
            for q in range(p):
                shift += penalty / 2

                vp = np.zeros(num_qubits)
                vp[i * num_nodes + p] = 1
                pauli_list.append([-penalty / 2, Pauli(vp, zero)])

                vp = np.zeros(num_qubits)
                vp[i * num_nodes + q] = 1
                pauli_list.append([-penalty / 2, Pauli(vp, zero)])

                vp = np.zeros(num_qubits)
                vp[i * num_nodes + p] = 1
                vp[i * num_nodes + q] = 1
                pauli_list.append([penalty / 2, Pauli(vp, zero)])
    shift += 2 * penalty * num_nodes
    return Operator(paulis=pauli_list), shift


def tsp_value(z, w):
    """Compute the TSP value of a solution.

    Args:
        z (list[int]): list of cities.
        w (numpy.ndarray): adjacency matrix.

    Returns:
        float: value of the cut.
    """
    ret = 0.0
    for i in range(len(z) - 1):
        ret += w[z[i], z[i + 1]]
    ret += w[z[-1], z[0]]
    return ret


def tsp_feasible(x):
    """Check whether a solution is feasible or not.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        bool: feasible or not.
    """
    n = int(np.sqrt(len(x)))
    y = np.zeros((n, n))
    for i in range(n):
        for p in range(n):
            y[i, p] = x[i * n + p]
    for i in range(n):
        if sum(y[i, p] for p in range(n)) != 1:
            return False
    for p in range(n):
        if sum(y[i, p] for i in range(n)) != 1:
            return False
    return True


def get_tsp_solution(x):
    """Get graph solution from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        list[int]: sequence of cities to traverse.
    """
    n = int(np.sqrt(len(x)))
    z = []
    for p in range(n):
        for i in range(n):
            if x[i * n + p] >= 0.999:
                assert len(z) == p
                z.append(i)
    return z


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

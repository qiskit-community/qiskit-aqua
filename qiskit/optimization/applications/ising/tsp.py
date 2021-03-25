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

""" Convert symmetric TSP instances into Pauli list
Deal with TSPLIB format. It supports only EUC_2D edge weight type.
See https://wwwproxy.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/
and http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html
Design the tsp object `w` as a two-dimensional np.array
e.g., w[i, j] = x means that the length of a edge between i and j is x
Note that the weights are symmetric, i.e., w[j, i] = x always holds.
"""

import logging
from collections import namedtuple

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua import aqua_globals
from qiskit.aqua.operators import WeightedPauliOperator

logger = logging.getLogger(__name__)

"""Instance data of TSP"""
TspData = namedtuple('TspData', 'name dim coord w')


def calc_distance(coord, name='tmp'):
    """ calculate distance """
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
        high (float): upper bound of coordinate.
        savefile (str or None): name of file where to save graph.
        seed (int or None): random seed - if None, will not initialize.
        name (str): name of an instance

    Returns:
        TspData: instance data.

    """
    assert n > 0
    if seed:
        aqua_globals.random_seed = seed

    coord = aqua_globals.random.uniform(low, high, (n, 2))
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
    coord = []
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
                    logger.warning('This supports only "TSP" type. Actual: %s', typ)
            elif line.startswith('DIMENSION'):
                dim = int(line.split(':')[1])
                coord = np.zeros((dim, 2))
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                typ = line.split(':')[1]
                typ.strip()
                if typ != 'EUC_2D':
                    logger.warning('This supports only "EUC_2D" edge weight. Actual: %s', typ)
            elif line.startswith('NODE_COORD_SECTION'):
                coord_section = True
            elif coord_section:
                v = line.split()
                index = int(v[0]) - 1
                coord[index][0] = float(v[1])
                coord[index][1] = float(v[2])
    return calc_distance(coord, name)


def get_operator(ins, penalty=1e5):
    """Generate Hamiltonian for TSP of a graph.

    Args:
        ins (TspData) : TSP data including coordinates and distances.
        penalty (float) : Penalty coefficient for the constraints

    Returns:
        tuple(WeightedPauliOperator, float): operator for the Hamiltonian and a
        constant shift for the obj function.

    """
    num_nodes = ins.dim
    num_qubits = num_nodes ** 2
    zero = np.zeros(num_qubits, dtype=bool)
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            for p__ in range(num_nodes):
                q = (p__ + 1) % num_nodes
                shift += ins.w[i, j] / 4

                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[i * num_nodes + p__] = True
                pauli_list.append([-ins.w[i, j] / 4, Pauli((z_p, zero))])

                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[j * num_nodes + q] = True
                pauli_list.append([-ins.w[i, j] / 4, Pauli((z_p, zero))])

                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[i * num_nodes + p__] = True
                z_p[j * num_nodes + q] = True
                pauli_list.append([ins.w[i, j] / 4, Pauli((z_p, zero))])

    for i in range(num_nodes):
        for p__ in range(num_nodes):
            z_p = np.zeros(num_qubits, dtype=bool)
            z_p[i * num_nodes + p__] = True
            pauli_list.append([penalty, Pauli((z_p, zero))])
            shift += -penalty

    for p__ in range(num_nodes):
        for i in range(num_nodes):
            for j in range(i):
                shift += penalty / 2

                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[i * num_nodes + p__] = True
                pauli_list.append([-penalty / 2, Pauli((z_p, zero))])

                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[j * num_nodes + p__] = True
                pauli_list.append([-penalty / 2, Pauli((z_p, zero))])

                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[i * num_nodes + p__] = True
                z_p[j * num_nodes + p__] = True
                pauli_list.append([penalty / 2, Pauli((z_p, zero))])

    for i in range(num_nodes):
        for p__ in range(num_nodes):
            for q in range(p__):
                shift += penalty / 2

                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[i * num_nodes + p__] = True
                pauli_list.append([-penalty / 2, Pauli((z_p, zero))])

                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[i * num_nodes + q] = True
                pauli_list.append([-penalty / 2, Pauli((z_p, zero))])

                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[i * num_nodes + p__] = True
                z_p[i * num_nodes + q] = True
                pauli_list.append([penalty / 2, Pauli((z_p, zero))])
    shift += 2 * penalty * num_nodes
    return WeightedPauliOperator(paulis=pauli_list), shift


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
        for p__ in range(n):
            y[i, p__] = x[i * n + p__]
    for i in range(n):
        if sum(y[i, p] for p in range(n)) != 1:
            return False
    for p__ in range(n):
        if sum(y[i, p__] for i in range(n)) != 1:
            return False
    return True


def get_tsp_solution(x):
    """Get graph solution from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        list[int]: sequence of cities to traverse.
            The i-th item in the list corresponds to the city which is visited in the i-th step.
            The list for an infeasible answer e.g. [[0,1],1,] can be interpreted as
            visiting [city0 and city1] as the first city, then visit city1 as the second city,
            then visit no where as the third city).
    """
    n = int(np.sqrt(len(x)))
    z = []
    for p__ in range(n):
        p_th_step = []
        for i in range(n):
            if x[i * n + p__] >= 0.999:
                p_th_step.append(i)
        if len(p_th_step) == 1:
            z.extend(p_th_step)
        else:
            z.append(p_th_step)
    return z

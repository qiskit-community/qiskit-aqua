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

"""
Convert max-cut instances into Pauli list
Deal with Gset format. See https://web.stanford.edu/~yyye/yyye/Gset/
Design the max-cut object `w` as a two-dimensional np.array
e.g., w[i, j] = x means that the weight of a edge between i and j is x
Note that the weights are symmetric, i.e., w[j, i] = x always holds.
"""

import logging

import numpy as np
from docplex.mp.model import Model

from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.optimization import QuadraticProgram

logger = logging.getLogger(__name__)


def get_operator(weight_matrix):
    """Generate Hamiltonian for the max-cut problem of a graph.

    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.

    Returns:
        WeightedPauliOperator: operator for the Hamiltonian
        float: a constant shift for the obj function.

    """
    num_nodes = weight_matrix.shape[0]
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                x_p = np.zeros(num_nodes, dtype=np.bool)
                z_p = np.zeros(num_nodes, dtype=np.bool)
                z_p[i] = True
                z_p[j] = True
                pauli_list.append([0.5 * weight_matrix[i, j], Pauli(z_p, x_p)])
                shift -= 0.5 * weight_matrix[i, j]
    return WeightedPauliOperator(paulis=pauli_list), shift


def max_cut_value(x, w):
    """Compute the value of a cut.

    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix.

    Returns:
        float: value of the cut.
    """
    # pylint: disable=invalid-name
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


# todo: review location of this function
def cut_value(x: np.ndarray, w: np.ndarray):
    """Compute the value of a cut.

    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix.

    Returns:
        float: value of the cut.
    """
    # pylint: disable=invalid-name
    return np.dot((1 - x), np.dot(w, x))


def max_cut_qp(adjacency_matrix: np.array) -> QuadraticProgram:
    """
    Creates the max-cut instance based on the adjacency graph.
    """

    size = len(adjacency_matrix)

    # todo: should we use DOCplex here?
    mdl = Model()
    x = [mdl.binary_var('x%s' % i) for i in range(size)]

    objective_terms = []
    for i in range(size):
        for j in range(size):
            if adjacency_matrix[i, j] != 0.:
                objective_terms.append(
                    adjacency_matrix[i, j] * x[i] * (1 - x[j]))

    objective = mdl.sum(objective_terms)
    mdl.maximize(objective)

    qp = QuadraticProgram()
    qp.from_docplex(mdl)

    return qp

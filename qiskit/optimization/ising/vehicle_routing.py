# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Converts vehicle routing instances into a list of Paulis,
and provides some related routines (extracting a solution,
checking its objective function value).
"""

import warnings

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator


def get_vehiclerouting_matrices(instance, n, K):  # pylint: disable=invalid-name
    """Constructs auxiliary matrices from a vehicle routing instance,
        which represent the encoding into a binary quadratic program.
        This is used in the construction of the qubit ops and computation
        of the solution cost.

    Args:
        instance (numpy.ndarray) : a customers-to-customers distance matrix.
        n (integer) : the number of customers.
        K (integer) : the number of vehicles available.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray, float):
            a matrix defining the interactions between variables.
            a matrix defining the contribution from the individual variables.
            the constant offset.
    """
    # pylint: disable=invalid-name
    # N = (n - 1) * n
    A = np.max(instance) * 100  # A parameter of cost function

    # Determine the weights w
    instance_vec = instance.reshape(n**2)
    w_list = [instance_vec[x] for x in range(n**2) if instance_vec[x] > 0]
    w = np.zeros(n * (n - 1))
    for i_i, _ in enumerate(w_list):
        w[i_i] = w_list[i_i]

    # Some additional variables
    id_n = np.eye(n)
    im_n_1 = np.ones([n - 1, n - 1])
    iv_n_1 = np.ones(n)
    iv_n_1[0] = 0
    iv_n = np.ones(n - 1)
    neg_iv_n_1 = np.ones(n) - iv_n_1

    v = np.zeros([n, n * (n - 1)])
    for i_i in range(n):
        count = i_i - 1
        for j_j in range(n * (n - 1)):

            if j_j // (n - 1) == i_i:
                count = i_i

            if j_j // (n - 1) != i_i and j_j % (n - 1) == count:
                v[i_i][j_j] = 1.

    v_n = np.sum(v[1:], axis=0)

    # Q defines the interactions between variables
    Q = A * (np.kron(id_n, im_n_1) + np.dot(v.T, v))

    # g defines the contribution from the individual variables
    g = w - 2 * A * (np.kron(iv_n_1, iv_n) + v_n.T) - \
        2 * A * K * (np.kron(neg_iv_n_1, iv_n) + v[0].T)

    # c is the constant offset
    c = 2 * A * (n - 1) + 2 * A * (K**2)

    return (Q, g, c)


def get_vehiclerouting_cost(instance, n, K, x_sol):  # pylint: disable=invalid-name
    """Computes the cost of a solution to an instance of a vehicle routing problem.

    Args:
        instance (numpy.ndarray) : a customers-to-customers distance matrix.
        n (integer) : the number of customers.
        K (integer) : the number of vehicles available.
        x_sol (numpy.ndarray): a solution, i.e., a path, in its binary representation.

    Returns:
        float: objective function value.
    """
    # pylint: disable=invalid-name
    (Q, g, c) = get_vehiclerouting_matrices(instance, n, K)

    def fun(x):
        return np.dot(np.around(x), np.dot(Q, np.around(x))) + np.dot(g, np.around(x)) + c
    cost = fun(x_sol)
    return cost


def get_operator(instance, n, K):  # pylint: disable=invalid-name
    """Converts an instance of a vehicle routing problem into a list of Paulis.

    Args:
        instance (numpy.ndarray) : a customers-to-customers distance matrix.
        n (integer) : the number of customers.
        K (integer) : the number of vehicles available.

    Returns:
        WeightedPauliOperator: operator for the Hamiltonian.
    """
    # pylint: disable=invalid-name
    N = (n - 1) * n
    (Q, g__, c) = get_vehiclerouting_matrices(instance, n, K)

    # Defining the new matrices in the Z-basis
    i_v = np.ones(N)
    q_z = (Q / 4)
    g_z = (-g__ / 2 - np.dot(i_v, Q / 4) - np.dot(Q / 4, i_v))
    c_z = (c + np.dot(g__ / 2, i_v) + np.dot(i_v, np.dot(Q / 4, i_v)))

    c_z = c_z + np.trace(q_z)
    q_z = q_z - np.diag(np.diag(q_z))

    # Getting the Hamiltonian in the form of a list of Pauli terms

    pauli_list = []
    for i in range(N):
        if g_z[i] != 0:
            w_p = np.zeros(N)
            v_p = np.zeros(N)
            v_p[i] = 1
            pauli_list.append((g_z[i], Pauli(v_p, w_p)))
    for i in range(N):
        for j in range(i):
            if q_z[i, j] != 0:
                w_p = np.zeros(N)
                v_p = np.zeros(N)
                v_p[i] = 1
                v_p[j] = 1
                pauli_list.append((2 * q_z[i, j], Pauli(v_p, w_p)))

    pauli_list.append((c_z, Pauli(np.zeros(N), np.zeros(N))))
    return WeightedPauliOperator(paulis=pauli_list)


def get_vehiclerouting_solution(instance, n, K, result):  # pylint: disable=invalid-name
    """Tries to obtain a feasible solution (in vector form) of an instance
        of vehicle routing from the results dictionary.

    Args:
        instance (numpy.ndarray) : a customers-to-customers distance matrix.
        n (integer) : the number of customers.
        K (integer) : the number of vehicles available.
        result (dictionary) : a dictionary obtained by QAOA.run or VQE.run containing key 'eigvecs'.

    Returns:
        numpy.ndarray: a solution, i.e., a path, in its binary representation.

    #TODO: support statevector simulation, results should be a statevector or counts format, not
           a result from algorithm run
    """
    # pylint: disable=invalid-name
    del instance, K  # unused
    v = result['eigvecs'][0]
    N = (n - 1) * n

    index_value = [x for x in range(len(v)) if v[x] == max(v)][0]
    string_value = "{0:b}".format(index_value)

    while len(string_value) < N:
        string_value = '0' + string_value

    x_sol = list()
    for elements in string_value:
        if elements == '0':
            x_sol.append(0)
        else:
            x_sol.append(1)

    x_sol = np.flip(x_sol, axis=0)

    return x_sol


def get_vehiclerouting_qubitops(instance, n, K):
    """ get vehicle routing qubit ops """
    # pylint: disable=invalid-name
    warnings.warn("get_vehiclerouting_qubitops function has been changed to get_operator"
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return get_operator(instance, n, K)

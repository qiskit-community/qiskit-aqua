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

# Converts vehicle routing instnces into a list of Paulis,
# and provides some related routines (extracting a solution,
# checking its objective function value).

from collections import OrderedDict

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.aqua import Operator


def get_vehiclerouting_matrices(instance, n, K):
    """Constructs auxiliary matrices from a vehicle routing instance,
        which represent the encoding into a binary quadratic program.
        This is used in the construction of the qubit ops and computation
        of the solution cost.

    Args:
        instance (numpy.ndarray) : a customers-to-customers distance matrix.
        n (integer) : the number of customers.
        K (integer) : the number of vehicles available.

    Returns:
        Q (numpy.ndarray) : a matrix defining the interactions between variables.
        g (numpy.ndarray) : a matrix defining the contribution from the individual variables.
        c (float) : the constant offset.
        """

    N = (n - 1) * n
    A = np.max(instance) * 100  # A parameter of cost function

    # Determine the weights w
    instance_vec = instance.reshape(n**2)
    w_list = [instance_vec[x] for x in range(n**2) if instance_vec[x] > 0]
    w = np.zeros(n * (n - 1))
    for ii in range(len(w_list)):
        w[ii] = w_list[ii]

    # Some additional variables
    Id_n = np.eye(n)
    Im_n_1 = np.ones([n - 1, n - 1])
    Iv_n_1 = np.ones(n)
    Iv_n_1[0] = 0
    Iv_n = np.ones(n - 1)
    neg_Iv_n_1 = np.ones(n) - Iv_n_1

    v = np.zeros([n, n * (n - 1)])
    for ii in range(n):
        count = ii - 1
        for jj in range(n * (n - 1)):

            if jj // (n - 1) == ii:
                count = ii

            if jj // (n - 1) != ii and jj % (n - 1) == count:
                v[ii][jj] = 1.

    vn = np.sum(v[1:], axis=0)

    # Q defines the interactions between variables
    Q = A * (np.kron(Id_n, Im_n_1) + np.dot(v.T, v))

    # g defines the contribution from the individual variables
    g = w - 2 * A * (np.kron(Iv_n_1,Iv_n) + vn.T) - \
            2 * A * K * (np.kron(neg_Iv_n_1, Iv_n) + v[0].T)

    # c is the constant offset
    c = 2 * A * (n - 1) + 2 * A * (K**2)

    return (Q, g, c)


def get_vehiclerouting_cost(instance, n, K, x_sol):
    """Computes the cost of a solution to an instnance of a vehicle routing problem.

    Args:
        instance (numpy.ndarray) : a customers-to-customers distance matrix.
        n (integer) : the number of customers.
        K (integer) : the number of vehicles available.
        x_sol (numpy.ndarray): a solution, i.e., a path, in its binary representation.

    Returns:
        cost (float): objective function value.
        """
    (Q, g, c) = get_vehiclerouting_matrices(instance, n, K)
    fun = lambda x: np.dot(np.around(x), np.dot(Q, np.around(x))) + np.dot(
        g, np.around(x)) + c
    cost = fun(x_sol)
    return cost


def get_vehiclerouting_qubitops(instance, n, K):
    """Converts an instnance of a vehicle routing problem into a list of Paulis.

    Args:
        instance (numpy.ndarray) : a customers-to-customers distance matrix.
        n (integer) : the number of customers.
        K (integer) : the number of vehicles available.

    Returns:
        operator.Operator: operator for the Hamiltonian.
        """

    N = (n - 1) * n
    (Q, g, c) = get_vehiclerouting_matrices(instance, n, K)

    # Defining the new matrices in the Z-basis
    Iv = np.ones(N)
    Qz = (Q / 4)
    gz = (-g / 2 - np.dot(Iv, Q / 4) - np.dot(Q / 4, Iv))
    cz = (c + np.dot(g / 2, Iv) + np.dot(Iv, np.dot(Q / 4, Iv)))

    cz = cz + np.trace(Qz)
    Qz = Qz - np.diag(np.diag(Qz))

    # Getting the Hamiltonian in the form of a list of Pauli terms

    pauli_list = []
    for i in range(N):
        if gz[i] != 0:
            wp = np.zeros(N)
            vp = np.zeros(N)
            vp[i] = 1
            pauli_list.append((gz[i], Pauli(vp, wp)))
    for i in range(N):
        for j in range(i):
            if Qz[i, j] != 0:
                wp = np.zeros(N)
                vp = np.zeros(N)
                vp[i] = 1
                vp[j] = 1
                pauli_list.append((2 * Qz[i, j], Pauli(vp, wp)))

    pauli_list.append((cz, Pauli(np.zeros(N), np.zeros(N))))
    return Operator(paulis=pauli_list)


def get_vehiclerouting_solution(instance, n, K, result):
    """Tries to obtain a feasible solution (in vector form) of an instnance 
        of vehicle routing from the results dictionary.

    Args:
        instance (numpy.ndarray) : a customers-to-customers distance matrix.
        n (integer) : the number of customers.
        K (integer) : the number of vehicles available.
        result (dictionary) : a dictionary obtained by QAOA.run or VQE.run containing key 'eigvecs'.

    Returns:
        x_sol (numpy.ndarray): a solution, i.e., a path, in its binary representation.
        """

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

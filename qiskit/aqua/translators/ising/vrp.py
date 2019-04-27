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

# Converts vehicle routing instnces into a list of Paulis

from collections import OrderedDict

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.aqua import Operator


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
        A = np.max(instance) * 100  # A parameter of cost function

        # Determine the weights w
        instance_vec = instance.reshape(n ** 2)
        w_list = [instance_vec[x] for x in range(n ** 2) if instance_vec[x] > 0]
        w = np.zeros(n * (n - 1))
        for ii in range(len(w_list)):
            w[ii] = w_list[ii]

        # Some additional variables
        Id_n = np.eye(n)
        Im_n_1 = np.ones([n - 1, n - 1])
        Iv_n_1 = np.ones(n)
        Iv_n_1[0] = 0
        Iv_n = np.ones(n-1)
        neg_Iv_n_1 = np.ones(n) - Iv_n_1

        v = np.zeros([n, n*(n-1)])
        for ii in range(n):
            count = ii-1
            for jj in range(n*(n-1)):

                if jj//(n-1) == ii:
                    count = ii

                if jj//(n-1) != ii and jj%(n-1) == count:
                    v[ii][jj] = 1.

        vn = np.sum(v[1:], axis=0)

        # Q defines the interactions between variables
        Q = A*(np.kron(Id_n, Im_n_1) + np.dot(v.T, v))

        # g defines the contribution from the individual variables
        g = w - 2 * A * (np.kron(Iv_n_1,Iv_n) + vn.T) - \
                2 * A * K * (np.kron(neg_Iv_n_1, Iv_n) + v[0].T)

        # c is the constant offset
        c = 2 * A * (n-1) + 2 * A * (K ** 2)

        try:
            max(x_sol)
            # Evaluates the cost distance from a binary representation of a path
            fun = lambda x: np.dot(np.around(x), np.dot(Q, np.around(x))) + np.dot(g, np.around(x)) + c
            cost = fun(x_sol)
        except:
            cost = 0

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

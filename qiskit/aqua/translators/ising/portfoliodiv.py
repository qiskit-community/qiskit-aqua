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

# Convert portfolio optimization instances into Pauli list

from collections import OrderedDict

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit_aqua import Operator


def get_portfoliodiversification_qubitops(rho,n,q,max_trials=1000):
        instance = rho

        #N = (n + 1) * n  # number of qubits
        N = n**2 + n

        A = np.max(np.abs(rho)) * 1000  # A parameter of cost function

        # Determine the weights w
        instance_vec = rho.reshape(n ** 2)

        # quadratic term Q
        q0 = np.zeros([N,1])
        Q1 = np.zeros([N,N])
        Q2 = np.zeros([N,N])
        Q3 = np.zeros([N, N])

        for x in range(n**2,n**2+n):
            q0[x] = 1

        Q0 = A*np.dot(q0,q0.T)
        for ii in range(0,n):
            v0 = np.zeros([N,1])
            for jj in range(n*ii,n*(ii+1)):
                v0[jj] = 1
            Q1 = Q1 + np.dot(v0,v0.T)
        Q1 = A*Q1

        for jj in range(0,n):
            v0 = np.zeros([N,1])
            v0[n*jj+jj] = 1
            v0[n**2+jj] = -1
            Q2 = Q2 + np.dot(v0, v0.T)
        Q2 = A*Q2


        for ii in range(0, n):
            for jj in range(0,n):
                Q3[ii*n + jj, n**2+jj] = -0.5
                Q3[n ** 2 + jj,ii * n + jj] = -0.5


        Q3 = A * Q3

        Q = Q0+Q1+Q2+Q3

        # linear term c:
        c0 = np.zeros(N)
        c1 = np.zeros(N)
        c2 = np.zeros(N)
        c3 = np.zeros(N)

        for x in range(n**2):
            c0[x] = instance_vec[x]
        for x in range(n**2,n**2+n):
            c1[x] = -2*A*q
        for x in range(n**2):
            c2[x] = -2*A
        for x in range(n**2):
            c3[x] = A

        g = c0+c1+c2+c3

        # constant term r
        c = A*(q**2 + n)

        try:
            max(x_sol)
            # Evaluates the cost distance from a binary representation
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


def get_portfolio_qubitops(mu, sigma, q, budget, penalty):

    # get problem dimension
    n = len(mu)
    e = np.ones(n)
    E = np.matmul(np.asmatrix(e).T, np.asmatrix(e))

    # map problem to Ising model
    offset = - np.dot(mu, e)/2 + penalty*budget**2 - budget*n*penalty + n**2*penalty/4 + q/4*np.dot(e, np.dot(sigma, e))
    mu_z = mu/2 + budget*penalty*e - n*penalty/2*e - q/2*np.dot(sigma, e)
    sigma_z = penalty/4*E + q/4*sigma

    # construct operator
    pauli_list = []
    for i in range(n):
        i_ = i
        # i_ = n-i-1
        if np.abs(mu_z[i_]) > 1e-6:
            xp = np.zeros(n, dtype=np.bool)
            zp = np.zeros(n, dtype=np.bool)
            zp[i_] = True
            pauli_list.append([mu_z[i_], Pauli(zp, xp)])
        for j in range(i):
            j_ = j
            # j_ = n-j-1
            if np.abs(sigma_z[i_, j_]) > 1e-6:
                xp = np.zeros(n, dtype=np.bool)
                zp = np.zeros(n, dtype=np.bool)
                zp[i_] = True
                zp[j_] = True
                pauli_list.append([2*sigma_z[i_, j_], Pauli(zp, xp)])
        offset += sigma_z[i_, i_]

    return Operator(paulis=pauli_list), offset


def portfolio_value(x, mu, sigma, q, budget, penalty):
    return q * np.dot(x, np.dot(sigma, x)) - np.dot(mu, x) + penalty * pow(sum(x) - budget, 2)


def portfolio_expected_value(x, mu):
    return np.dot(mu, x)


def portfolio_variance(x, sigma):
    return np.dot(x, np.dot(sigma, x))


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
        x = np.zeros(n, dtype=int)
        for i in range(n):
            x[i] = k % 2
            k >>= 1
        return x
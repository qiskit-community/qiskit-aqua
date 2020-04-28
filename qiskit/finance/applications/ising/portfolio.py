# -*- coding: utf-8 -*-

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
Convert portfolio optimization instances into Pauli list
"""

import numpy as np
from sklearn.datasets import make_spd_matrix
from qiskit.quantum_info import Pauli

from qiskit.aqua import aqua_globals
from qiskit.aqua.operators import WeightedPauliOperator


def random_model(n, seed=None):
    """Generate random model (mu, sigma) for portfolio optimization problem.

    Args:
        n (int): number of assets.
        seed (int or None): random seed - if None, will not initialize.

    Returns:
        numpy.narray: expected return vector
        numpy.ndarray: covariance matrix

    """
    if seed:
        aqua_globals.random_seed = seed

    # draw random return values between [0, 1]
    m_u = aqua_globals.random.uniform(size=n, low=0, high=1)

    # construct positive semi-definite covariance matrix
    sigma = make_spd_matrix(n)

    return m_u, sigma


def get_operator(mu, sigma, q, budget, penalty):  # pylint: disable=invalid-name
    """ get qubit op """
    # pylint: disable=invalid-name
    # get problem dimension
    n = len(mu)
    e = np.ones(n)
    E = np.matmul(np.asmatrix(e).T, np.asmatrix(e))

    # map problem to Ising model
    offset = -1 * np.dot(mu, e) / 2 + penalty * budget ** 2 - \
        budget * n * penalty + n ** 2 * penalty / 4 + q / 4 * np.dot(e, np.dot(sigma, e))
    mu_z = mu / 2 + budget * penalty * e - n * penalty / 2 * e - q / 2 * np.dot(sigma, e)
    sigma_z = penalty / 4 * E + q / 4 * sigma

    # construct operator
    pauli_list = []
    for i in range(n):
        i_ = i
        # i_ = n - i - 1
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
                pauli_list.append([2 * sigma_z[i_, j_], Pauli(zp, xp)])
        offset += sigma_z[i_, i_]

    return WeightedPauliOperator(paulis=pauli_list), offset


def portfolio_value(x, mu, sigma, q, budget, penalty):  # pylint: disable=invalid-name
    """ returns portfolio value """
    return q * np.dot(x, np.dot(sigma, x)) - np.dot(mu, x) + penalty * pow(sum(x) - budget, 2)


def portfolio_expected_value(x, mu):  # pylint: disable=invalid-name
    """ returns portfolio expected value """
    return np.dot(mu, x)


def portfolio_variance(x, sigma):
    """ returns portfolio variance """
    return np.dot(x, np.dot(sigma, x))
